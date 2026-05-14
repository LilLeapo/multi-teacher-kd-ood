"""Logit- and feature-based OOD scorers. All scores return ID-likeness:
higher = more in-distribution, lower = more out-of-distribution.

Logit-based (cheap, no fitting needed):
  msp, maxlogit, energy, gen
Feature-based (need fit-on-ID-train step):
  KNNScorer, MahalanobisScorer, ViMScorer
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------- logit-only scorers ----------

def score_msp(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1).max(dim=-1).values


def score_maxlogit(logits: torch.Tensor) -> torch.Tensor:
    return logits.max(dim=-1).values


def score_energy(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    return T * torch.logsumexp(logits / T, dim=-1)


def score_gen(logits: torch.Tensor, gamma: float = 0.1, top_m: int | None = None) -> torch.Tensor:
    """GEN: Liu et al., NeurIPS 2023. Higher = more ID.

    Returns -sum_i p_i^gamma (1 - p_i)^gamma over top-M classes.
    """
    p = F.softmax(logits, dim=-1)
    if top_m is not None and top_m < p.size(-1):
        p = p.topk(top_m, dim=-1).values
    return -(p.pow(gamma) * (1 - p).pow(gamma)).sum(dim=-1)


# ---------- feature-based scorers ----------

@dataclass
class KNNScorer:
    """Sun et al. ICML 2022 — distance to k-th nearest training neighbour in
    L2-normalised feature space. We use cosine similarity (= 1 - 0.5*L2^2 on
    unit sphere), so the top-k *largest* similarity gives the k-th nearest.
    """
    train_feats: torch.Tensor | None = None  # (N, D) normalised
    k: int = 50

    def fit(self, train_features: torch.Tensor) -> None:
        self.train_feats = F.normalize(train_features, dim=-1)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        assert self.train_feats is not None, "KNNScorer.fit(...) first"
        f = F.normalize(features, dim=-1)
        # Cosine similarity in chunks to keep memory bounded
        chunk = 4096
        out = []
        for i in range(0, f.size(0), chunk):
            sim = f[i : i + chunk] @ self.train_feats.T  # (chunk, N_train)
            topk = sim.topk(self.k, dim=-1).values  # (chunk, k)
            out.append(topk[:, -1])  # k-th-largest similarity = -distance (up to const)
        return torch.cat(out, dim=0)


@dataclass
class MahalanobisScorer:
    """Lee et al. NeurIPS 2018 — negative min Mahalanobis distance to the
    closest class-conditional Gaussian mean (shared covariance)."""
    means: torch.Tensor | None = None  # (C, D)
    cov_inv: torch.Tensor | None = None  # (D, D)

    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor, num_classes: int) -> None:
        D = train_features.size(1)
        means = torch.stack([
            train_features[train_labels == c].mean(dim=0) for c in range(num_classes)
        ])  # (C, D)
        centred = train_features - means[train_labels]
        cov = centred.T @ centred / (train_features.size(0) - num_classes)
        cov = cov + 1e-4 * torch.eye(D, device=cov.device, dtype=cov.dtype)
        self.means = means
        self.cov_inv = torch.linalg.inv(cov)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        assert self.means is not None
        # m_c(x) = (x - mu_c)^T S^-1 (x - mu_c); compute in chunks
        chunk = 1024
        out = []
        for i in range(0, features.size(0), chunk):
            diff = features[i : i + chunk].unsqueeze(1) - self.means.unsqueeze(0)  # (B, C, D)
            m = torch.einsum("bcd,de,bce->bc", diff, self.cov_inv, diff)  # (B, C)
            out.append(-m.min(dim=-1).values)
        return torch.cat(out, dim=0)


@dataclass
class ViMScorer:
    """Wang et al. CVPR 2022 — virtual logit matching.

    Shift features by u = -W_pinv @ b so the bias-free classifier is centred,
    keep the top-K principal subspace of training features, score by residual
    norm in the orthogonal complement (calibrated against logit magnitude).
    """
    u: torch.Tensor | None = None             # (D,)
    principal: torch.Tensor | None = None     # (D, K) orthonormal
    alpha: float | None = None

    def fit(
        self,
        train_features: torch.Tensor,
        train_logits: torch.Tensor,
        classifier_W: torch.Tensor,
        classifier_b: torch.Tensor,
        principal_dim: int = 512,
    ) -> None:
        D = train_features.size(1)
        K = min(principal_dim, D)
        u = -torch.linalg.pinv(classifier_W) @ classifier_b  # (D,)
        f_c = train_features + u  # (N, D)
        cov = f_c.T @ f_c / f_c.size(0)
        # eigh: ascending eigenvalues
        _, vecs = torch.linalg.eigh(cov)
        principal = vecs[:, -K:].contiguous()  # (D, K)
        # Residual norm on training
        proj = f_c @ principal              # (N, K)
        f_recon = proj @ principal.T        # (N, D)
        p_train = torch.norm(f_c - f_recon, dim=-1)  # (N,)
        max_logit_train = train_logits.max(dim=-1).values  # (N,)
        alpha = (max_logit_train.sum() / p_train.sum()).item()
        self.u, self.principal, self.alpha = u, principal, alpha

    def score(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        assert self.principal is not None
        f_c = features + self.u
        proj = f_c @ self.principal
        f_recon = proj @ self.principal.T
        p = torch.norm(f_c - f_recon, dim=-1)
        virtual = self.alpha * p
        return torch.logsumexp(logits, dim=-1) - virtual
