"""Router networks for sample-level teacher gating.

RouterMLP                   — 2-layer MLP feature → teacher-mixture logits, used by CSR-KD.
SelectiveResidualRouterMLP  — adds an r(x) head for sample-level intervention strength, used by SRR.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterMLP(nn.Module):
    def __init__(self, in_dim: int, num_experts: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        self.in_dim = int(in_dim)
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(dropout)),
                nn.Linear(self.hidden_dim, self.num_experts),
            )
        else:
            self.net = nn.Linear(self.in_dim, self.num_experts)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 2:
            raise ValueError(f"feat must be rank-2 [B,D], got shape={tuple(feat.shape)}")
        return self.net(feat)

    def probs(self, feat: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        return F.softmax(self.forward(feat) / float(tau), dim=1)


class SelectiveResidualRouterMLP(nn.Module):
    """Router with an extra intervention head r(x) ∈ [r_min, r_max] for SRR."""

    def __init__(
        self,
        in_dim: int,
        num_experts: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        intervention_prior: float = 0.2,
    ):
        super().__init__()
        self.gate = RouterMLP(in_dim, num_experts, hidden_dim, dropout)
        if hidden_dim > 0:
            self.r_head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(dropout)),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.r_head = nn.Linear(in_dim, 1)
        self._init_intervention_head(float(intervention_prior))

    def _init_intervention_head(self, prior: float):
        prior = min(max(prior, 1e-4), 1.0 - 1e-4)
        bias = torch.logit(torch.tensor(prior, dtype=torch.float32)).item()
        last = self.r_head[-1] if isinstance(self.r_head, nn.Sequential) else self.r_head
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, bias)

    def forward(self, feat: torch.Tensor):
        return self.gate(feat), self.r_head(feat).squeeze(1)

    def probs(self, feat: torch.Tensor, tau: float = 1.0):
        gate_logits, r_logit = self.forward(feat)
        return F.softmax(gate_logits / float(tau), dim=1), torch.sigmoid(r_logit)
