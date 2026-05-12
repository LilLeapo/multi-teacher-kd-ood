"""OOD scoring + standard metrics (AUROC, AUPR, FPR@95TPR)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


@torch.no_grad()
def compute_ood_scores(logits: torch.Tensor, scores: List[str], energy_T: float = 1.0) -> Dict[str, np.ndarray]:
    """Return per-sample ID-likeness scores (higher = more ID-like)."""
    out: Dict[str, np.ndarray] = {}
    if "msp" in scores:
        out["msp"] = F.softmax(logits, dim=-1).max(dim=-1).values.cpu().numpy()
    if "maxlogit" in scores:
        out["maxlogit"] = logits.max(dim=-1).values.cpu().numpy()
    if "energy" in scores:
        out["energy"] = (energy_T * torch.logsumexp(logits / energy_T, dim=-1)).cpu().numpy()
    return out


@torch.no_grad()
def score_dataset(model, loader, device, scores: List[str], energy_T: float = 1.0) -> Dict[str, np.ndarray]:
    model.eval()
    bufs: Dict[str, list] = {s: [] for s in scores}
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        logits = model(x)
        s = compute_ood_scores(logits, scores, energy_T=energy_T)
        for k, v in s.items():
            bufs[k].append(v)
    return {k: np.concatenate(v, axis=0) for k, v in bufs.items()}


def fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """FPR when TPR on ID samples reaches `target_tpr`. Higher score = ID-like."""
    threshold = np.quantile(id_scores, 1.0 - target_tpr)
    return float((ood_scores >= threshold).mean())


def ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "aupr_in": float(average_precision_score(y_true, y_score)),
        "aupr_out": float(average_precision_score(1 - y_true, -y_score)),
        "fpr95": fpr_at_tpr(id_scores, ood_scores, 0.95),
    }
