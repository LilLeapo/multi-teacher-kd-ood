"""Knowledge distillation losses."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    """Standard KL-based KD with temperature T. Scaled by T^2 (Hinton et al.)."""
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def multi_teacher_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits_list: Iterable[torch.Tensor],
    T: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average KL against each teacher.

    `weights` can be:
        None          → uniform (1/M, ..., 1/M)
        shape (M,)    → global teacher weights, must sum to 1
        shape (B, M)  → sample-level teacher weights, each row sums to 1
    """
    teacher_logits_list = list(teacher_logits_list)
    M = len(teacher_logits_list)
    B, C = student_logits.shape
    if weights is None:
        weights = student_logits.new_full((M,), 1.0 / M)

    if weights.ndim == 1:
        loss = student_logits.new_zeros(())
        for w, tl in zip(weights, teacher_logits_list):
            loss = loss + w * kd_loss(student_logits, tl, T)
        return loss

    # Per-sample weights: aggregate teacher distributions first, then a single KL.
    if weights.shape != (B, M):
        raise ValueError(f"per-sample weights must have shape (B={B}, M={M}), got {tuple(weights.shape)}")
    s_log = F.log_softmax(student_logits / T, dim=-1)  # (B, C)
    t_probs = torch.stack(
        [F.softmax(tl / T, dim=-1) for tl in teacher_logits_list], dim=1
    )  # (B, M, C)
    mixed = (weights.unsqueeze(-1) * t_probs).sum(dim=1)  # (B, C)
    mixed = mixed.clamp_min(1e-12)
    # KL(P || Q) = sum P log P - sum P log Q  (mean over batch)
    kl = (mixed * (mixed.log() - s_log)).sum(dim=-1).mean()
    return kl * (T * T)


def combined_loss(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    teacher_logits_list: Iterable[torch.Tensor],
    alpha: float,
    T: float,
    weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(student_logits, targets, label_smoothing=label_smoothing)
    kd = multi_teacher_kd_loss(student_logits, teacher_logits_list, T, weights)
    total = (1 - alpha) * ce + alpha * kd
    return total, ce.detach(), kd.detach()
