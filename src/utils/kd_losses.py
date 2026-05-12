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
    """Average KL against each teacher. If `weights` is given, it must sum to 1."""
    teacher_logits_list = list(teacher_logits_list)
    if weights is None:
        weights = student_logits.new_full((len(teacher_logits_list),), 1.0 / len(teacher_logits_list))
    loss = student_logits.new_zeros(())
    for w, tl in zip(weights, teacher_logits_list):
        loss = loss + w * kd_loss(student_logits, tl, T)
    return loss


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
