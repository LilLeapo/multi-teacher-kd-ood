"""Generic training/eval helpers shared by teacher and distillation entry points."""
from __future__ import annotations

import math
import os
import random
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _param_groups(model: nn.Module, weight_decay: float):
    """Split params into decayed (ndim>=2) and undecayed (biases / norm scalars).

    Standard practice for AdamW-trained vision transformers / ConvNeXt: weight
    decay on weight matrices only, not on biases or LayerNorm/BatchNorm scale.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    t = cfg["train"]
    opt_name = t.get("optimizer", "sgd").lower()

    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=t["lr"],
            momentum=t["momentum"],
            weight_decay=t["weight_decay"],
            nesterov=t.get("nesterov", True),
        )

    if opt_name == "adamw":
        groups = _param_groups(model, weight_decay=t["weight_decay"])
        return torch.optim.AdamW(
            groups,
            lr=t["lr"],
            betas=tuple(t.get("betas", (0.9, 0.999))),
            eps=t.get("eps", 1e-8),
        )

    raise ValueError(f"Unknown optimizer {opt_name}. Use 'sgd' or 'adamw'.")


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    t = cfg["train"]
    warmup_epochs = t.get("warmup_epochs", 0)
    total_epochs = t["epochs"]

    if t["scheduler"] == "cosine":
        def lr_lambda(epoch_float: float) -> float:
            if epoch_float < warmup_epochs:
                return max(epoch_float / max(1, warmup_epochs), 1e-3)
            progress = (epoch_float - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step / steps_per_epoch))

    if t["scheduler"] == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.5 * total_epochs * steps_per_epoch), int(0.75 * total_epochs * steps_per_epoch)],
            gamma=0.1,
        )

    raise ValueError(f"Unknown scheduler {t['scheduler']}")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.size(0)
    return correct / max(1, total)


class BatchMixer:
    """Stochastic batch-level MixUp/CutMix using torchvision.transforms.v2.

    Configured via `cfg["augment"]["mixup"]` and `cfg["augment"]["cutmix"]`.
    Each step independently chooses one of {identity, mixup, cutmix} according
    to the probabilities; when neither is enabled the mixer is a no-op and
    leaves integer labels untouched.

    Output labels are one-hot soft targets when mixed; integer when not — use
    `soft_target_cross_entropy` downstream and check `is_soft(y)`.
    """

    def __init__(self, num_classes: int, augment_cfg: dict):
        self.num_classes = num_classes
        mixup_cfg = (augment_cfg or {}).get("mixup", {}) or {}
        cutmix_cfg = (augment_cfg or {}).get("cutmix", {}) or {}
        self.mixup_alpha = float(mixup_cfg.get("alpha", 0.0))
        self.mixup_prob = float(mixup_cfg.get("prob", 0.5))
        self.cutmix_alpha = float(cutmix_cfg.get("alpha", 0.0))
        self.cutmix_prob = float(cutmix_cfg.get("prob", 0.5))

        self._mixup = None
        self._cutmix = None
        if self.mixup_alpha > 0.0:
            from torchvision.transforms.v2 import MixUp
            self._mixup = MixUp(alpha=self.mixup_alpha, num_classes=num_classes)
        if self.cutmix_alpha > 0.0:
            from torchvision.transforms.v2 import CutMix
            self._cutmix = CutMix(alpha=self.cutmix_alpha, num_classes=num_classes)

    @property
    def enabled(self) -> bool:
        return self._mixup is not None or self._cutmix is not None

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if not self.enabled:
            return x, y
        r = random.random()
        if self._mixup is not None and r < self.mixup_prob:
            return self._mixup(x, y)
        # second window covers cutmix
        if self._cutmix is not None and r < self.mixup_prob + self.cutmix_prob:
            return self._cutmix(x, y)
        return x, y


def soft_target_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CE for either integer (B,) or soft (B, C) targets."""
    if target.ndim == 1:
        return F.cross_entropy(logits, target)
    return -(F.log_softmax(logits, dim=-1) * target).sum(dim=-1).mean()


def train_supervised(
    model: nn.Module,
    train_loader,
    test_loader,
    cfg: dict,
    ckpt_path: str,
    extra_step_fn: Callable | None = None,
) -> dict:
    """Standard supervised training with cosine schedule + AMP.

    Optional MixUp / CutMix (configured under `augment.mixup` / `augment.cutmix`)
    produces soft targets and disables label_smoothing for that batch.

    Saves best-acc checkpoint at `ckpt_path`. Returns a small log dict.
    """
    device = cfg["device"]
    seed_everything(cfg.get("seed", 42))
    model.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = GradScaler(enabled=cfg["train"].get("amp", True))
    ls = cfg["train"].get("label_smoothing", 0.0)
    hard_loss = nn.CrossEntropyLoss(label_smoothing=ls)

    num_classes = cfg.get("num_classes")
    mixer = BatchMixer(num_classes=num_classes, augment_cfg=cfg.get("augment", {}))
    if mixer.enabled:
        print(
            f"[mix] MixUp(alpha={mixer.mixup_alpha}, p={mixer.mixup_prob}) "
            f"CutMix(alpha={mixer.cutmix_alpha}, p={mixer.cutmix_prob})",
            flush=True,
        )

    grad_clip = cfg["train"].get("grad_clip", 0.0)
    best_acc = 0.0
    history = []
    epochs = cfg["train"]["epochs"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x, y = mixer(x, y)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["train"].get("amp", True)):
                logits = model(x)
                if y.ndim == 1:
                    loss = hard_loss(logits, y)
                else:
                    loss = soft_target_cross_entropy(logits, y)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)
        acc = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "loss": train_loss, "test_acc": acc, "time": elapsed})
        print(f"[epoch {epoch+1}/{epochs}] loss={train_loss:.4f} test_acc={acc*100:.2f}% ({elapsed:.1f}s)", flush=True)

        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "acc": acc, "epoch": epoch}, ckpt_path)

        if extra_step_fn is not None:
            extra_step_fn(epoch, model)

    return {"best_acc": best_acc, "history": history}
