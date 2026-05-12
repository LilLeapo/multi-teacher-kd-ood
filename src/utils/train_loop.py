"""Generic training/eval helpers shared by teacher and distillation entry points."""
from __future__ import annotations

import math
import os
import random
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    t = cfg["train"]
    return torch.optim.SGD(
        model.parameters(),
        lr=t["lr"],
        momentum=t["momentum"],
        weight_decay=t["weight_decay"],
        nesterov=t.get("nesterov", True),
    )


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


def train_supervised(
    model: nn.Module,
    train_loader,
    test_loader,
    cfg: dict,
    ckpt_path: str,
    extra_step_fn: Callable | None = None,
) -> dict:
    """Standard supervised CE training with cosine schedule + AMP.

    Saves best-acc checkpoint at `ckpt_path`. Returns a small log dict.
    """
    device = cfg["device"]
    seed_everything(cfg.get("seed", 42))
    model.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = GradScaler(enabled=cfg["train"].get("amp", True))
    ls = cfg["train"].get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)

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
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["train"].get("amp", True)):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
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
