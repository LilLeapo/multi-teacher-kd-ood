"""Multi-teacher KD on a student model.

Teachers are loaded from per-arch checkpoints and run live (eval mode) on each
augmented batch — keeps targets consistent with the student's augmented input
without needing the cached-logits-on-eval-transform trick.

Usage:
    python -m src.distill --teacher-config configs/teachers.yaml \\
                          --student-config configs/students.yaml \\
                          --student shufflenetv2_x0_5
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from .data import build_id_loaders
from .models import build_model
from .utils import load_config
from .utils.kd_losses import combined_loss
from .utils.train_loop import build_optimizer, build_scheduler, evaluate, seed_everything


def _load_teachers(teacher_cfg: dict, names: List[str], device: str) -> Tuple[List[nn.Module], torch.Tensor]:
    teachers, accs = [], []
    for name in names:
        spec = next(t for t in teacher_cfg["teachers"] if t["name"] == name)
        model = build_model(
            spec["arch"],
            num_classes=teacher_cfg["num_classes"],
            cifar_stem=spec.get("cifar_stem", True),
        )
        ckpt_path = os.path.join(teacher_cfg["ckpt_root"], teacher_cfg["id_dataset"], "teachers", f"{name}.pt")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"])
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        teachers.append(model)
        accs.append(float(state.get("acc", 0.0)))
        print(f"  loaded teacher {name} (val_acc={accs[-1]*100:.2f}%)")
    return teachers, torch.tensor(accs, device=device)


def _build_weights(scheme: str, accs: torch.Tensor) -> torch.Tensor:
    if scheme == "uniform":
        w = torch.ones_like(accs)
    elif scheme == "accuracy":
        w = accs.clone()
    else:
        raise ValueError(f"Unsupported teacher_weighting={scheme}. Use 'uniform' or 'accuracy'.")
    return w / w.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-config", required=True)
    p.add_argument("--student-config", required=True)
    p.add_argument("--student", required=True)
    args = p.parse_args()

    s_cfg = load_config(args.student_config)
    t_cfg = load_config(args.teacher_config)
    s_spec = next(s for s in s_cfg["students"] if s["name"] == args.student)

    # Pull distill-specific overrides into the canonical train block.
    distill = s_cfg["distill"]
    s_cfg["train"] = {**s_cfg["train"], **{k: v for k, v in distill.items() if k in s_cfg["train"]}}

    device = s_cfg["device"]
    seed_everything(s_cfg.get("seed", 42))

    train_loader, test_loader = build_id_loaders(s_cfg)
    student = build_model(s_spec["arch"], num_classes=s_cfg["num_classes"], cifar_stem=s_spec.get("cifar_stem", True)).to(device)

    print(f"Loading {len(distill['teachers'])} teachers:")
    teachers, accs = _load_teachers(t_cfg, distill["teachers"], device)
    weights = _build_weights(s_cfg["kd"].get("teacher_weighting", "uniform"), accs)
    print(f"teacher weights = {weights.cpu().tolist()}")

    alpha = distill.get("alpha", s_cfg["kd"]["alpha"])
    T = distill.get("temperature", s_cfg["kd"]["temperature"])
    ls = s_cfg["train"].get("label_smoothing", 0.0)

    optimizer = build_optimizer(student, s_cfg)
    scheduler = build_scheduler(optimizer, s_cfg, steps_per_epoch=len(train_loader))
    scaler = GradScaler(enabled=s_cfg["train"].get("amp", True))

    ckpt_path = os.path.join(s_cfg["ckpt_root"], s_cfg["id_dataset"], "students", f"{s_spec['name']}_kd.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    log_path = os.path.join(s_cfg["log_root"], s_cfg["id_dataset"], "students", f"{s_spec['name']}_kd.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    best_acc = 0.0
    history = []
    epochs = s_cfg["train"]["epochs"]
    for epoch in range(epochs):
        student.train()
        t0 = time.time()
        ce_run = kd_run = total_n = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=s_cfg["train"].get("amp", True)):
                with torch.no_grad():
                    teacher_logits = [t(x) for t in teachers]
                s_logits = student(x)
                loss, ce_v, kd_v = combined_loss(
                    s_logits, y, teacher_logits, alpha=alpha, T=T, weights=weights, label_smoothing=ls
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            n = x.size(0)
            ce_run += ce_v.item() * n
            kd_run += kd_v.item() * n
            total_n += n

        acc = evaluate(student, test_loader, device)
        rec = {
            "epoch": epoch,
            "ce": ce_run / total_n,
            "kd": kd_run / total_n,
            "test_acc": acc,
            "time": time.time() - t0,
        }
        history.append(rec)
        print(f"[epoch {epoch+1}/{epochs}] ce={rec['ce']:.4f} kd={rec['kd']:.4f} acc={acc*100:.2f}% ({rec['time']:.1f}s)", flush=True)

        if acc > best_acc:
            best_acc = acc
            torch.save({"model": student.state_dict(), "acc": acc, "epoch": epoch}, ckpt_path)

    with open(log_path, "w") as f:
        json.dump({"best_acc": best_acc, "teachers": distill["teachers"], "history": history}, f, indent=2)
    print(f"[done] {s_spec['name']} best_acc={best_acc*100:.2f}% saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
