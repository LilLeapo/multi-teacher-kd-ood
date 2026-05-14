"""Train a single teacher from scratch.

Usage:
    python -m src.train_teacher --config configs/teachers.yaml --teacher resnet50

Idempotency:
    If the target checkpoint already exists, training is skipped. Set the env
    var FORCE_RETRAIN=1 to overwrite. The CLI flag --force has the same effect.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from .data import build_id_loaders
from .models import build_model
from .utils import load_config
from .utils.train_loop import train_supervised


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "y"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--teacher", required=True, help="teacher `name` from the config")
    p.add_argument("--force", action="store_true", help="retrain even if checkpoint exists")
    args = p.parse_args()

    cfg = load_config(args.config, override_name=args.teacher)
    spec = next(t for t in cfg["teachers"] if t["name"] == args.teacher)
    ckpt_path = os.path.join(cfg["ckpt_root"], cfg["id_dataset"], "teachers", f"{spec['name']}.pt")

    force = args.force or _truthy_env("FORCE_RETRAIN")
    if os.path.isfile(ckpt_path) and not force:
        try:
            state = torch.load(ckpt_path, map_location="cpu")
            acc = float(state.get("acc", 0.0))
            print(f"[skip] {spec['name']}: checkpoint exists ({ckpt_path}), best_acc={acc*100:.2f}%. "
                  "Set FORCE_RETRAIN=1 (or pass --force) to retrain.")
            return
        except Exception as e:
            print(f"[warn] {spec['name']}: checkpoint at {ckpt_path} unreadable ({e}); retraining.")

    train_loader, test_loader = build_id_loaders(cfg)
    model = build_model(
        spec["arch"],
        num_classes=cfg["num_classes"],
        cifar_stem=spec.get("cifar_stem", True),
        model_kwargs=spec.get("model_kwargs"),
    )

    log = train_supervised(model, train_loader, test_loader, cfg, ckpt_path)

    log_dir = os.path.join(cfg["log_root"], cfg["id_dataset"], "teachers")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"{spec['name']}.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"[done] {spec['name']} best_acc={log['best_acc']*100:.2f}% saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
