"""Train a single teacher from scratch.

Usage:
    python -m src.train_teacher --config configs/teachers.yaml --teacher resnet50
"""
from __future__ import annotations

import argparse
import json
import os

from .data import build_id_loaders
from .models import build_model
from .utils import load_config
from .utils.train_loop import train_supervised


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--teacher", required=True, help="teacher `name` from the config")
    args = p.parse_args()

    cfg = load_config(args.config, override_name=args.teacher)
    spec = next(t for t in cfg["teachers"] if t["name"] == args.teacher)

    train_loader, test_loader = build_id_loaders(cfg)
    model = build_model(spec["arch"], num_classes=cfg["num_classes"], cifar_stem=spec.get("cifar_stem", True))

    ckpt_path = os.path.join(cfg["ckpt_root"], cfg["id_dataset"], "teachers", f"{spec['name']}.pt")
    log = train_supervised(model, train_loader, test_loader, cfg, ckpt_path)

    log_dir = os.path.join(cfg["log_root"], cfg["id_dataset"], "teachers")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"{spec['name']}.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"[done] {spec['name']} best_acc={log['best_acc']*100:.2f}% saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
