"""Dump teacher logits over the ID train set so distillation doesn't repeat the forward pass.

Stored as `outputs/logits/<id>/<teacher>.pt` containing:
  { "logits": float16 [N, num_classes], "indices": int64 [N], "acc": float }

Indices match the order the train set is iterated below (eval transform, shuffle=False),
so distill.py builds its own loader with the same ordering.
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .data.datasets import _id_dataset, _eval_tf
from .models import build_model
from .utils import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--teacher", required=True)
    args = p.parse_args()

    cfg = load_config(args.config, override_name=args.teacher)
    spec = next(t for t in cfg["teachers"] if t["name"] == args.teacher)
    device = cfg["device"]

    ds = _id_dataset(cfg["id_dataset"], cfg["data_root"], train=True, tf=_eval_tf())
    loader = DataLoader(
        ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )

    model = build_model(spec["arch"], num_classes=cfg["num_classes"], cifar_stem=spec.get("cifar_stem", True))
    ckpt_path = os.path.join(cfg["ckpt_root"], cfg["id_dataset"], "teachers", f"{spec['name']}.pt")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device).eval()

    logits_all, correct, total = [], 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
            logits_all.append(logits.cpu().to(torch.float16))
    logits_all = torch.cat(logits_all, dim=0)

    out_dir = os.path.join(cfg["logit_root"], cfg["id_dataset"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{spec['name']}.pt")
    torch.save({"logits": logits_all, "acc": correct / max(1, total)}, out_path)
    print(f"[done] {spec['name']} train_acc={correct/total*100:.2f}% -> {out_path}")


if __name__ == "__main__":
    main()
