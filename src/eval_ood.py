"""Evaluate a trained model on ID + OOD datasets.

Reports MSP / MaxLogit / Energy scores + AUROC / AUPR / FPR95 against each OOD set.

Usage:
    python -m src.eval_ood --config configs/students.yaml --ckpt checkpoints/cifar100/students/shufflenetv2_x0_5_kd.pt --arch shufflenetv2_x0_5 --tag student_shufflenet
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from .data import build_id_loaders, build_ood_loader
from .models import build_model
from .utils import load_config
from .utils.ood_metrics import ood_metrics, score_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--no-cifar-stem", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = cfg["device"]

    model = build_model(args.arch, num_classes=cfg["num_classes"], cifar_stem=not args.no_cifar_stem)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device).eval()

    scores_to_eval = cfg["ood"]["scores"]
    energy_T = cfg["ood"].get("energy_temperature", 1.0)

    _, id_test_loader = build_id_loaders(cfg)
    id_scores = score_dataset(model, id_test_loader, device, scores_to_eval, energy_T)

    results = {"tag": args.tag, "arch": args.arch, "id_dataset": cfg["id_dataset"], "ckpt": args.ckpt, "per_score": {}}
    for s in scores_to_eval:
        results["per_score"][s] = {"ood": {}}
    for ood_spec in cfg["ood_eval"]:
        ood_name = ood_spec["name"]
        ood_loader = build_ood_loader(ood_name, cfg)
        ood_scores = score_dataset(model, ood_loader, device, scores_to_eval, energy_T)
        for s in scores_to_eval:
            results["per_score"][s]["ood"][ood_name] = ood_metrics(id_scores[s], ood_scores[s])
            print(f"  [{s}] {ood_name}: {results['per_score'][s]['ood'][ood_name]}")

    out_dir = os.path.join(cfg["result_root"], cfg["id_dataset"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ood_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
