"""Step 2 — 4-teacher logit-averaging OOD AUROC table.

For logit-only scores (MSP / MaxLogit / Energy / GEN) we average teacher logits
*before* scoring. For feature-based scores (KNN / Mahalanobis / ViM) we score
each teacher in its own feature space (dimensions differ across architectures)
and average the resulting scores — this is the natural ensemble extension.

Output: outputs/results/<id>/ensemble_ood.{json,md}. The Markdown table is
formatted to be copy-paste-ready into a paper draft.

Usage:
    python -m src.ensemble_ood --config configs/teachers.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import build_id_loaders, build_ood_loader
from .data.datasets import _eval_tf, _id_dataset
from .models import build_model
from .utils import load_config
from .utils.advanced_scores import (
    KNNScorer,
    MahalanobisScorer,
    ViMScorer,
    score_energy,
    score_gen,
    score_maxlogit,
    score_msp,
)
from .utils.feature_extract import extract, head_weight_bias
from .utils.ood_metrics import ood_metrics


LOGIT_SCORES = ["msp", "maxlogit", "energy", "gen"]
FEATURE_SCORES = ["knn", "mahalanobis", "vim"]
ALL_SCORES = LOGIT_SCORES + FEATURE_SCORES


@torch.no_grad()
def _forward_collect(model, arch: str, loader: DataLoader, device: str):
    """Return (features [N, D] fp16, logits [N, C] fp16, labels [N] or None)."""
    feats, logits, labels = [], [], []
    has_labels = None
    with extract(model, arch) as fx:
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            f, lg = fx(x)
            feats.append(f.detach().to(torch.float16).cpu())
            logits.append(lg.detach().to(torch.float16).cpu())
            if has_labels is None:
                has_labels = len(batch) > 1
            if has_labels:
                labels.append(batch[1])
    feats = torch.cat(feats, dim=0)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0) if has_labels else None
    return feats, logits, labels


def _id_train_loader_eval_tf(cfg) -> DataLoader:
    ds = _id_dataset(cfg["id_dataset"], cfg["data_root"], train=True, tf=_eval_tf())
    return DataLoader(
        ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )


def _id_likeness(score_name: str, logits: torch.Tensor) -> torch.Tensor:
    if score_name == "msp":
        return score_msp(logits)
    if score_name == "maxlogit":
        return score_maxlogit(logits)
    if score_name == "energy":
        return score_energy(logits, T=1.0)
    if score_name == "gen":
        return score_gen(logits, gamma=0.1)
    raise ValueError(score_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/teachers.yaml")
    p.add_argument("--knn-k", type=int, default=50)
    p.add_argument("--vim-dim", type=int, default=512)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = cfg["device"]
    num_classes = cfg["num_classes"]
    id_name = cfg["id_dataset"]
    ood_names = [o["name"] for o in cfg["ood_eval"]]

    teacher_specs = cfg["teachers"]
    teacher_names = [t["name"] for t in teacher_specs]
    print(f"=== ensemble OOD ({id_name} ID, OOD={ood_names}) over {teacher_names} ===")

    # Build dataset loaders once
    id_train_loader = _id_train_loader_eval_tf(cfg)
    _, id_test_loader = build_id_loaders(cfg)
    ood_loaders = {name: build_ood_loader(name, cfg) for name in ood_names}

    # Per-teacher: forward all sets, fit feature scorers, score everything.
    # We aggregate logit-based by averaging *logits*, feature-based by averaging *scores*.
    sum_logits_id_test = None
    sum_logits_ood = {n: None for n in ood_names}
    sum_feat_scores_id_test = {s: None for s in FEATURE_SCORES}
    sum_feat_scores_ood = {s: {n: None for n in ood_names} for s in FEATURE_SCORES}

    for spec in teacher_specs:
        name, arch = spec["name"], spec["arch"]
        ckpt_path = os.path.join(cfg["ckpt_root"], id_name, "teachers", f"{name}.pt")
        print(f"  [{name}] loading {ckpt_path}")
        model = build_model(arch, num_classes=num_classes, cifar_stem=spec.get("cifar_stem", True))
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"])
        model.to(device).eval()

        print(f"  [{name}] forward ID train")
        f_tr, l_tr, y_tr = _forward_collect(model, arch, id_train_loader, device)
        print(f"  [{name}] forward ID test + OOD sets")
        f_te, l_te, _ = _forward_collect(model, arch, id_test_loader, device)
        f_ood, l_ood = {}, {}
        for ood_name, ld in ood_loaders.items():
            f_ood[ood_name], l_ood[ood_name], _ = _forward_collect(model, arch, ld, device)

        # Aggregate logits (logit-based scores ensemble via mean logits)
        if sum_logits_id_test is None:
            sum_logits_id_test = l_te.float()
        else:
            sum_logits_id_test = sum_logits_id_test + l_te.float()
        for n in ood_names:
            if sum_logits_ood[n] is None:
                sum_logits_ood[n] = l_ood[n].float()
            else:
                sum_logits_ood[n] = sum_logits_ood[n] + l_ood[n].float()

        # Fit feature scorers on ID train; use fp32 on GPU for numerical stability.
        print(f"  [{name}] fit KNN / Mahalanobis / ViM")
        f_tr_g = f_tr.float().to(device)
        l_tr_g = l_tr.float().to(device)
        y_tr_g = y_tr.to(device)
        W, b = head_weight_bias(model, arch)
        W, b = W.to(device).float(), b.to(device).float()

        knn = KNNScorer(k=args.knn_k); knn.fit(f_tr_g)
        maha = MahalanobisScorer(); maha.fit(f_tr_g, y_tr_g, num_classes)
        vim = ViMScorer(); vim.fit(f_tr_g, l_tr_g, W, b, principal_dim=args.vim_dim)
        del f_tr_g, l_tr_g, y_tr_g

        # Score ID test and each OOD set
        def _score_per_teacher(features, logits):
            f_g = features.float().to(device)
            l_g = logits.float().to(device)
            return {
                "knn": knn.score(f_g).cpu(),
                "mahalanobis": maha.score(f_g).cpu(),
                "vim": vim.score(f_g, l_g).cpu(),
            }

        s_te = _score_per_teacher(f_te, l_te)
        for sname in FEATURE_SCORES:
            sum_feat_scores_id_test[sname] = (
                s_te[sname] if sum_feat_scores_id_test[sname] is None
                else sum_feat_scores_id_test[sname] + s_te[sname]
            )
        for ood_name in ood_names:
            s_o = _score_per_teacher(f_ood[ood_name], l_ood[ood_name])
            for sname in FEATURE_SCORES:
                sum_feat_scores_ood[sname][ood_name] = (
                    s_o[sname] if sum_feat_scores_ood[sname][ood_name] is None
                    else sum_feat_scores_ood[sname][ood_name] + s_o[sname]
                )

        # Free GPU
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    M = len(teacher_specs)
    # Convert sums to means
    mean_logits_id = sum_logits_id_test / M
    mean_logits_ood = {n: sum_logits_ood[n] / M for n in ood_names}
    mean_feat_id = {s: sum_feat_scores_id_test[s] / M for s in FEATURE_SCORES}
    mean_feat_ood = {s: {n: sum_feat_scores_ood[s][n] / M for n in ood_names} for s in FEATURE_SCORES}

    # Compute AUROC per (score, ood)
    results: Dict[str, Dict[str, float]] = {s: {} for s in ALL_SCORES}
    for s in LOGIT_SCORES:
        id_scores = _id_likeness(s, mean_logits_id).numpy()
        for n in ood_names:
            ood_scores = _id_likeness(s, mean_logits_ood[n]).numpy()
            results[s][n] = ood_metrics(id_scores, ood_scores)["auroc"]
    for s in FEATURE_SCORES:
        id_scores = mean_feat_id[s].numpy()
        for n in ood_names:
            ood_scores = mean_feat_ood[s][n].numpy()
            results[s][n] = ood_metrics(id_scores, ood_scores)["auroc"]

    # Print + save table
    pretty = {"energy": "Energy", "msp": "MSP", "maxlogit": "MaxLogit", "gen": "GEN",
              "knn": "KNN", "mahalanobis": "Mahalanobis", "vim": "ViM"}
    pretty_ood = {"cifar10": "CIFAR-10 (near)", "cifar100": "CIFAR-100 (near)",
                  "svhn": "SVHN", "textures": "Textures"}
    ood_header = [pretty_ood.get(n, n) for n in ood_names]

    rows = []
    order = ["energy", "msp", "maxlogit", "gen", "knn", "mahalanobis", "vim"]
    for s in order:
        vals = [results[s][n] for n in ood_names]
        rows.append((pretty[s], vals, float(np.mean(vals))))

    print()
    print(f"{M}-teacher logit averaging on {id_name.upper()} ID:")
    print()
    header = f"{'Score':<12} | " + " | ".join(f"{h:^15}" for h in ood_header) + " |  mean"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for label, vals, mean in rows:
        cells = " | ".join(f"{v:^15.4f}" for v in vals)
        print(f"{label:<12} | {cells} | {mean:.4f}")

    # Markdown
    md_lines = [f"## {M}-teacher logit averaging on {id_name.upper()} ID", ""]
    md_lines.append("| Score | " + " | ".join(ood_header) + " | mean |")
    md_lines.append("|" + "---|" * (len(ood_header) + 2))
    for label, vals, mean in rows:
        md_lines.append("| " + " | ".join([label] + [f"{v:.4f}" for v in vals] + [f"{mean:.4f}"]) + " |")

    out_dir = os.path.join(cfg["result_root"], id_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ensemble_ood.json"), "w") as f:
        json.dump({"teachers": teacher_names, "id": id_name, "ood": ood_names, "auroc": results}, f, indent=2)
    with open(os.path.join(out_dir, "ensemble_ood.md"), "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\n[done] wrote {out_dir}/ensemble_ood.{{json,md}}")


if __name__ == "__main__":
    main()
