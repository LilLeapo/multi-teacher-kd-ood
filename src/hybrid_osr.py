"""Experiment 3 — student evaluation: ID acc + standalone OSR + hybrid OSR.

For each (student_arch, method, seed) we report:

    1. ID accuracy on CIFAR-100 test
    2. Standalone OSR: student's own Energy as ID/OOD score → AUROC / FPR95
    3. Hybrid OSR: student handles ID classification, but the teacher *ensemble*
       Energy decides ID-vs-OOD. Hybrid OSR's AUROC/FPR95 depend only on the
       teacher ensemble — they're identical across students — and serve as
       the upper-bound reference the standalone column is benchmarked against.

This pipeline reuses logit caches written by `src.jensen_gap dump`:

    - teacher cache at `outputs/logit_cache/<id>/teachers/<teacher>__<split>.pt`
    - student cache at `outputs/logit_cache/<id>/students/<tag>__<split>.pt`

If a student cache is missing it is built on the fly (single forward over each split).

Usage:
    python -m src.hybrid_osr \\
        --teacher-config configs/teachers.yaml \\
        --student-config configs/students.yaml \\
        --teachers resnet50 densenet121 wide_resnet50_2 resnext50_32x4d convnext_tiny \\
        --student shufflenetv2_x0_5 \\
        --methods uniform accuracy learned_global \\
        --seeds 42 123 3407 \\
        --tag exp3_simplified
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

from .jensen_gap import (
    _all_splits, _cache_path, _dump_one, _energy, _loader_for_split,
)
from .models import build_model
from .utils import load_config
from .utils.ood_metrics import ood_metrics


def _ensure_student_cache(
    cfg: dict, arch: str, tag: str, cifar_stem: bool = True, force: bool = False
) -> Dict[str, torch.Tensor]:
    """Make sure the student logit cache exists for all splits. Returns {split: logits}."""
    splits = _all_splits(cfg)
    out: Dict[str, torch.Tensor] = {}
    missing = []
    for s in splits:
        p = _cache_path(cfg, "student", tag, s)
        if os.path.exists(p) and not force:
            out[s] = torch.load(p, map_location="cpu")["logits"].float()
        else:
            missing.append(s)

    if not missing:
        return out

    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(cfg["ckpt_root"], cfg["id_dataset"], "students", f"{tag}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"student checkpoint missing: {ckpt_path}")
    print(f"  [student-cache] building cache for {tag} (missing splits: {missing})")
    model = build_model(arch, num_classes=cfg["num_classes"], cifar_stem=cifar_stem)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device).eval()

    os.makedirs(os.path.dirname(_cache_path(cfg, "student", tag, missing[0])), exist_ok=True)
    for s in missing:
        loader, has_labels = _loader_for_split(s, cfg)
        logits, labels, acc = _dump_one(model, loader, device, has_labels)
        torch.save({"logits": logits, "labels": labels, "acc": acc,
                    "model": tag, "split": s, "arch": arch},
                   _cache_path(cfg, "student", tag, s))
        out[s] = logits.float()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _student_id_acc(cfg: dict, tag: str) -> float:
    """Re-derive ID test accuracy from cached student logits + labels."""
    cache = torch.load(_cache_path(cfg, "student", tag, "id_test"), map_location="cpu")
    labels = cache.get("labels")
    logits = cache["logits"].float()
    if labels is None:
        return float("nan")
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def _load_teacher_logits_all_splits(
    cfg: dict, teacher_names: List[str]
) -> Dict[str, torch.Tensor]:
    """Average the teacher logits across the ensemble, for every split."""
    splits = _all_splits(cfg)
    out: Dict[str, torch.Tensor] = {}
    for s in splits:
        stack = []
        for name in teacher_names:
            p = _cache_path(cfg, "teacher", name, s)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"teacher logit cache missing: {p} (run jensen_gap dump first)"
                )
            stack.append(torch.load(p, map_location="cpu")["logits"].float())
        out[s] = torch.stack(stack).mean(dim=0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-config", required=True)
    p.add_argument("--student-config", required=True)
    p.add_argument("--teachers", nargs="+", required=True,
                   help="teacher names whose logit caches form the ensemble")
    p.add_argument("--student", required=True,
                   help="student arch name from the student config (e.g. shufflenetv2_x0_5)")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", required=True, type=int)
    p.add_argument("--tag", required=True)
    p.add_argument("--energy-T", type=float, default=1.0)
    p.add_argument("--force", action="store_true", help="rebuild student caches")
    args = p.parse_args()

    s_cfg = load_config(args.student_config)
    # `_dump_one` and `_loader_for_split` use the same cfg for I/O paths; teacher
    # and student configs differ only in `students:` vs `teachers:` blocks, so
    # reuse the student cfg for everything OOD-related.
    cfg = s_cfg

    s_spec = next(s for s in s_cfg["students"] if s["name"] == args.student)
    s_arch = s_spec["arch"]
    cifar_stem = s_spec.get("cifar_stem", True)
    id_name = cfg["id_dataset"]
    ood_names = [o["name"] for o in cfg["ood_eval"]]
    T = float(args.energy_T)

    # ---- ensemble OSR baseline (constant across students) ----
    print(f"=== hybrid OSR (tag={args.tag}, ID={id_name}, OOD={ood_names}) ===")
    print(f"  teachers ({len(args.teachers)}): {args.teachers}")
    print(f"  student arch: {s_arch}")
    ens_logits = _load_teacher_logits_all_splits(cfg, args.teachers)
    ens_energy = {s: _energy(t, T).numpy() for s, t in ens_logits.items()}
    hybrid = {ood: ood_metrics(ens_energy["id_test"], ens_energy[ood]) for ood in ood_names}
    # Ensemble ID acc (with labels from any teacher cache)
    id_labels = torch.load(_cache_path(cfg, "teacher", args.teachers[0], "id_test"),
                           map_location="cpu").get("labels")
    ens_id_acc = float((ens_logits["id_test"].argmax(dim=-1) == id_labels).float().mean().item()) \
        if id_labels is not None else None

    # ---- per (method, seed) student evaluation ----
    per_run: Dict[str, Dict[str, object]] = {}
    for method in args.methods:
        for seed in args.seeds:
            tag = f"{s_spec['name']}__{method}__seed{seed}"
            ckpt = os.path.join(cfg["ckpt_root"], cfg["id_dataset"], "students", f"{tag}.pt")
            if not os.path.exists(ckpt):
                print(f"[warn] skipping {tag} (no checkpoint)")
                continue
            print(f"  [eval] {tag}")
            stu_logits = _ensure_student_cache(cfg, s_arch, tag, cifar_stem, force=args.force)
            id_acc = _student_id_acc(cfg, tag)
            stu_energy = {s: _energy(stu_logits[s], T).numpy() for s in stu_logits}
            stand = {ood: ood_metrics(stu_energy["id_test"], stu_energy[ood])
                     for ood in ood_names}
            per_run[tag] = {
                "method": method, "seed": seed, "id_acc": id_acc,
                "standalone": stand,
            }

    # ---- aggregate by method (mean ± std across seeds) ----
    by_method: Dict[str, Dict[str, object]] = {}
    for method in args.methods:
        seeds_present = [r for tag, r in per_run.items() if r["method"] == method]
        if not seeds_present:
            continue
        accs = np.array([r["id_acc"] for r in seeds_present])
        block: Dict[str, object] = {
            "n_seeds": len(seeds_present),
            "id_acc_mean": float(accs.mean()),
            "id_acc_std": float(accs.std()),
            "per_ood": {},
        }
        for ood in ood_names:
            au = np.array([r["standalone"][ood]["auroc"] for r in seeds_present])
            fp = np.array([r["standalone"][ood]["fpr95"] for r in seeds_present])
            block["per_ood"][ood] = {
                "auroc_mean": float(au.mean()),
                "auroc_std": float(au.std()),
                "fpr95_mean": float(fp.mean()),
                "fpr95_std": float(fp.std()),
            }
        by_method[method] = block

    # ---- console output ----
    print()
    print("=== T3 — student-side results ===")
    print(f"{'method':>16s} | {'n':>3s} | {'ID acc':>14s} | "
          + " | ".join(f"{ood} AUROC".rjust(16) for ood in ood_names))
    print("-" * (45 + 19 * len(ood_names)))
    for method, blk in by_method.items():
        cells = " | ".join(
            f"{blk['per_ood'][ood]['auroc_mean']:.4f} ± {blk['per_ood'][ood]['auroc_std']:.4f}".rjust(16)
            for ood in ood_names
        )
        print(f"{method:>16s} | {blk['n_seeds']:>3d} | "
              f"{blk['id_acc_mean']*100:>6.2f}% ± {blk['id_acc_std']*100:.2f}% | {cells}")
    print()
    print("=== Hybrid baseline (teacher ensemble Energy decides OOD) ===")
    ens_acc_str = f"{ens_id_acc*100:.2f}%" if ens_id_acc is not None else "—"
    print(f"  ensemble ID acc = {ens_acc_str}")
    for ood in ood_names:
        h = hybrid[ood]
        print(f"  {ood:>10s}  AUROC={h['auroc']:.4f}  FPR95={h['fpr95']:.4f}")
    print()
    print("Headline: standalone (best student) vs hybrid AUROC per OOD set:")
    for ood in ood_names:
        best = max((blk['per_ood'][ood]['auroc_mean'] for blk in by_method.values()), default=0.0)
        print(f"  {ood:>10s}: standalone_best={best:.4f}  hybrid={hybrid[ood]['auroc']:.4f}  "
              f"gap={hybrid[ood]['auroc']-best:+.4f}")

    # ---- write JSON + Markdown ----
    out_dir = os.path.join(cfg["result_root"], cfg["id_dataset"])
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"hybrid_osr__{args.tag}")
    payload = {
        "tag": args.tag,
        "id_dataset": id_name,
        "ood": ood_names,
        "teachers": args.teachers,
        "student_arch": s_arch,
        "energy_T": T,
        "per_run": per_run,
        "by_method": by_method,
        "hybrid": hybrid,
        "ensemble_id_acc": ens_id_acc,
    }
    with open(prefix + ".json", "w") as f:
        json.dump(payload, f, indent=2)

    md = [f"# Hybrid OSR report — `{args.tag}` (ID={id_name})", ""]
    md.append(f"- Teachers ({len(args.teachers)}): {', '.join(f'`{t}`' for t in args.teachers)}")
    md.append(f"- Student arch: `{s_arch}`")
    md.append(f"- OOD: {', '.join(f'`{o}`' for o in ood_names)}")
    md.append(f"- Energy T = {T}")
    md.append("")
    md.append("## T3 — standalone student OSR (mean ± std across seeds)")
    md.append("")
    hdrs = ["method", "n", "ID acc"] + [f"{ood} AUROC" for ood in ood_names] \
           + [f"{ood} FPR95" for ood in ood_names]
    md.append("| " + " | ".join(hdrs) + " |")
    md.append("|" + "|".join("---" for _ in hdrs) + "|")
    for method, blk in by_method.items():
        row = [method, str(blk["n_seeds"]),
               f"{blk['id_acc_mean']*100:.2f}% ± {blk['id_acc_std']*100:.2f}%"]
        for ood in ood_names:
            row.append(f"{blk['per_ood'][ood]['auroc_mean']:.4f} ± "
                       f"{blk['per_ood'][ood]['auroc_std']:.4f}")
        for ood in ood_names:
            row.append(f"{blk['per_ood'][ood]['fpr95_mean']:.4f} ± "
                       f"{blk['per_ood'][ood]['fpr95_std']:.4f}")
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("## Hybrid baseline (teacher ensemble Energy)")
    md.append("")
    md.append(f"- Ensemble ID acc: {ens_acc_str}")
    md.append("")
    md.append("| OOD | AUROC | FPR95 |")
    md.append("|---|---|---|")
    for ood in ood_names:
        h = hybrid[ood]
        md.append(f"| `{ood}` | {h['auroc']:.4f} | {h['fpr95']:.4f} |")
    with open(prefix + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n[done] wrote:\n  {prefix}.json\n  {prefix}.md")


if __name__ == "__main__":
    main()
