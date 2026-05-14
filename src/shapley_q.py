"""Step 3 — exact 2^M Shapley enumeration on per-sample teacher contributions.

For each ID test sample x and each coalition S ⊆ {teachers}, the characteristic
function is v(S, x) = -CE(mean logits over S, y_x), with v(∅, x) = -log(C)
(uniform prediction baseline). Shapley value φ_i(x) is the standard weighted
sum of marginal contributions. We then convert per-sample (φ_1, …, φ_M) to a
distribution g_S2(x) = softmax(φ) and report

    q_shap(x) = 1 - H(g_S2(x)) / log(M)

q_shap ∈ [0, 1]. High = a small subset of teachers dominates the prediction
for that sample (sample-level differentiation). Low = all teachers contribute
equally (Shapley signal is degenerate).

Verdict cutoffs follow the previous-setup empirical thresholds:
    mean < 0.05 AND p95 < 0.15  →  Fate A (collapse, N1 reproduces)
    mean > 0.15 AND p90 > 0.40  →  Fate B (clear differentiation)
    otherwise                   →  middle state, run S3 v2 + global gate

Usage:
    python -m src.shapley_q --config configs/teachers.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from typing import List

import torch
import torch.nn.functional as F

from .data import build_id_loaders
from .models import build_model
from .utils import load_config


@torch.no_grad()
def _logits_on_id_test(cfg, teacher_specs, device: str):
    """Returns (M, N, C) stacked logits and (N,) labels."""
    _, id_test_loader = build_id_loaders(cfg)
    per_teacher_logits = []
    labels_collected = None
    for spec in teacher_specs:
        name, arch = spec["name"], spec["arch"]
        ckpt = os.path.join(cfg["ckpt_root"], cfg["id_dataset"], "teachers", f"{name}.pt")
        print(f"  forwarding {name}")
        model = build_model(arch, num_classes=cfg["num_classes"], cifar_stem=spec.get("cifar_stem", True))
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        model.to(device).eval()
        chunks_l, chunks_y = [], []
        for x, y in id_test_loader:
            x = x.to(device, non_blocking=True)
            chunks_l.append(model(x).float().cpu())
            chunks_y.append(y)
        per_teacher_logits.append(torch.cat(chunks_l, dim=0))
        if labels_collected is None:
            labels_collected = torch.cat(chunks_y, dim=0)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    logits = torch.stack(per_teacher_logits, dim=0)  # (M, N, C)
    return logits, labels_collected


def _shapley_weights(M: int) -> List[float]:
    """w(|S|) for |S| = 0..M-1 in the φ formula."""
    return [math.factorial(s) * math.factorial(M - s - 1) / math.factorial(M) for s in range(M)]


def compute_shapley(logits_M_N_C: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return Shapley value tensor (M, N) per sample."""
    M, N, C = logits_M_N_C.shape
    device = logits_M_N_C.device
    coalitions = list(itertools.product([0, 1], repeat=M))  # 2^M tuples
    weights_by_size = _shapley_weights(M)

    # v(S, x) for every coalition: (|coalitions|, N)
    v = torch.empty(len(coalitions), N, device=device)
    log_C = math.log(C)
    for k, c in enumerate(coalitions):
        mask = torch.tensor(c, device=device, dtype=torch.bool)
        if not mask.any():
            v[k] = -log_C
        else:
            ens = logits_M_N_C[mask].mean(dim=0)  # (N, C)
            ce = F.cross_entropy(ens, labels, reduction="none")
            v[k] = -ce

    coalition_index = {c: k for k, c in enumerate(coalitions)}
    shapley = torch.zeros(M, N, device=device)
    for i in range(M):
        for k, c in enumerate(coalitions):
            if c[i] == 1:
                continue
            s_size = sum(c)
            c_with = list(c); c_with[i] = 1
            k_with = coalition_index[tuple(c_with)]
            shapley[i] = shapley[i] + weights_by_size[s_size] * (v[k_with] - v[k])
    return shapley


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/teachers.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = cfg["device"]
    teacher_specs = cfg["teachers"]
    M = len(teacher_specs)
    teacher_names = [t["name"] for t in teacher_specs]
    print(f"=== Shapley q_shap on ID test ({cfg['id_dataset']}) with M={M} teachers ===")

    logits, labels = _logits_on_id_test(cfg, teacher_specs, device)
    logits = logits.to(device)
    labels = labels.to(device)

    print(f"  enumerating {2**M} coalitions for {labels.size(0)} samples ...")
    shapley = compute_shapley(logits, labels)  # (M, N)

    g = F.softmax(shapley, dim=0)  # (M, N) probability over teachers per sample
    eps = 1e-12
    H = -(g * (g + eps).log()).sum(dim=0)  # (N,)
    q_shap = 1.0 - H / math.log(M)         # (N,)

    qs = q_shap.cpu().numpy()
    stats = {
        "mean": float(qs.mean()),
        "median": float(torch.tensor(qs).median().item()),
        "p25": float(torch.tensor(qs).quantile(0.25).item()),
        "p75": float(torch.tensor(qs).quantile(0.75).item()),
        "p90": float(torch.tensor(qs).quantile(0.90).item()),
        "p95": float(torch.tensor(qs).quantile(0.95).item()),
        "frac_above_0.15": float((qs > 0.15).mean()),
        "frac_above_0.30": float((qs > 0.30).mean()),
    }

    if stats["mean"] < 0.05 and stats["p95"] < 0.15:
        verdict = "FATE_A: q_shap collapse — Shapley degenerate, N1 reproduces."
    elif stats["mean"] > 0.15 and stats["p90"] > 0.40:
        verdict = "FATE_B: clear sample-level differentiation."
    else:
        verdict = "MIDDLE: run S3 v2 + global gate to decide."

    per_teacher_mean = {teacher_names[i]: float(shapley[i].mean().item()) for i in range(M)}

    out = {
        "teachers": teacher_names,
        "id_dataset": cfg["id_dataset"],
        "M": M,
        "n_samples": int(labels.size(0)),
        "q_shap_stats": stats,
        "per_teacher_mean_shapley": per_teacher_mean,
        "verdict": verdict,
    }

    print()
    print(f"q_shap stats:  mean={stats['mean']:.4f}  median={stats['median']:.4f}"
          f"  p90={stats['p90']:.4f}  p95={stats['p95']:.4f}")
    print("Per-teacher mean Shapley (positive = on average helps):")
    for k, v in per_teacher_mean.items():
        print(f"  {k:<24} {v:+.4f}")
    print()
    print(f"VERDICT: {verdict}")

    out_dir = os.path.join(cfg["result_root"], cfg["id_dataset"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "shapley_q.json"), "w") as f:
        json.dump(out, f, indent=2)
    # Per-sample dump for downstream analysis
    torch.save(
        {"q_shap": q_shap.cpu(), "shapley": shapley.cpu(), "teachers": teacher_names, "labels": labels.cpu()},
        os.path.join(out_dir, "shapley_q.pt"),
    )
    print(f"[done] wrote {out_dir}/shapley_q.{{json,pt}}")


if __name__ == "__main__":
    main()
