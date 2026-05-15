"""Precompute SLG sample-level Shapley gates over the ID training set (no augmentation).

For each training sample x_i with label y_i and each coalition S ⊆ teachers:
    v(S, x_i) = log p_S^{y_i}(x_i)   with v(∅, x_i) = -log C.
Shapley φ_k(x_i) is the standard weighted sum of marginal contributions.
The sample-level gate is g_SLG_k(x_i) = softmax(φ_k(x_i) / τ_gate) over k.

Outputs:
    outputs/results/<id>/slg_gate.pt
        {"gate": (N, M) float32,
         "phi":  (N, M) float32,
         "teachers": list[str],
         "tau_prob": float, "tau_gate": float}

Idempotent: skips if the output file exists (use --overwrite to force).

Usage:
    python -m src.precompute_slg --config configs/teachers.yaml
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
from typing import List

import torch
import torch.nn.functional as F

from .data import build_id_train_loader_indexed
from .models import build_model
from .utils import load_config


@torch.no_grad()
def _dump_teacher_logits_train_noaug(cfg, teacher_specs, device: str):
    """Forward every teacher once over the training set (no augmentation, eval mode).

    Returns (logits: (M, N, C) float32 on CPU, labels: (N,) long on CPU).
    """
    loader = build_id_train_loader_indexed(cfg, train_aug=False)
    n_total = len(loader.dataset)
    num_classes = cfg["num_classes"]
    M = len(teacher_specs)
    logits = torch.zeros(M, n_total, num_classes, dtype=torch.float32)
    labels = torch.zeros(n_total, dtype=torch.long)
    filled = torch.zeros(n_total, dtype=torch.bool)

    for ti, spec in enumerate(teacher_specs):
        name, arch = spec["name"], spec["arch"]
        ckpt_path = os.path.join(cfg["ckpt_root"], cfg["id_dataset"], "teachers", f"{name}.pt")
        print(f"  forwarding teacher {name} ({arch}) over training set (no-aug)")
        model = build_model(arch, num_classes=num_classes, cifar_stem=spec.get("cifar_stem", True))
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"])
        model.to(device).eval()
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)
            out = model(x).float().cpu()
            idx_long = idx.long()
            logits[ti, idx_long] = out
            if ti == 0:
                labels[idx_long] = y.long()
                filled[idx_long] = True
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not filled.all():
        miss = int((~filled).sum())
        raise RuntimeError(f"Indexed training loader missed {miss} samples — check dataset wrapper.")
    return logits, labels


def _shapley_weights(M: int) -> List[float]:
    return [math.factorial(s) * math.factorial(M - s - 1) / math.factorial(M) for s in range(M)]


def compute_shapley_log_prob(
    logits_M_N_C: torch.Tensor,
    labels: torch.Tensor,
    tau_prob: float = 1.0,
) -> torch.Tensor:
    """Per-sample Shapley over teachers with v(S, x) = log p_S^y(x; tau_prob).

    Returns φ : (M, N) on the same device as logits.
    """
    M, N, C = logits_M_N_C.shape
    device = logits_M_N_C.device
    coalitions = list(itertools.product([0, 1], repeat=M))
    weights_by_size = _shapley_weights(M)
    log_C = math.log(C)

    v = torch.empty(len(coalitions), N, device=device)
    arange_N = torch.arange(N, device=device)
    for k, c in enumerate(coalitions):
        mask = torch.tensor(c, device=device, dtype=torch.bool)
        if not mask.any():
            v[k] = -log_C
        else:
            ens_logits = logits_M_N_C[mask].mean(dim=0)  # (N, C)
            log_p = F.log_softmax(ens_logits / float(tau_prob), dim=-1)
            v[k] = log_p[arange_N, labels]  # (N,)

    coalition_index = {c: k for k, c in enumerate(coalitions)}
    phi = torch.zeros(M, N, device=device)
    for i in range(M):
        for k, c in enumerate(coalitions):
            if c[i] == 1:
                continue
            s_size = sum(c)
            c_with = list(c); c_with[i] = 1
            k_with = coalition_index[tuple(c_with)]
            phi[i] = phi[i] + weights_by_size[s_size] * (v[k_with] - v[k])
    return phi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/teachers.yaml")
    p.add_argument("--tau-prob", type=float, default=1.0, help="Temperature inside the characteristic function softmax")
    p.add_argument("--tau-gate", type=float, default=1.0, help="Temperature for φ → gate softmax")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu"

    out_dir = os.path.join(cfg["result_root"], cfg["id_dataset"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "slg_gate.pt")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] SLG gate cache already exists: {out_path}")
        return

    teacher_specs = cfg["teachers"]
    teacher_names = [t["name"] for t in teacher_specs]
    M = len(teacher_specs)
    print(f"=== Precomputing SLG gate on ID training set ({cfg['id_dataset']}) with M={M} teachers ===")

    logits_cpu, labels_cpu = _dump_teacher_logits_train_noaug(cfg, teacher_specs, device)
    print(f"  computing exact 2^{M} Shapley for {labels_cpu.size(0)} samples")
    logits = logits_cpu.to(device)
    labels = labels_cpu.to(device)

    phi = compute_shapley_log_prob(logits, labels, tau_prob=args.tau_prob)  # (M, N)
    gate = F.softmax(phi / float(args.tau_gate), dim=0)                     # (M, N)

    # Sanity: row sums to 1 within numerical noise
    col_sum = gate.sum(dim=0)
    print(f"  gate col_sum  mean={col_sum.mean().item():.6f}  min={col_sum.min().item():.6f}  max={col_sum.max().item():.6f}")

    # Save shape (N, M): row-per-sample is more natural for downstream indexing.
    out = {
        "gate": gate.t().contiguous().cpu(),
        "phi": phi.t().contiguous().cpu(),
        "teachers": teacher_names,
        "tau_prob": float(args.tau_prob),
        "tau_gate": float(args.tau_gate),
        "id_dataset": cfg["id_dataset"],
        "n_samples": int(labels_cpu.size(0)),
    }
    torch.save(out, out_path)
    print(f"[done] wrote {out_path}   (gate shape {tuple(out['gate'].shape)})")


if __name__ == "__main__":
    main()
