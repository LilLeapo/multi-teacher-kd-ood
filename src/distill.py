"""Multi-teacher KD with method dispatch — supports six weighting schemes.

Methods (selected via `--method`, or `kd.teacher_weighting` in the student config):

    uniform         — fixed 1/M weights
    accuracy        — fixed weights ∝ teacher val acc
    learned_global  — single learnable M-vector (softmax + entropy bonus)
    slg             — per-sample Shapley gate, read from outputs/results/<id>/slg_gate.pt
    csr_kd          — RouterMLP on student features, joint-trained, mixes with SLG gate
    srr             — SelectiveResidualRouter (gate + r(x)) with optional anchor loss

Outputs:
    checkpoints/<id>/students/<student>__<method>__seed{S}.pt   # best ID-acc ckpt
    logs/<id>/students/<student>__<method>__seed{S}.json        # history

Skip-if-exists:
    By default skips a run when the checkpoint file already exists. Pass --force
    to retrain. Useful for the multi-seed × multi-method orchestrator.

Usage:
    python -m src.distill --teacher-config configs/teachers.yaml \\
                          --student-config configs/students.yaml \\
                          --student shufflenetv2_x0_5 \\
                          --method csr_kd --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from .data import build_id_loaders, build_id_train_loader_indexed
from .models import build_model, RouterMLP, SelectiveResidualRouterMLP
from .utils import load_config
from .utils.feature_extract import FeatureExtractor
from .utils.kd_losses import multi_teacher_kd_loss
from .utils.train_loop import build_optimizer, build_scheduler, evaluate, seed_everything


# ---------- teacher loading ----------

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


def _fixed_weights(scheme: str, accs: torch.Tensor) -> torch.Tensor:
    if scheme == "uniform":
        w = torch.ones_like(accs)
    elif scheme == "accuracy":
        w = accs.clone()
    else:
        raise ValueError(f"_fixed_weights got scheme={scheme}")
    return w / w.sum()


# ---------- SLG gate cache ----------

def _load_slg_gate(cfg, teacher_names: List[str], device: str) -> torch.Tensor:
    """Load precomputed (N, M) SLG gate tensor. Asserts teacher list matches."""
    path = os.path.join(cfg["result_root"], cfg["id_dataset"], "slg_gate.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SLG gate cache not found at {path} — run `python -m src.precompute_slg` first."
        )
    obj = torch.load(path, map_location="cpu")
    if obj["teachers"] != teacher_names:
        raise ValueError(
            f"SLG cache teacher list {obj['teachers']} != configured teachers {teacher_names}"
        )
    return obj["gate"].to(device)  # (N, M)


# ---------- entropy helpers ----------

def _entropy(p: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return -(p * (p + eps).log()).sum(dim=dim)


def _conf_weight_from_slg(g_slg: torch.Tensor, M: int, conf_min: float) -> torch.Tensor:
    """w_sup(x) = max(conf_min, exp(-H(g_SLG) / log M))   (Eq. 2.27 in the thesis)."""
    H = _entropy(g_slg, dim=-1)
    w = torch.exp(-H / math.log(M))
    return torch.clamp(w, min=conf_min)


def _alpha_schedule(t_frac: float, alpha_start: float, alpha_end: float,
                    ramp_start: float, ramp_end: float) -> float:
    if t_frac <= ramp_start:
        return alpha_start
    if t_frac >= ramp_end:
        return alpha_end
    f = (t_frac - ramp_start) / max(1e-9, (ramp_end - ramp_start))
    return alpha_start + (alpha_end - alpha_start) * f


# ---------- one training run ----------

def run(args):
    s_cfg = load_config(args.student_config)
    t_cfg = load_config(args.teacher_config)
    s_spec = next(s for s in s_cfg["students"] if s["name"] == args.student)

    # Merge distill block into train block (epochs/lr/wd/scheduler).
    distill = s_cfg["distill"]
    s_cfg["train"] = {**s_cfg["train"], **{k: v for k, v in distill.items() if k in s_cfg["train"]}}

    # Method comes from CLI override, else from config.
    method = args.method or s_cfg["kd"].get("teacher_weighting", "uniform")
    seed = args.seed if args.seed is not None else s_cfg.get("seed", 42)

    device = s_cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[warn] CUDA unavailable, falling back to CPU")
    seed_everything(seed)

    # Output paths keyed by (student, method, seed) for clean multi-seed orchestration.
    tag = f"{s_spec['name']}__{method}__seed{seed}"
    ckpt_path = os.path.join(s_cfg["ckpt_root"], s_cfg["id_dataset"], "students", f"{tag}.pt")
    log_path = os.path.join(s_cfg["log_root"], s_cfg["id_dataset"], "students", f"{tag}.json")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if os.path.exists(ckpt_path) and not args.force:
        print(f"[skip] {tag} — checkpoint already exists at {ckpt_path}")
        return

    # ---- data ----
    # SLG / CSR-KD / SRR need per-sample indexing to look up the gate cache.
    needs_index = method in {"slg", "csr_kd", "srr"}
    if needs_index:
        train_loader = build_id_train_loader_indexed(s_cfg, train_aug=True)
        _, test_loader = build_id_loaders(s_cfg)
    else:
        train_loader, test_loader = build_id_loaders(s_cfg)

    # ---- student + teachers ----
    student = build_model(
        s_spec["arch"], num_classes=s_cfg["num_classes"],
        cifar_stem=s_spec.get("cifar_stem", True),
    ).to(device)
    teachers, accs = _load_teachers(t_cfg, distill["teachers"], device)
    M = len(teachers)
    teacher_names = distill["teachers"]

    # ---- per-method extras ----
    method_cfg = s_cfg.get("method", {}).get(method, {}) if method in s_cfg.get("method", {}) else {}

    extra_params: List[nn.Parameter] = []
    router: nn.Module | None = None
    feat_extractor: FeatureExtractor | None = None
    feat_dim = 0
    slg_gate_cache: torch.Tensor | None = None
    anchor_logits_cache: torch.Tensor | None = None
    learned_global = None

    if method in {"slg", "csr_kd", "srr"}:
        slg_gate_cache = _load_slg_gate(s_cfg, teacher_names, device)  # (N, M)
        print(f"  loaded SLG gate cache (N={slg_gate_cache.size(0)}, M={M})")

    if method == "learned_global":
        learned_global = nn.Parameter(torch.zeros(M, device=device))
        extra_params.append(learned_global)

    if method in {"csr_kd", "srr"}:
        # Probe feature dim with one dry forward.
        feat_extractor = FeatureExtractor(student, s_spec["arch"])
        with torch.no_grad():
            x_dummy = torch.zeros(1, 3, 32, 32, device=device)
            feats, _ = feat_extractor(x_dummy)
            feat_dim = feats.shape[1]
        if method == "csr_kd":
            router = RouterMLP(in_dim=feat_dim, num_experts=M, hidden_dim=256).to(device)
        else:
            r_prior = float(method_cfg.get("r_prior", 0.20))
            router = SelectiveResidualRouterMLP(
                in_dim=feat_dim, num_experts=M, hidden_dim=256,
                intervention_prior=r_prior,
            ).to(device)
        extra_params += list(router.parameters())

    if method == "srr":
        # Anchor logits = an SLG-trained student (same seed) forward over training set, cached on disk.
        anchor_logits_cache = _maybe_build_anchor_cache(
            s_cfg, s_spec, seed, device, train_loader
        )
        if anchor_logits_cache is None:
            print("[warn] SRR anchor disabled — no SLG checkpoint found for this (student, seed)")

    # ---- optimizer / scheduler ----
    optimizer = build_optimizer(student, s_cfg)
    if extra_params:
        # Use a smaller LR for router/global-gate params, mirroring the paper.
        extra_lr = float(method_cfg.get("extra_lr", 1.0e-3))
        optimizer.add_param_group({"params": extra_params, "lr": extra_lr, "weight_decay": 0.0})
    scheduler = build_scheduler(optimizer, s_cfg, steps_per_epoch=len(train_loader))
    scaler = GradScaler(enabled=s_cfg["train"].get("amp", True))

    alpha = float(distill.get("alpha", s_cfg["kd"]["alpha"]))
    T = float(distill.get("temperature", s_cfg["kd"]["temperature"]))
    ls = s_cfg["train"].get("label_smoothing", 0.0)

    # method-specific hyperparams (with sane defaults if not overridden in config)
    alpha_start = float(method_cfg.get("alpha_start", 0.20))
    alpha_end = float(method_cfg.get("alpha_end", 0.72))
    ramp_start = float(method_cfg.get("alpha_ramp_start_frac", 0.20))
    ramp_end = float(method_cfg.get("alpha_ramp_end_frac", 0.85))
    sup_start = float(method_cfg.get("sup_weight_start", 1.0))
    sup_end = float(method_cfg.get("sup_weight_end", 0.45))
    sup_ramp_start = float(method_cfg.get("sup_decay_start_frac", 0.20))
    sup_ramp_end = float(method_cfg.get("sup_decay_end_frac", 0.85))
    conf_min = float(method_cfg.get("confidence_min", 0.25))
    ent_bonus_w = float(method_cfg.get("entropy_bonus_weight", 0.005))
    anchor_w = float(method_cfg.get("anchor_weight", 0.50))
    anchor_conf = float(method_cfg.get("anchor_conf_threshold", 0.80))
    r_min = float(method_cfg.get("r_min", 0.05))
    r_max = float(method_cfg.get("r_max", 0.40))
    r_prior_val = float(method_cfg.get("r_prior", 0.20))
    r_prior_w = float(method_cfg.get("r_prior_weight", 0.20))

    # Fixed-weight schemes are precomputed once.
    fixed_w = None
    if method in {"uniform", "accuracy"}:
        fixed_w = _fixed_weights(method, accs)
        print(f"  fixed teacher weights = {fixed_w.cpu().tolist()}")

    history = []
    best_acc = 0.0
    epochs = s_cfg["train"]["epochs"]

    for epoch in range(epochs):
        student.train()
        if router is not None:
            router.train()
        t0 = time.time()
        ce_run = kd_run = aux_run = total_n = 0.0
        t_frac = epoch / max(1, epochs - 1)
        alpha_t = _alpha_schedule(t_frac, alpha_start, alpha_end, ramp_start, ramp_end)
        sup_t = _alpha_schedule(t_frac, sup_start, sup_end, sup_ramp_start, sup_ramp_end)

        for batch in train_loader:
            if needs_index:
                x, y, idx = batch
                idx_dev = idx.to(device, non_blocking=True)
            else:
                x, y = batch
                idx_dev = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=s_cfg["train"].get("amp", True)):
                with torch.no_grad():
                    teacher_logits = [t(x) for t in teachers]

                # Forward student (with feature extraction if needed).
                if feat_extractor is not None:
                    s_feats, s_logits = feat_extractor(x)
                    s_feats = s_feats.float()
                else:
                    s_logits = student(x)

                # ---- determine gate weights ----
                aux = s_logits.new_zeros(())
                if method in {"uniform", "accuracy"}:
                    weights = fixed_w
                elif method == "learned_global":
                    g = F.softmax(learned_global, dim=0)  # (M,)
                    weights = g
                    aux = aux - ent_bonus_w * _entropy(g)
                elif method == "slg":
                    weights = slg_gate_cache[idx_dev]  # (B, M)
                elif method == "csr_kd":
                    g_slg = slg_gate_cache[idx_dev]                       # (B, M)
                    g_router = F.softmax(router(s_feats), dim=-1)         # (B, M)
                    g_final = (1.0 - alpha_t) * g_slg + alpha_t * g_router
                    weights = g_final
                    # router supervision (conf-weighted KL(g_slg || g_router))
                    w_conf = _conf_weight_from_slg(g_slg, M, conf_min)
                    kl_per = (g_slg * (g_slg.clamp_min(1e-12).log() - g_router.clamp_min(1e-12).log())).sum(dim=-1)
                    aux = aux + sup_t * (w_conf * kl_per).mean()
                    # entropy bonus on router output (encourage non-collapse)
                    aux = aux - ent_bonus_w * _entropy(g_router, dim=-1).mean()
                elif method == "srr":
                    g_slg = slg_gate_cache[idx_dev]
                    gate_logits, r_logit = router(s_feats)
                    g_cand = F.softmax(gate_logits, dim=-1)
                    r = r_min + (r_max - r_min) * torch.sigmoid(r_logit)  # (B,)
                    g_final = (1.0 - r.unsqueeze(-1)) * g_slg + r.unsqueeze(-1) * g_cand
                    weights = g_final
                    # r-prior: pull mean r toward r_prior_val
                    aux = aux + r_prior_w * (r.mean() - r_prior_val) ** 2
                    # anchor loss (conditional on anchor's max-softmax confidence)
                    if anchor_logits_cache is not None:
                        a_logits = anchor_logits_cache[idx_dev].to(s_logits.dtype)
                        a_probs = F.softmax(a_logits, dim=-1)
                        mask = (a_probs.max(dim=-1).values >= anchor_conf).float()
                        if mask.sum() > 0:
                            mse = ((s_logits - a_logits) ** 2).mean(dim=-1)
                            aux = aux + anchor_w * (mask * mse).sum() / mask.sum()
                else:
                    raise ValueError(f"Unknown method {method}")

                # ---- losses ----
                ce_v = F.cross_entropy(s_logits, y, label_smoothing=ls)
                kd_v = multi_teacher_kd_loss(s_logits, teacher_logits, T=T, weights=weights)
                loss = (1.0 - alpha) * ce_v + alpha * kd_v + aux

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            n = x.size(0)
            ce_run += ce_v.item() * n
            kd_run += kd_v.item() * n
            aux_run += float(aux.detach()) * n
            total_n += n

        acc = evaluate(student, test_loader, device)
        rec = {
            "epoch": epoch,
            "ce": ce_run / total_n,
            "kd": kd_run / total_n,
            "aux": aux_run / total_n,
            "alpha_t": alpha_t,
            "sup_t": sup_t,
            "test_acc": acc,
            "time": time.time() - t0,
        }
        history.append(rec)
        print(
            f"[{tag}][epoch {epoch+1}/{epochs}] "
            f"ce={rec['ce']:.4f} kd={rec['kd']:.4f} aux={rec['aux']:.4f} "
            f"α_t={alpha_t:.3f} acc={acc*100:.2f}% ({rec['time']:.1f}s)",
            flush=True,
        )

        if acc > best_acc:
            best_acc = acc
            save_obj = {"model": student.state_dict(), "acc": acc, "epoch": epoch,
                        "method": method, "seed": seed}
            if router is not None:
                save_obj["router"] = router.state_dict()
            if learned_global is not None:
                save_obj["learned_global"] = learned_global.detach().cpu()
            torch.save(save_obj, ckpt_path)

    if feat_extractor is not None:
        feat_extractor.close()

    with open(log_path, "w") as f:
        json.dump({"best_acc": best_acc, "method": method, "seed": seed,
                   "teachers": teacher_names, "history": history}, f, indent=2)
    print(f"[done] {tag} best_acc={best_acc*100:.2f}% -> {ckpt_path}")


def _maybe_build_anchor_cache(s_cfg, s_spec, seed, device, train_loader_indexed):
    """For SRR: cache logits of the SLG-trained student of the same (student, seed) over training set.

    Returns (N, C) tensor on `device`, or None if SLG checkpoint is missing.
    """
    anchor_ckpt = os.path.join(
        s_cfg["ckpt_root"], s_cfg["id_dataset"], "students",
        f"{s_spec['name']}__slg__seed{seed}.pt",
    )
    if not os.path.exists(anchor_ckpt):
        return None
    cache_path = os.path.join(
        s_cfg["result_root"], s_cfg["id_dataset"],
        f"anchor_logits__{s_spec['name']}__seed{seed}.pt",
    )
    if os.path.exists(cache_path):
        print(f"  [srr] reusing cached anchor logits at {cache_path}")
        return torch.load(cache_path, map_location=device)

    print(f"  [srr] building anchor logit cache from {anchor_ckpt}")
    anchor = build_model(
        s_spec["arch"], num_classes=s_cfg["num_classes"],
        cifar_stem=s_spec.get("cifar_stem", True),
    )
    state = torch.load(anchor_ckpt, map_location="cpu")
    anchor.load_state_dict(state["model"])
    anchor.to(device).eval()
    for p in anchor.parameters():
        p.requires_grad_(False)

    # Use a no-aug indexed loader for the cache (deterministic and matches sample idx).
    noaug = build_id_train_loader_indexed(s_cfg, train_aug=False)
    n_total = len(noaug.dataset)
    num_classes = s_cfg["num_classes"]
    logits_cache = torch.zeros(n_total, num_classes, dtype=torch.float32, device=device)
    with torch.no_grad():
        for x, _y, idx in noaug:
            x = x.to(device, non_blocking=True)
            out = anchor(x).float()
            logits_cache[idx.long()] = out
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(logits_cache.cpu(), cache_path)
    print(f"  [srr] wrote anchor cache to {cache_path}")
    del anchor
    if device == "cuda":
        torch.cuda.empty_cache()
    return logits_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-config", required=True)
    p.add_argument("--student-config", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--method", default=None,
                   choices=[None, "uniform", "accuracy", "learned_global", "slg", "csr_kd", "srr"],
                   help="Override kd.teacher_weighting from the student config")
    p.add_argument("--seed", type=int, default=None,
                   help="Override seed from the student config")
    p.add_argument("--force", action="store_true",
                   help="Retrain even if the checkpoint already exists")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
