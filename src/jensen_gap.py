"""Teacher / student logit cache + Jensen-gap analysis (new main line, EXP 1 + EXP 2).

Two subcommands:

    dump      forward a list of models over {ID test, OOD test sets}, cache logits.
    analyze   read the cache and produce:
                - per-model Energy AUROC / FPR95 (table T1 / T2 rows)
                - ensemble Energy AUROC / FPR95 (uniform logit averaging)
                - per-sample Jensen gap Δ(x) = mean_i s_i(x) - s_ens(x)
                  with mean / median / p90 / p99 per split
                - E_OOD[Δ] / E_ID[Δ] ratio per OOD set
                - 4-split overlaid Δ histogram (PNG)

Usage:

    python -m src.jensen_gap dump \\
        --config configs/teachers.yaml --kind teacher \\
        --models resnet50 densenet121 wide_resnet50_2 resnext50_32x4d convnext_tiny

    python -m src.jensen_gap analyze \\
        --config configs/teachers.yaml \\
        --models resnet50 densenet121 wide_resnet50_2 resnext50_32x4d convnext_tiny \\
        --tag heterogeneous

Cache layout (kind=teacher, ID=cifar100):

    outputs/logit_cache/cifar100/teachers/{model}__{split}.pt
        { "logits": float16 [N, C], "labels": int64 [N] or None, "acc": float|None }

`split` is one of: `id_test`, `cifar10`, `svhn`, `textures` (anything in cfg["ood_eval"]
plus the ID test set).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import build_id_loaders, build_ood_loader
from .models import build_model
from .utils import load_config
from .utils.ood_metrics import fpr_at_tpr, ood_metrics


# ----------------------------- paths -----------------------------

def _cache_dir(cfg: dict, kind: str) -> str:
    """kind ∈ {teacher, student}; ID dataset name is read from cfg."""
    sub = "teachers" if kind == "teacher" else "students"
    return os.path.join(cfg.get("logit_cache_root", "outputs/logit_cache"),
                        cfg["id_dataset"], sub)


def _cache_path(cfg: dict, kind: str, model_name: str, split: str) -> str:
    return os.path.join(_cache_dir(cfg, kind), f"{model_name}__{split}.pt")


def _result_prefix(cfg: dict, tag: str) -> str:
    out_dir = os.path.join(cfg["result_root"], cfg["id_dataset"])
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"jensen_gap__{tag}")


# ----------------------------- model loading -----------------------------

def _ckpt_path_for(cfg: dict, kind: str, name: str, spec: dict | None = None) -> str:
    """Where the trained checkpoint for `name` lives.

    Teachers:  checkpoints/<id>/teachers/<name>.pt
    Students:  checkpoints/<id>/students/<name>.pt   (name = full distill tag, e.g.
               shufflenetv2_x0_5__uniform__seed42)
    """
    sub = "teachers" if kind == "teacher" else "students"
    return os.path.join(cfg["ckpt_root"], cfg["id_dataset"], sub, f"{name}.pt")


def _resolve_spec(cfg: dict, kind: str, name: str) -> dict:
    """Look up a model's arch / cifar_stem from the appropriate config block.

    For teachers we expect a `teachers` list in cfg. For students the caller
    must supply `--arch` explicitly because student "names" are full distill tags.
    """
    if kind == "teacher":
        for spec in cfg.get("teachers", []):
            if spec["name"] == name:
                return spec
        raise ValueError(f"teacher '{name}' not found in config")
    # student: synthesised spec; caller fills arch via CLI.
    raise ValueError("student spec lookup must use --arch")


# ----------------------------- dataset loaders -----------------------------

def _all_splits(cfg: dict) -> List[str]:
    return ["id_test"] + [o["name"] for o in cfg["ood_eval"]]


def _loader_for_split(split: str, cfg: dict) -> Tuple[DataLoader, bool]:
    """Return (loader, has_labels). ID test has labels; OOD loaders may too,
    but we don't rely on them for OOD scoring."""
    if split == "id_test":
        _, loader = build_id_loaders(cfg)
        return loader, True
    return build_ood_loader(split, cfg), False


# ----------------------------- dump -----------------------------

@torch.no_grad()
def _dump_one(model: torch.nn.Module, loader: DataLoader, device: str,
              has_labels: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[float]]:
    model.eval()
    logits_buf, labels_buf = [], []
    correct = total = 0
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        logits = model(x).float()
        logits_buf.append(logits.cpu().to(torch.float16))
        if has_labels and len(batch) > 1:
            y = batch[1]
            labels_buf.append(y)
            correct += (logits.argmax(dim=-1).cpu() == y).sum().item()
            total += y.size(0)
    logits = torch.cat(logits_buf, dim=0)
    labels = torch.cat(labels_buf, dim=0) if labels_buf else None
    acc = correct / total if total > 0 else None
    return logits, labels, acc


def cmd_dump(args):
    cfg = load_config(args.config)
    device = cfg["device"] if torch.cuda.is_available() or cfg["device"] != "cuda" else "cuda"
    cache_dir = _cache_dir(cfg, args.kind)
    os.makedirs(cache_dir, exist_ok=True)

    splits = _all_splits(cfg)
    # Build loaders once (datasets are cheap to re-iterate).
    loaders = {s: _loader_for_split(s, cfg) for s in splits}

    for name in args.models:
        # decide which splits still need dumping
        todo = []
        for s in splits:
            path = _cache_path(cfg, args.kind, name, s)
            if os.path.exists(path) and not args.force:
                print(f"[skip] {name} / {s} already cached -> {path}")
                continue
            todo.append(s)
        if not todo:
            continue

        # resolve arch
        if args.kind == "teacher":
            spec = _resolve_spec(cfg, "teacher", name)
            arch = spec["arch"]
            cifar_stem = spec.get("cifar_stem", True)
            model_kwargs = spec.get("model_kwargs", None)
        else:
            assert args.arch is not None, "student dump needs --arch"
            arch = args.arch
            cifar_stem = True
            model_kwargs = None

        ckpt_path = _ckpt_path_for(cfg, args.kind, name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"checkpoint missing for {name}: {ckpt_path}")

        print(f"[load] {name}  arch={arch}  ckpt={ckpt_path}")
        model = build_model(arch, num_classes=cfg["num_classes"],
                            cifar_stem=cifar_stem, model_kwargs=model_kwargs)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"])
        model.to(device).eval()

        for s in todo:
            loader, has_labels = loaders[s]
            logits, labels, acc = _dump_one(model, loader, device, has_labels)
            payload = {"logits": logits, "labels": labels, "acc": acc,
                       "model": name, "split": s, "arch": arch}
            out_path = _cache_path(cfg, args.kind, name, s)
            torch.save(payload, out_path)
            n = logits.size(0)
            extra = f" acc={acc*100:.2f}%" if acc is not None else ""
            print(f"  [dump] {name} / {s}: N={n}{extra} -> {out_path}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()


# ----------------------------- analyze -----------------------------

def _energy(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """s(x) = T * logsumexp(z / T). Higher = more ID-like."""
    return T * torch.logsumexp(logits.float() / T, dim=-1)


def _load_logits(cfg: dict, kind: str, name: str, split: str) -> torch.Tensor:
    path = _cache_path(cfg, kind, name, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing logit cache: {path} (run `dump` first)")
    return torch.load(path, map_location="cpu")["logits"].float()


def _delta_stats(delta: torch.Tensor) -> Dict[str, float]:
    d = delta.cpu().numpy().astype(np.float64)
    return {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "p90": float(np.percentile(d, 90)),
        "p99": float(np.percentile(d, 99)),
        "std": float(d.std()),
        "n": int(d.size),
    }


def _format_table(headers: List[str], rows: List[List[str]]) -> Tuple[str, str]:
    """Return (plain_text, markdown). Both are right-padded for readability."""
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    pt = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    pt += "\n" + "-+-".join("-" * w for w in widths) + "\n"
    pt += "\n".join(" | ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)
    md = "| " + " | ".join(headers) + " |\n"
    md += "|" + "|".join("---" for _ in headers) + "|\n"
    md += "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return pt, md


def cmd_analyze(args):
    cfg = load_config(args.config)
    id_name = cfg["id_dataset"]
    ood_names = [o["name"] for o in cfg["ood_eval"]]
    T = float(args.energy_T)
    M = len(args.models)

    # ---- load all logits (model × split) ----
    by_split: Dict[str, Dict[str, torch.Tensor]] = {}
    for s in ["id_test"] + ood_names:
        by_split[s] = {name: _load_logits(cfg, args.kind, name, s) for name in args.models}

    # ---- per-model Energy scores ----
    per_model_energy = {s: {name: _energy(t, T) for name, t in d.items()}
                        for s, d in by_split.items()}
    # ---- ensemble: average logits, then Energy ----
    ensemble_logits = {s: torch.stack([d[name] for name in args.models]).mean(dim=0)
                       for s, d in by_split.items()}
    ensemble_energy = {s: _energy(t, T) for s, t in ensemble_logits.items()}
    # ---- mean of per-model energies (used by Jensen gap definition) ----
    mean_per_model_energy = {s: torch.stack(
        [per_model_energy[s][name] for name in args.models]
    ).mean(dim=0) for s in per_model_energy}

    # ---- Jensen gap Δ(x) = mean_i s_i(x) - s_ens(x) (≥ 0 by Jensen) ----
    delta = {s: (mean_per_model_energy[s] - ensemble_energy[s]) for s in by_split}

    # ---- AUROC / FPR95 per model and ensemble ----
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    id_scores_per_model = {name: per_model_energy["id_test"][name].numpy()
                           for name in args.models}
    id_scores_ens = ensemble_energy["id_test"].numpy()
    for ood in ood_names:
        m_block = {}
        for name in args.models:
            ood_scores = per_model_energy[ood][name].numpy()
            m_block[name] = ood_metrics(id_scores_per_model[name], ood_scores)
        ens_ood_scores = ensemble_energy[ood].numpy()
        m_block["__ensemble__"] = ood_metrics(id_scores_ens, ens_ood_scores)
        metrics[ood] = m_block

    # ---- Δ stats per split ----
    delta_stats = {s: _delta_stats(d) for s, d in delta.items()}
    id_mean = delta_stats["id_test"]["mean"]
    ood_ratio = {ood: delta_stats[ood]["mean"] / max(1e-12, id_mean) for ood in ood_names}

    # ---- per-model and ensemble accuracy on ID test (for context) ----
    id_acc = {}
    for name in args.models:
        cache = torch.load(_cache_path(cfg, args.kind, name, "id_test"), map_location="cpu")
        id_acc[name] = cache.get("acc", None)
    # Ensemble ID acc from averaged logits.
    id_labels_path = _cache_path(cfg, args.kind, args.models[0], "id_test")
    id_labels = torch.load(id_labels_path, map_location="cpu").get("labels")
    if id_labels is not None:
        ens_acc = (ensemble_logits["id_test"].argmax(dim=-1) == id_labels).float().mean().item()
    else:
        ens_acc = None

    # ----------------------------- T1 table -----------------------------
    headers = ["Model"] + [f"{ood} AUROC" for ood in ood_names] \
              + [f"{ood} FPR95" for ood in ood_names] + ["ID acc"]
    rows = []
    for name in args.models:
        r = [name]
        for ood in ood_names:
            r.append(f"{metrics[ood][name]['auroc']:.4f}")
        for ood in ood_names:
            r.append(f"{metrics[ood][name]['fpr95']:.4f}")
        r.append(f"{id_acc[name]*100:.2f}%" if id_acc[name] is not None else "—")
        rows.append(r)
    # ensemble row
    r = ["ensemble"]
    for ood in ood_names:
        r.append(f"{metrics[ood]['__ensemble__']['auroc']:.4f}")
    for ood in ood_names:
        r.append(f"{metrics[ood]['__ensemble__']['fpr95']:.4f}")
    r.append(f"{ens_acc*100:.2f}%" if ens_acc is not None else "—")
    rows.append(r)
    table_t1_pt, table_t1_md = _format_table(headers, rows)

    # ----------------------------- Δ table -----------------------------
    d_hdr = ["Split", "mean Δ", "median Δ", "p90 Δ", "p99 Δ", "std", "N", "mean / ID mean"]
    d_rows = []
    for s in ["id_test"] + ood_names:
        st = delta_stats[s]
        ratio = "1.000" if s == "id_test" else f"{ood_ratio[s]:.3f}"
        d_rows.append([s, f"{st['mean']:.4f}", f"{st['median']:.4f}",
                       f"{st['p90']:.4f}", f"{st['p99']:.4f}",
                       f"{st['std']:.4f}", str(st['n']), ratio])
    table_d_pt, table_d_md = _format_table(d_hdr, d_rows)

    # ----------------------------- console -----------------------------
    print()
    print(f"=== Jensen-gap report ({args.kind}, tag={args.tag}, ID={id_name}, M={M}, T={T}) ===")
    print(f"Models: {args.models}")
    print()
    print(f"[T1] Per-model + ensemble Energy AUROC / FPR95")
    print(table_t1_pt)
    print()
    print(f"[Δ] Jensen gap Δ(x) = mean_i s_i(x) - s_ens(x)")
    print(table_d_pt)
    print()
    print(f"Key headline: E_OOD[Δ] / E_ID[Δ] per OOD set:")
    for ood in ood_names:
        print(f"  {ood:>10s}: {ood_ratio[ood]:.3f}x")

    # ----------------------------- save artifacts -----------------------------
    prefix = _result_prefix(cfg, args.tag)

    payload = {
        "tag": args.tag,
        "kind": args.kind,
        "id_dataset": id_name,
        "ood": ood_names,
        "models": args.models,
        "energy_T": T,
        "metrics": {ood: {name: metrics[ood][name] for name in metrics[ood]} for ood in ood_names},
        "delta_stats": delta_stats,
        "ood_ratio": ood_ratio,
        "id_acc_per_model": id_acc,
        "ensemble_id_acc": ens_acc,
    }
    with open(prefix + ".json", "w") as f:
        json.dump(payload, f, indent=2)

    md = []
    md.append(f"# Jensen-gap report — tag=`{args.tag}` ({args.kind}, M={M})")
    md.append("")
    md.append(f"- ID dataset: `{id_name}`")
    md.append(f"- OOD: {', '.join(f'`{o}`' for o in ood_names)}")
    md.append(f"- Energy temperature T = {T}")
    md.append(f"- Models: {', '.join(f'`{m}`' for m in args.models)}")
    md.append("")
    md.append("## T1 — Per-model + ensemble Energy AUROC / FPR95")
    md.append("")
    md.append(table_t1_md)
    md.append("")
    md.append("## Δ(x) statistics per split")
    md.append("")
    md.append(table_d_md)
    md.append("")
    md.append("### E_OOD[Δ] / E_ID[Δ] (headline number for new main line)")
    md.append("")
    md.append("| OOD set | ratio |")
    md.append("|---|---|")
    for ood in ood_names:
        md.append(f"| `{ood}` | {ood_ratio[ood]:.3f} |")
    with open(prefix + ".md", "w") as f:
        f.write("\n".join(md) + "\n")

    # ----------------------------- F1 histogram -----------------------------
    _plot_histograms(delta, ["id_test"] + ood_names, prefix + "__hist.png", args.tag)

    print(f"\n[done] wrote:\n  {prefix}.json\n  {prefix}.md\n  {prefix}__hist.png")


def _plot_histograms(delta_by_split: Dict[str, torch.Tensor],
                     order: List[str], out_path: str, tag: str) -> None:
    """4-split overlaid histogram of Δ(x). Saved as PNG (matplotlib, no seaborn)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available; skipping histogram")
        return

    # Times New Roman for paper-ready figures; STIX for math glyphs so the Δ /
    # subscripts in the title match the body font. The fallback chain handles
    # systems without TNR installed.
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = [
        "Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif",
    ]
    matplotlib.rcParams["mathtext.fontset"] = "stix"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"id_test": "tab:blue", "cifar10": "tab:orange",
              "svhn": "tab:green", "textures": "tab:red"}
    # Common bin range: 0..max p99 across splits, log-scale-friendly.
    all_vals = torch.cat([delta_by_split[s] for s in order]).cpu().numpy()
    hi = float(np.percentile(all_vals, 99.5)) * 1.05
    hi = max(hi, 1e-3)
    bins = np.linspace(0.0, hi, 80)
    for s in order:
        d = delta_by_split[s].cpu().numpy()
        ax.hist(d, bins=bins, density=True, alpha=0.45,
                label=f"{s} (μ={d.mean():.3f})",
                color=colors.get(s, None))
    ax.set_xlabel(r"$\Delta(x) = \overline{s_i(x)} - s_{\mathrm{ens}}(x)$")
    ax.set_ylabel("density")
    ax.set_title(f"Jensen gap Δ distribution — {tag}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    # Δ ≥ 0 by Jensen's inequality; pin the x-axis origin to suppress matplotlib's
    # default left margin that would otherwise show a sliver of x<0.
    ax.set_xlim(left=0.0)
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ----------------------------- CLI -----------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("dump", help="forward models over splits and cache logits")
    pd.add_argument("--config", required=True)
    pd.add_argument("--kind", choices=["teacher", "student"], default="teacher")
    pd.add_argument("--models", nargs="+", required=True)
    pd.add_argument("--arch", default=None, help="required when --kind student")
    pd.add_argument("--force", action="store_true")
    pd.set_defaults(func=cmd_dump)

    pa = sub.add_parser("analyze", help="produce T1/Δ tables + histogram from cache")
    pa.add_argument("--config", required=True)
    pa.add_argument("--kind", choices=["teacher", "student"], default="teacher")
    pa.add_argument("--models", nargs="+", required=True)
    pa.add_argument("--tag", required=True, help="output filename tag, e.g. heterogeneous")
    pa.add_argument("--energy-T", type=float, default=1.0)
    pa.set_defaults(func=cmd_analyze)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
