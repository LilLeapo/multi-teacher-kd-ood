"""Tiny YAML loader with `extends:` support and per-name `overrides:` merging."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str, override_name: str | None = None) -> Dict[str, Any]:
    """Load a YAML config, resolving `extends:` relative to the config file's dir.

    If `override_name` is given, the matching subtree under `overrides:` is merged in.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(path), parent)
        base = load_config(parent_path)
        cfg = _deep_merge(base, cfg)
    if override_name:
        overrides = cfg.get("overrides", {})
        if override_name in overrides:
            cfg = _deep_merge(cfg, overrides[override_name])
    return cfg
