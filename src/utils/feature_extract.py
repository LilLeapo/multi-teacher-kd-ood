"""Extract penultimate features and logits with a single forward pass.

Implementation strategy: register a forward hook on the model's classifier head.
The hook captures the input (= flat penultimate features after pool) and we get
the logits as the model's normal output. Works in eval mode where dropout is
a no-op, so the captured features match what a pretrained-features baseline
would see.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Tuple

import torch
import torch.nn as nn


def _classifier_head(model: nn.Module, arch: str) -> nn.Linear:
    if arch.startswith(("resnet", "wide_resnet", "resnext")):
        return model.fc
    if arch.startswith("densenet"):
        return model.classifier
    if arch.startswith("efficientnet"):
        cls = model.classifier
        return cls[-1] if isinstance(cls, nn.Sequential) else cls
    if arch.startswith("shufflenet"):
        return model.fc
    if arch.startswith("repvgg"):
        return model.linear
    raise ValueError(f"No known classifier head for arch={arch}")


class FeatureExtractor:
    """Wraps a model so calling it returns (features, logits)."""

    def __init__(self, model: nn.Module, arch: str):
        self.model = model
        self.head = _classifier_head(model, arch)
        self._buf = {}
        self._handle = self.head.register_forward_hook(self._hook)

    def _hook(self, _module, inputs, _output):
        # inputs[0] is the input to the head: the flat penultimate feature
        self._buf["features"] = inputs[0].detach()

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._buf.clear()
        logits = self.model(x)
        return self._buf["features"], logits

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def head_weight_bias(model: nn.Module, arch: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (W, b) of the final Linear head — used by ViM."""
    head = _classifier_head(model, arch)
    return head.weight.detach(), head.bias.detach()


@contextmanager
def extract(model: nn.Module, arch: str):
    fx = FeatureExtractor(model, arch)
    try:
        yield fx
    finally:
        fx.close()
