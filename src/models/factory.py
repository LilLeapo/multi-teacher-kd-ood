"""Architecture factory. All teachers + students are built from scratch (weights=None).

CIFAR stem adaptation: replace the 7x7 stride-2 stem and initial 3x3 maxpool with a 3x3
stride-1 conv. This is the standard CIFAR adaptation for ImageNet-class architectures
and is essential when training from 32x32 input — otherwise the feature map collapses
to 1x1 before reaching the final stage.
"""
from __future__ import annotations

from typing import List

import torch.nn as nn
from torchvision import models as tvm

from .repvgg import repvgg_a0

_TEACHER_BUILDERS = {
    "resnet50": lambda nc: tvm.resnet50(weights=None, num_classes=nc),
    "resnet18": lambda nc: tvm.resnet18(weights=None, num_classes=nc),
    "resnet34": lambda nc: tvm.resnet34(weights=None, num_classes=nc),
    "resnet101": lambda nc: tvm.resnet101(weights=None, num_classes=nc),
    "resnet152": lambda nc: tvm.resnet152(weights=None, num_classes=nc),
    "densenet121": lambda nc: tvm.densenet121(weights=None, num_classes=nc),
    "wide_resnet50_2": lambda nc: tvm.wide_resnet50_2(weights=None, num_classes=nc),
    "resnext50_32x4d": lambda nc: tvm.resnext50_32x4d(weights=None, num_classes=nc),
    "efficientnet_b0": lambda nc: tvm.efficientnet_b0(weights=None, num_classes=nc),
}

_STUDENT_BUILDERS = {
    "shufflenetv2_x0_5": lambda nc: tvm.shufflenet_v2_x0_5(weights=None, num_classes=nc),
    "repvgg_a0": lambda nc: repvgg_a0(num_classes=nc, cifar_stem=True),
}

_ALL = {**_TEACHER_BUILDERS, **_STUDENT_BUILDERS}


def list_supported() -> List[str]:
    return sorted(_ALL.keys())


def build_model(arch: str, num_classes: int, cifar_stem: bool = True) -> nn.Module:
    if arch not in _ALL:
        raise ValueError(f"Unknown arch {arch}. Supported: {list_supported()}")
    model = _ALL[arch](num_classes)
    if cifar_stem:
        _apply_cifar_stem(model, arch)
    return model


def _apply_cifar_stem(model: nn.Module, arch: str) -> None:
    """In-place rewrite of the stem so 32x32 inputs aren't downsampled to oblivion."""
    if arch.startswith(("resnet", "wide_resnet", "resnext")):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return

    if arch.startswith("densenet"):
        features = model.features
        out_c = features.conv0.out_channels
        features.conv0 = nn.Conv2d(3, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        features.pool0 = nn.Identity()
        return

    if arch.startswith("efficientnet"):
        # First Conv2dNormActivation in features[0]; its first child is the conv.
        first_block = model.features[0]
        old_conv = first_block[0]
        first_block[0] = nn.Conv2d(
            old_conv.in_channels,
            old_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        return

    if arch.startswith("shufflenet"):
        old = model.conv1[0]
        model.conv1[0] = nn.Conv2d(
            old.in_channels, old.out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
        return

    if arch.startswith("repvgg"):
        # RepVGG is built with cifar_stem=True by default — nothing to do.
        return

    raise NotImplementedError(f"CIFAR stem adaptation not implemented for {arch}")
