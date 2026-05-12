"""Minimal RepVGG-A0 (training-time multi-branch) — no deploy-time fusion needed."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn(in_c: int, out_c: int, k: int, stride: int, padding: int, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_c),
    )


class RepVGGBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1, groups: int = 1):
        super().__init__()
        self.nonlinearity = nn.ReLU(inplace=True)
        self.identity = nn.BatchNorm2d(in_c) if (in_c == out_c and stride == 1) else None
        self.conv3x3 = _conv_bn(in_c, out_c, 3, stride, 1, groups)
        self.conv1x1 = _conv_bn(in_c, out_c, 1, stride, 0, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv3x3(x) + self.conv1x1(x)
        if self.identity is not None:
            y = y + self.identity(x)
        return self.nonlinearity(y)


class RepVGG(nn.Module):
    """RepVGG-A0: width multipliers a=0.75, b=2.5; stages [2,4,14,1]."""

    def __init__(self, num_blocks=(2, 4, 14, 1), width=(48, 48, 96, 192, 1280), num_classes: int = 100, cifar_stem: bool = True):
        super().__init__()
        in_planes = width[0]
        stem_stride = 1 if cifar_stem else 2
        self.stage0 = RepVGGBlock(3, in_planes, stride=stem_stride)
        self.stage1 = self._make_stage(in_planes, width[1], num_blocks[0], stride=2)
        self.stage2 = self._make_stage(width[1], width[2], num_blocks[1], stride=2)
        self.stage3 = self._make_stage(width[2], width[3], num_blocks[2], stride=2)
        self.stage4 = self._make_stage(width[3], width[4], num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(width[4], num_classes)

    def _make_stage(self, in_c: int, out_c: int, n: int, stride: int) -> nn.Sequential:
        layers = [RepVGGBlock(in_c, out_c, stride=stride)]
        for _ in range(n - 1):
            layers.append(RepVGGBlock(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        return self.linear(x)


def repvgg_a0(num_classes: int = 100, cifar_stem: bool = True) -> RepVGG:
    return RepVGG(num_blocks=(2, 4, 14, 1), width=(48, 48, 96, 192, 1280),
                  num_classes=num_classes, cifar_stem=cifar_stem)
