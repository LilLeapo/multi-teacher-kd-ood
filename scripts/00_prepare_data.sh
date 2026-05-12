#!/usr/bin/env bash
# Trigger the torchvision download paths so subsequent steps don't race on disk I/O.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

log "preparing CIFAR-100 / CIFAR-10 / SVHN / DTD into $REPO_ROOT/data"
python - <<'PY'
import os
from torchvision import datasets
root = "data"
os.makedirs(root, exist_ok=True)
datasets.CIFAR100(root, train=True, download=True)
datasets.CIFAR100(root, train=False, download=True)
datasets.CIFAR10(root, train=True, download=True)
datasets.CIFAR10(root, train=False, download=True)
datasets.SVHN(os.path.join(root, "svhn"), split="test", download=True)
datasets.DTD(os.path.join(root, "dtd"), split="test", partition=1, download=True)
print("[done] datasets ready")
PY
