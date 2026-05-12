"""ID and OOD datasets. All resized to 32x32 to match the CIFAR-trained backbones."""
from __future__ import annotations

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)

NUM_CLASSES = {"cifar100": 100, "cifar10": 10}


def _train_tf(cfg) -> transforms.Compose:
    aug = cfg.get("augment", {})
    ops = []
    pad = aug.get("random_crop_padding", 4)
    if pad:
        ops.append(transforms.RandomCrop(32, padding=pad, padding_mode="reflect"))
    if aug.get("random_horizontal_flip", True):
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    if aug.get("cutout", False):
        ops.append(_Cutout(length=16))
    return transforms.Compose(ops)


def _eval_tf() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


class _Cutout:
    def __init__(self, length: int = 16):
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)
        y = torch.randint(0, h, (1,)).item()
        x = torch.randint(0, w, (1,)).item()
        y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
        x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img


def _id_dataset(name: str, root: str, train: bool, tf) -> Dataset:
    if name == "cifar100":
        return datasets.CIFAR100(root=root, train=train, transform=tf, download=True)
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=train, transform=tf, download=True)
    raise ValueError(f"Unknown ID dataset {name}")


def _ood_dataset(name: str, root: str, split: str, tf) -> Dataset:
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=False, transform=tf, download=True)
    if name == "cifar100":
        return datasets.CIFAR100(root=root, train=False, transform=tf, download=True)
    if name == "svhn":
        return datasets.SVHN(root=os.path.join(root, "svhn"), split="test", transform=tf, download=True)
    if name == "textures":
        return datasets.DTD(
            root=os.path.join(root, "dtd"),
            split="test",
            partition=1,
            transform=tf,
            download=True,
        )
    raise ValueError(f"Unknown OOD dataset {name}")


def build_id_loaders(cfg) -> Tuple[DataLoader, DataLoader]:
    name = cfg["id_dataset"]
    root = cfg["data_root"]
    train_ds = _id_dataset(name, root, train=True, tf=_train_tf(cfg))
    test_ds = _id_dataset(name, root, train=False, tf=_eval_tf())
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        drop_last=True,
        persistent_workers=cfg["num_workers"] > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        persistent_workers=cfg["num_workers"] > 0,
    )
    return train_loader, test_loader


def build_ood_loader(name: str, cfg) -> DataLoader:
    ds = _ood_dataset(name, cfg["data_root"], split="test", tf=_eval_tf())
    return DataLoader(
        ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )
