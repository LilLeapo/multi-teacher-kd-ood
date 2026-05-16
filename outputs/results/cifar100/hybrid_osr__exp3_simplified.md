# Hybrid OSR report — `exp3_simplified` (ID=cifar100)

- Teachers (5): `resnet50`, `densenet121`, `wide_resnet50_2`, `resnext50_32x4d`, `convnext_tiny`
- Student arch: `shufflenetv2_x0_5`
- OOD: `cifar10`, `svhn`, `textures`
- Energy T = 1.0

## T3 — standalone student OSR (mean ± std across seeds)

| method | n | ID acc | cifar10 AUROC | svhn AUROC | textures AUROC | cifar10 FPR95 | svhn FPR95 | textures FPR95 |
|---|---|---|---|---|---|---|---|---|
| uniform | 3 | 69.94% ± 0.33% | 0.7373 ± 0.0047 | 0.8376 ± 0.0222 | 0.7672 ± 0.0015 | 0.8412 ± 0.0073 | 0.7018 ± 0.0604 | 0.7683 ± 0.0107 |
| accuracy | 3 | 69.37% ± 0.34% | 0.7380 ± 0.0030 | 0.8215 ± 0.0287 | 0.7604 ± 0.0020 | 0.8356 ± 0.0054 | 0.7142 ± 0.0634 | 0.7791 ± 0.0040 |
| learned_global | 3 | 67.87% ± 0.17% | 0.7259 ± 0.0032 | 0.7310 ± 0.0551 | 0.7376 ± 0.0152 | 0.8619 ± 0.0049 | 0.9040 ± 0.0389 | 0.8504 ± 0.0298 |

## Hybrid baseline (teacher ensemble Energy)

- Ensemble ID acc: 85.08%

| OOD | AUROC | FPR95 |
|---|---|---|
| `cifar10` | 0.8442 | 0.6981 |
| `svhn` | 0.8413 | 0.7019 |
| `textures` | 0.8222 | 0.7266 |
