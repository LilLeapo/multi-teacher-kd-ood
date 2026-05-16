# Jensen-gap report — tag=`heterogeneous` (teacher, M=5)

- ID dataset: `cifar100`
- OOD: `cifar10`, `svhn`, `textures`
- Energy temperature T = 1.0
- Models: `resnet50`, `densenet121`, `wide_resnet50_2`, `resnext50_32x4d`, `convnext_tiny`

## T1 — Per-model + ensemble Energy AUROC / FPR95

| Model | cifar10 AUROC | svhn AUROC | textures AUROC | cifar10 FPR95 | svhn FPR95 | textures FPR95 | ID acc |
|---|---|---|---|---|---|---|---|
| resnet50 | 0.8136 | 0.7790 | 0.7772 | 0.7640 | 0.8161 | 0.8048 | 81.19% |
| densenet121 | 0.8096 | 0.8510 | 0.7916 | 0.7675 | 0.6632 | 0.7436 | 81.32% |
| wide_resnet50_2 | 0.8175 | 0.7800 | 0.7618 | 0.7429 | 0.7780 | 0.8005 | 82.07% |
| resnext50_32x4d | 0.8136 | 0.7625 | 0.7811 | 0.7477 | 0.8148 | 0.8138 | 82.08% |
| convnext_tiny | 0.6894 | 0.5830 | 0.7106 | 0.8324 | 0.9036 | 0.7713 | 81.58% |
| ensemble | 0.8442 | 0.8413 | 0.8222 | 0.6981 | 0.7019 | 0.7266 | 85.08% |

## Δ(x) statistics per split

| Split | mean Δ | median Δ | p90 Δ | p99 Δ | std | N | mean / ID mean |
|---|---|---|---|---|---|---|---|
| id_test | 0.2005 | 0.0360 | 0.6696 | 1.1788 | 0.2915 | 10000 | 1.000 |
| cifar10 | 0.5222 | 0.5234 | 0.9262 | 1.3456 | 0.3165 | 10000 | 2.605 |
| svhn | 0.6974 | 0.6911 | 1.1341 | 1.5379 | 0.3390 | 26032 | 3.479 |
| textures | 0.5855 | 0.5786 | 1.0521 | 1.5610 | 0.3716 | 1880 | 2.920 |

### E_OOD[Δ] / E_ID[Δ] (headline number for new main line)

| OOD set | ratio |
|---|---|
| `cifar10` | 2.605 |
| `svhn` | 3.479 |
| `textures` | 2.920 |
