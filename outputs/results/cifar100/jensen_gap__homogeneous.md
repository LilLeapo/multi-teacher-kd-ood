# Jensen-gap report — tag=`homogeneous` (teacher, M=5)

- ID dataset: `cifar100`
- OOD: `cifar10`, `svhn`, `textures`
- Energy temperature T = 1.0
- Models: `resnet50_seed42`, `resnet50_seed123`, `resnet50_seed3407`, `resnet50_seed7`, `resnet50_seed2024`

## T1 — Per-model + ensemble Energy AUROC / FPR95

| Model | cifar10 AUROC | svhn AUROC | textures AUROC | cifar10 FPR95 | svhn FPR95 | textures FPR95 | ID acc |
|---|---|---|---|---|---|---|---|
| resnet50_seed42 | 0.8153 | 0.7960 | 0.7724 | 0.7445 | 0.7704 | 0.7904 | 81.21% |
| resnet50_seed123 | 0.8008 | 0.8345 | 0.7844 | 0.7503 | 0.7679 | 0.7830 | 81.12% |
| resnet50_seed3407 | 0.8106 | 0.7841 | 0.7819 | 0.7574 | 0.8376 | 0.7947 | 81.46% |
| resnet50_seed7 | 0.8013 | 0.7490 | 0.7811 | 0.7649 | 0.8621 | 0.8106 | 80.79% |
| resnet50_seed2024 | 0.8170 | 0.7204 | 0.7728 | 0.7483 | 0.8909 | 0.8229 | 80.97% |
| ensemble | 0.8311 | 0.8076 | 0.8050 | 0.7300 | 0.7978 | 0.7846 | 83.10% |

## Δ(x) statistics per split

| Split | mean Δ | median Δ | p90 Δ | p99 Δ | std | N | mean / ID mean |
|---|---|---|---|---|---|---|---|
| id_test | 0.1149 | 0.0076 | 0.3940 | 0.7818 | 0.1883 | 10000 | 1.000 |
| cifar10 | 0.2991 | 0.2854 | 0.5788 | 0.9120 | 0.2149 | 10000 | 2.602 |
| svhn | 0.3758 | 0.3524 | 0.7430 | 1.1603 | 0.2791 | 26032 | 3.269 |
| textures | 0.3352 | 0.3159 | 0.6701 | 1.1057 | 0.2650 | 1880 | 2.916 |

### E_OOD[Δ] / E_ID[Δ] (headline number for new main line)

| OOD set | ratio |
|---|---|
| `cifar10` | 2.602 |
| `svhn` | 3.269 |
| `textures` | 2.916 |
