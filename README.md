# exp2 — multi-teacher KD, architecture-diverse pool

CIFAR-100 ID with 5 capability-balanced, architecture-diverse teachers distilled
into a heterogeneous student (ShuffleNetV2 0.5× or RepVGG-A0). The pool covers
four CNN-residual variants (ResNet-50, DenseNet-121, WRN-50-2, ResNeXt-50) and
ConvNeXt-Tiny as a modern-conv outsider (depthwise 7×7, LayerNorm, GELU,
stochastic depth) to break the CNN-residual monoculture. OOD evaluation against
CIFAR-10 (near), SVHN (far), Textures/DTD (far).

A negative-control variant (CIFAR-10 ID + capability-heterogeneous same-family
ResNets + same-architecture student) is included to expose the N1 boundary condition.

Models are trained from scratch on a CUDA box — nothing is downloaded as pretrained
weights, nothing trains locally on the Mac that authored this repo.

## Quick start (on the GPU host)

```bash
git clone <this-repo> exp2 && cd exp2
INSTALL=1 bash scripts/run_all.sh
```

Useful toggles:
- `INTRA_GPU=N`         run N teacher trainings concurrently on the same GPU (e.g. `INTRA_GPU=3` on a 5090)
- `MULTI_GPU=1`         one teacher per visible GPU, with a per-GPU queue
- `FORCE_RETRAIN=1`     retrain teachers even if their checkpoints already exist (default: skip)
- `RUN_CONTROL=1`       append the CIFAR-10 same-arch control
- `SKIP_DUMP=0`         also dump teacher logits for offline analysis
- `SKIP_ENSEMBLE_OOD=1` skip Step 6 (5-teacher ensemble OOD table)
- `SKIP_SHAPLEY=1`      skip Step 7 (Shapley q_shap analysis; 2^M = 32 coalitions at M=5)

## Layout

| Path                                       | What                                                          |
| ------------------------------------------ | ------------------------------------------------------------- |
| `configs/teachers.yaml`                    | 5-teacher pool (ResNet-50, DenseNet-121, WRN-50-2, ResNeXt-50, ConvNeXt-Tiny) |
| `configs/teacher_recipes/<name>.yaml`      | Per-teacher training overlay (merged when training that teacher only). Used for ConvNeXt-Tiny's AdamW + MixUp/CutMix/RandAug recipe; downstream inference paths ignore it. |
| `configs/students.yaml`                    | ShuffleNetV2 0.5×, RepVGG-A0; KD recipe                       |
| `configs/control_cifar10.yaml`             | Negative control                                              |
| `src/train_teacher.py`                     | Train one teacher (idempotent: skips if `<ckpt>.pt` exists; `--force` or `FORCE_RETRAIN=1` overrides) |
| `src/distill.py`                           | Multi-teacher KD (teachers run live in eval mode)             |
| `src/dump_logits.py`                       | Optional offline logit cache                                  |
| `src/eval_ood.py`                          | Per-model MSP / MaxLogit / Energy → AUROC / AUPR / FPR95      |
| `src/ensemble_ood.py`                      | Step 6: ensemble-logit OOD table with MSP / MaxLogit / Energy / GEN / KNN / Mahalanobis / ViM |
| `src/shapley_q.py`                         | Step 7: exact 2^M Shapley over teachers + q_shap distribution + Fate-A/B verdict |
| `scripts/run_all.sh`                       | One-click entry                                               |

## Outputs

```
checkpoints/<id>/teachers/<name>.pt        # best-acc teacher checkpoints
checkpoints/<id>/students/<name>_kd.pt     # distilled students
outputs/results/<id>/ood_<tag>.json        # per-model OOD metrics
logs/*.log                                 # per-stage stdout
```
