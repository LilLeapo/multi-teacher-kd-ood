# exp2 — multi-teacher KD, architecture-diverse pool

CIFAR-100 ID with 4 capability-balanced, architecture-diverse CNN teachers distilled
into a heterogeneous student (ShuffleNetV2 0.5× or RepVGG-A0). OOD evaluation against
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
- `INTRA_GPU=N`    run N teacher trainings concurrently on the same GPU (e.g. `INTRA_GPU=3` on a 5090)
- `MULTI_GPU=1`    one teacher per visible GPU, with a per-GPU queue
- `RUN_CONTROL=1`  append the CIFAR-10 same-arch control
- `SKIP_DUMP=0`    also dump teacher logits for offline analysis

## Layout

| Path                              | What                                                          |
| --------------------------------- | ------------------------------------------------------------- |
| `configs/teachers.yaml`           | 4-teacher pool (ResNet-50, DenseNet-121, WRN-50-2, ResNeXt-50)                  |
| `configs/students.yaml`           | ShuffleNetV2 0.5×, RepVGG-A0; KD recipe                       |
| `configs/control_cifar10.yaml`    | Negative control                                              |
| `src/train_teacher.py`            | Train one teacher                                             |
| `src/distill.py`                  | Multi-teacher KD (teachers run live in eval mode)             |
| `src/dump_logits.py`              | Optional offline logit cache                                  |
| `src/eval_ood.py`                 | MSP / MaxLogit / Energy → AUROC / AUPR / FPR95                |
| `scripts/run_all.sh`              | One-click entry                                               |

## Outputs

```
checkpoints/<id>/teachers/<name>.pt        # best-acc teacher checkpoints
checkpoints/<id>/students/<name>_kd.pt     # distilled students
outputs/results/<id>/ood_<tag>.json        # per-model OOD metrics
logs/*.log                                 # per-stage stdout
```
