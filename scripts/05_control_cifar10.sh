#!/usr/bin/env bash
# Negative control: CIFAR-10 ID, capability-heterogeneous ResNet teachers,
# same-architecture (ResNet-18) student. Demonstrates the N1 boundary.
#
# Reuses 01/03/04 by overriding the config envs.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export TEACHER_CONFIG="$CONTROL_CONFIG"
export STUDENT_CONFIG="$CONTROL_CONFIG"

log "=== CONTROL: CIFAR-10 + same-arch student ==="
bash scripts/01_train_teachers.sh
bash scripts/03_distill.sh
bash scripts/04_eval_ood.sh
