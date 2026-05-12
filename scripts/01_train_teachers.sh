#!/usr/bin/env bash
# Train every teacher listed in $TEACHER_CONFIG. Sequential by default;
# set PARALLEL=1 to launch one process per visible GPU.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

names=$(python - <<PY
import yaml, sys
from src.utils.config import load_config
cfg = load_config("${TEACHER_CONFIG}")
print(" ".join(t["name"] for t in cfg["teachers"]))
PY
)
log "teachers to train: $names"

train_one() {
  local name=$1
  local logf="logs/teacher_${name}.log"
  log "training $name -> $logf"
  python -m src.train_teacher --config "$TEACHER_CONFIG" --teacher "$name" 2>&1 | tee "$logf"
}

if [[ "${PARALLEL:-0}" == "1" ]]; then
  ngpu=$(python -c 'import torch; print(torch.cuda.device_count())')
  log "parallel mode across $ngpu GPUs"
  i=0
  for name in $names; do
    gpu=$((i % ngpu))
    CUDA_VISIBLE_DEVICES=$gpu train_one "$name" &
    i=$((i + 1))
  done
  wait
else
  for name in $names; do
    train_one "$name"
  done
fi

log "all teachers trained"
