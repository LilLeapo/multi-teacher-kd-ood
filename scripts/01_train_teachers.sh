#!/usr/bin/env bash
# Train every teacher listed in $TEACHER_CONFIG.
#
# Concurrency modes (pick one — default is sequential):
#   INTRA_GPU=N   run N teachers concurrently on the same GPU (FIFO queue;
#                 whichever finishes first frees a slot for the next teacher)
#   MULTI_GPU=1   one teacher per visible GPU, with a per-GPU queue
#
# `wait -n` requires bash 4.3+ (present on every modern Linux GPU host).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

names=$(python - <<PY
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

# --- Mode: intra-GPU concurrency (single GPU, N concurrent processes) ---
if [[ -n "${INTRA_GPU:-}" ]]; then
  n=$INTRA_GPU
  log "intra-GPU mode: $n concurrent processes on GPU 0"
  running=0
  for name in $names; do
    while (( running >= n )); do
      wait -n
      running=$((running - 1))
    done
    train_one "$name" &
    running=$((running + 1))
  done
  wait
  log "all teachers trained (intra-GPU $n-way)"
  exit 0
fi

# --- Mode: multi-GPU, one teacher per GPU at a time (per-GPU queue) ---
if [[ "${MULTI_GPU:-0}" == "1" ]]; then
  ngpu=$(python -c 'import torch; print(torch.cuda.device_count())')
  log "multi-GPU mode: $ngpu GPUs, one teacher per GPU at a time"
  declare -a queue
  for ((g=0; g<ngpu; g++)); do queue[$g]=""; done
  i=0
  for name in $names; do
    g=$((i % ngpu))
    queue[$g]="${queue[$g]} $name"
    i=$((i + 1))
  done
  for ((g=0; g<ngpu; g++)); do
    (
      for nm in ${queue[$g]}; do
        CUDA_VISIBLE_DEVICES=$g python -m src.train_teacher \
          --config "$TEACHER_CONFIG" --teacher "$nm" 2>&1 | tee "logs/teacher_${nm}.log"
      done
    ) &
  done
  wait
  log "all teachers trained (multi-GPU)"
  exit 0
fi

# --- Default: sequential ---
log "sequential mode (one teacher at a time)"
for name in $names; do
  train_one "$name"
done
log "all teachers trained (sequential)"
