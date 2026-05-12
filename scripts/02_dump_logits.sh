#!/usr/bin/env bash
# Optional: precompute teacher logits on the (un-augmented) ID train set.
# distill.py does NOT need these — it runs teachers live so KD targets stay
# consistent with the student's augmented input. Use this only for offline
# probing / logit analysis.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

names=$(python - <<PY
from src.utils.config import load_config
cfg = load_config("${TEACHER_CONFIG}")
print(" ".join(t["name"] for t in cfg["teachers"]))
PY
)

for name in $names; do
  log "dumping logits for $name"
  python -m src.dump_logits --config "$TEACHER_CONFIG" --teacher "$name" 2>&1 | tee "logs/dump_${name}.log"
done
