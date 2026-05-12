#!/usr/bin/env bash
# Distill every student in $STUDENT_CONFIG against the full 5-teacher pool.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

names=$(python - <<PY
from src.utils.config import load_config
cfg = load_config("${STUDENT_CONFIG}")
print(" ".join(s["name"] for s in cfg["students"]))
PY
)

for name in $names; do
  log "distilling student $name"
  python -m src.distill \
    --teacher-config "$TEACHER_CONFIG" \
    --student-config "$STUDENT_CONFIG" \
    --student "$name" 2>&1 | tee "logs/distill_${name}.log"
done
