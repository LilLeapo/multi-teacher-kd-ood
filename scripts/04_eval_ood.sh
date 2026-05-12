#!/usr/bin/env bash
# OOD eval for every trained model — teachers and distilled students.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

id_ds=$(python - <<PY
from src.utils.config import load_config
print(load_config("${STUDENT_CONFIG}")["id_dataset"])
PY
)

# Teachers
mapfile -t teacher_specs < <(python - <<PY
from src.utils.config import load_config
for t in load_config("${TEACHER_CONFIG}")["teachers"]:
    print(f"{t['name']}\t{t['arch']}")
PY
)
for spec in "${teacher_specs[@]}"; do
  name=$(echo "$spec" | cut -f1)
  arch=$(echo "$spec" | cut -f2)
  ckpt="checkpoints/${id_ds}/teachers/${name}.pt"
  [[ -f "$ckpt" ]] || { log "skip $name (no ckpt)"; continue; }
  log "eval teacher $name"
  python -m src.eval_ood --config "$TEACHER_CONFIG" --ckpt "$ckpt" --arch "$arch" --tag "teacher_${name}" 2>&1 | tee "logs/eval_teacher_${name}.log"
done

# Students
mapfile -t student_specs < <(python - <<PY
from src.utils.config import load_config
for s in load_config("${STUDENT_CONFIG}")["students"]:
    print(f"{s['name']}\t{s['arch']}")
PY
)
for spec in "${student_specs[@]}"; do
  name=$(echo "$spec" | cut -f1)
  arch=$(echo "$spec" | cut -f2)
  ckpt="checkpoints/${id_ds}/students/${name}_kd.pt"
  [[ -f "$ckpt" ]] || { log "skip $name (no ckpt)"; continue; }
  log "eval student $name"
  python -m src.eval_ood --config "$STUDENT_CONFIG" --ckpt "$ckpt" --arch "$arch" --tag "student_${name}_kd" 2>&1 | tee "logs/eval_student_${name}.log"
done

log "ood eval complete; results in outputs/results/${id_ds}/"
