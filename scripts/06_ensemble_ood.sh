#!/usr/bin/env bash
# Step 2: 4-teacher logit-averaging OOD AUROC table.
# Output: outputs/results/<id>/ensemble_ood.{json,md}
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

log "running ensemble OOD evaluation"
python -m src.ensemble_ood --config "$TEACHER_CONFIG" 2>&1 | tee "logs/ensemble_ood.log"
log "done — see outputs/results/*/ensemble_ood.md"
