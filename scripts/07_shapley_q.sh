#!/usr/bin/env bash
# Step 3: exact 2^M Shapley enumeration + q_shap distribution + verdict.
# Output: outputs/results/<id>/shapley_q.{json,pt}
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

log "running Shapley + q_shap analysis"
python -m src.shapley_q --config "$TEACHER_CONFIG" 2>&1 | tee "logs/shapley_q.log"
log "done — verdict in outputs/results/*/shapley_q.json"
