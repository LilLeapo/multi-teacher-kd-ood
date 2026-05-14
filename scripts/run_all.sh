#!/usr/bin/env bash
# One-click pipeline. Assumes a CUDA GPU box with `pip install -r requirements.txt`
# already run (or set INSTALL=1 to install first).
#
# Toggles:
#   INSTALL=1            pip install requirements before running
#   SKIP_DUMP=1          skip the optional logit dump (default: skip)
#   RUN_CONTROL=1        also run the CIFAR-10 same-arch control after the main pipeline
#   SKIP_ENSEMBLE_OOD=1  skip Step 6 (4-teacher ensemble OOD table)
#   SKIP_SHAPLEY=1       skip Step 7 (exact 2^M Shapley + q_shap verdict)
#
# Teacher-training concurrency (forwarded to scripts/01_train_teachers.sh):
#   INTRA_GPU=N          N teachers concurrently on the same GPU (e.g. INTRA_GPU=3 on a 5090)
#   MULTI_GPU=1          one teacher per visible GPU, per-GPU queue

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ "${INSTALL:-0}" == "1" ]]; then
  log "installing requirements"
  python -m pip install -r requirements.txt
fi

log "=== STEP 0: prepare data ==="
bash scripts/00_prepare_data.sh

log "=== STEP 1: train teachers ==="
bash scripts/01_train_teachers.sh

if [[ "${SKIP_DUMP:-1}" != "1" ]]; then
  log "=== STEP 2: dump teacher logits (optional) ==="
  bash scripts/02_dump_logits.sh
fi

log "=== STEP 3: distill students ==="
bash scripts/03_distill.sh

log "=== STEP 4: OOD eval (per-model) ==="
bash scripts/04_eval_ood.sh

if [[ "${RUN_CONTROL:-0}" == "1" ]]; then
  log "=== STEP 5: CIFAR-10 control ==="
  bash scripts/05_control_cifar10.sh
fi

if [[ "${SKIP_ENSEMBLE_OOD:-0}" != "1" ]]; then
  log "=== STEP 6: ensemble OOD (4-teacher logit averaging, 7 scores) ==="
  bash scripts/06_ensemble_ood.sh
fi

if [[ "${SKIP_SHAPLEY:-0}" != "1" ]]; then
  log "=== STEP 7: Shapley + q_shap verdict ==="
  bash scripts/07_shapley_q.sh
fi

log "PIPELINE COMPLETE."
log "  teacher / student checkpoints -> checkpoints/"
log "  per-run training logs         -> logs/"
log "  OOD eval JSONs                -> outputs/results/"
log "  ensemble OOD table            -> outputs/results/<id>/ensemble_ood.md"
log "  Shapley q_shap verdict        -> outputs/results/<id>/shapley_q.json"
