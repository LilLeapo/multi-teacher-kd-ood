#!/usr/bin/env bash
# Distill the full method × seed × student matrix for the main M=5 study.
#
# Methods (left-to-right is also the execution order — SLG must come before
# CSR-KD / SRR for the same (student, seed), since SRR's anchor reads the
# SLG checkpoint of the matching seed):
#
#     uniform → accuracy → learned_global → slg → csr_kd → srr
#
# distill.py is idempotent: if `checkpoints/<id>/students/<student>__<method>__seed{N}.pt`
# already exists, the run is skipped. Pass FORCE=1 to retrain everything.
#
# Toggles:
#   STUDENTS="shufflenetv2_x0_5 repvgg_a0"   (default: shufflenetv2_x0_5 + repvgg_a0)
#   SEEDS="42 123 3407"                       (default: 42 123 3407)
#   METHODS="uniform accuracy learned_global slg csr_kd srr"
#   FORCE=1                                   force retrain even if ckpt exists
#   REPVGG_SEEDS="42"                         RepVGG-A0 typically only runs a single seed
#
# SLG precompute runs once upfront — it depends on teachers but not on student or seed.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

: "${STUDENTS:=shufflenetv2_x0_5 repvgg_a0}"
: "${SEEDS:=42 123 3407}"
: "${REPVGG_SEEDS:=42}"
: "${METHODS:=uniform accuracy learned_global slg csr_kd srr}"

FORCE_FLAG=""
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_FLAG="--force"
fi

log "=== precompute SLG gate (once) ==="
python -m src.precompute_slg --config "${TEACHER_CONFIG}" 2>&1 | tee -a logs/precompute_slg.log

for student in $STUDENTS; do
  # Repvgg defaults to a single seed (robustness check, not main multi-seed table).
  case "$student" in
    repvgg_a0) student_seeds="$REPVGG_SEEDS" ;;
    *)         student_seeds="$SEEDS" ;;
  esac

  for seed in $student_seeds; do
    for method in $METHODS; do
      tag="${student}__${method}__seed${seed}"
      ckpt="checkpoints/cifar100/students/${tag}.pt"
      if [[ -f "$ckpt" && "${FORCE:-0}" != "1" ]]; then
        log "skip ${tag} (checkpoint exists)"
        continue
      fi
      log "distill ${tag}"
      python -m src.distill \
        --teacher-config "${TEACHER_CONFIG}" \
        --student-config "${STUDENT_CONFIG}" \
        --student "$student" \
        --method "$method" \
        --seed "$seed" \
        $FORCE_FLAG 2>&1 | tee "logs/distill_${tag}.log"
    done
  done
done

log "matrix complete. checkpoints under checkpoints/cifar100/students/"
