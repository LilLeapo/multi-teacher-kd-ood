#!/usr/bin/env bash
# New main line orchestrator: EXP 1 (teacher Jensen gap) → EXP 2 (homogeneous
# ResNet-50 control ensemble) → EXP 3 (simplified student matrix + hybrid OSR).
#
# Every stage is idempotent:
#   - jensen_gap dump skips splits whose cache already exists
#   - train_teacher / distill skip runs whose checkpoint already exists
#   - jensen_gap analyze and hybrid_osr are pure recomputes (cheap, always rerun)
#
# So this script can be interrupted and re-launched at will. Set FORCE=1 to
# rebuild caches; pass STAGES="1 3" to run a subset of stages.
#
# Toggles:
#   STAGES="1 2 3"          which stages to run (default: all)
#   HETERO_TEACHERS=...     space-separated names from teachers.yaml
#   HOMO_TEACHERS=...       space-separated names from teachers_homogeneous.yaml
#   STUDENT_METHODS=...     methods for EXP 3 (default: uniform accuracy learned_global)
#   STUDENT_SEEDS=...       seeds for EXP 3 (default: 42 123 3407)
#   FORCE=1                 rebuild all caches

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

: "${STAGES:=1 2 3}"
: "${HOMO_TEACHER_CONFIG:=configs/teachers_homogeneous.yaml}"
: "${HETERO_TEACHERS:=resnet50 densenet121 wide_resnet50_2 resnext50_32x4d convnext_tiny}"
: "${HOMO_TEACHERS:=resnet50_seed42 resnet50_seed123 resnet50_seed3407 resnet50_seed7 resnet50_seed2024}"
: "${STUDENT_METHODS:=uniform accuracy learned_global}"
: "${STUDENT_SEEDS:=42 123 3407}"

FORCE_FLAG=""
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_FLAG="--force"
fi

want_stage() {
  for s in $STAGES; do [[ "$s" == "$1" ]] && return 0; done
  return 1
}

# ---------------------------------------------------------------------------
# EXP 1 — heterogeneous teacher Jensen gap (T1 / F1 / E_OOD/E_ID ratios)
# ---------------------------------------------------------------------------
if want_stage 1; then
  log "=== EXP 1.1 — dump heterogeneous teacher logits over {id_test, OOD...} ==="
  python -m src.jensen_gap dump \
      --config "${TEACHER_CONFIG}" --kind teacher \
      --models ${HETERO_TEACHERS} \
      ${FORCE_FLAG} 2>&1 | tee logs/jensen_dump_heterogeneous.log

  log "=== EXP 1.2 — analyze heterogeneous ensemble (T1 + Δ + F1) ==="
  python -m src.jensen_gap analyze \
      --config "${TEACHER_CONFIG}" --kind teacher \
      --models ${HETERO_TEACHERS} \
      --tag heterogeneous 2>&1 | tee logs/jensen_analyze_heterogeneous.log
fi

# ---------------------------------------------------------------------------
# EXP 2 — train 5 × ResNet-50 homogeneous control, then dump + analyze
# ---------------------------------------------------------------------------
if want_stage 2; then
  log "=== EXP 2.1 — train homogeneous ResNet-50 ensemble (5 seeds) ==="
  for name in ${HOMO_TEACHERS}; do
    log "train ${name}"
    python -m src.train_teacher \
        --config "${HOMO_TEACHER_CONFIG}" --teacher "${name}" \
        2>&1 | tee "logs/teacher_${name}.log"
  done

  log "=== EXP 2.2 — dump homogeneous teacher logits ==="
  python -m src.jensen_gap dump \
      --config "${HOMO_TEACHER_CONFIG}" --kind teacher \
      --models ${HOMO_TEACHERS} \
      ${FORCE_FLAG} 2>&1 | tee logs/jensen_dump_homogeneous.log

  log "=== EXP 2.3 — analyze homogeneous ensemble (T2 row) ==="
  python -m src.jensen_gap analyze \
      --config "${HOMO_TEACHER_CONFIG}" --kind teacher \
      --models ${HOMO_TEACHERS} \
      --tag homogeneous 2>&1 | tee logs/jensen_analyze_homogeneous.log
fi

# ---------------------------------------------------------------------------
# EXP 3 — simplified student matrix + hybrid OSR table
# ---------------------------------------------------------------------------
if want_stage 3; then
  log "=== EXP 3.1 — train missing students (skips existing checkpoints) ==="
  for method in ${STUDENT_METHODS}; do
    for seed in ${STUDENT_SEEDS}; do
      tag="shufflenetv2_x0_5__${method}__seed${seed}"
      ckpt="checkpoints/cifar100/students/${tag}.pt"
      if [[ -f "$ckpt" ]]; then
        log "skip ${tag} (checkpoint exists)"
        continue
      fi
      log "distill ${tag}"
      python -m src.distill \
          --teacher-config "${TEACHER_CONFIG}" \
          --student-config "${STUDENT_CONFIG}" \
          --student shufflenetv2_x0_5 --method "${method}" --seed "${seed}" \
          2>&1 | tee "logs/distill_${tag}.log"
    done
  done

  log "=== EXP 3.2 — hybrid OSR evaluation (uses EXP 1 teacher caches) ==="
  python -m src.hybrid_osr \
      --teacher-config "${TEACHER_CONFIG}" \
      --student-config "${STUDENT_CONFIG}" \
      --teachers ${HETERO_TEACHERS} \
      --student shufflenetv2_x0_5 \
      --methods ${STUDENT_METHODS} \
      --seeds ${STUDENT_SEEDS} \
      --tag exp3_simplified 2>&1 | tee logs/hybrid_osr.log
fi

log "=== new main line complete (stages: ${STAGES}) ==="
log "Key artifacts under outputs/results/cifar100/:"
log "  jensen_gap__heterogeneous.{json,md,__hist.png}"
log "  jensen_gap__homogeneous.{json,md,__hist.png}"
log "  hybrid_osr__exp3_simplified.{json,md}"
