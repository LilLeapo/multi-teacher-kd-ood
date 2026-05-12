#!/usr/bin/env bash
# Common helpers; sourced by every script.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Default config to the main CIFAR-100 setup; override via env.
: "${TEACHER_CONFIG:=configs/teachers.yaml}"
: "${STUDENT_CONFIG:=configs/students.yaml}"
: "${CONTROL_CONFIG:=configs/control_cifar10.yaml}"

mkdir -p logs checkpoints outputs data

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
