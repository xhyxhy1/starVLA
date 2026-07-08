#!/bin/bash
# 4-GPU full training + tee log (做法 1: 先建目录再 tee)
# Run from repo root: bash examples/LIBERO/train_files/run_4gpu_with_log.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

RUN_ROOT=./results/Checkpoints
RUN_ID=libero_janusflow_4gpu_full
OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}"
LOG="${OUTPUT_DIR}/train_4gpu.log"

mkdir -p "${OUTPUT_DIR}"

echo "Log file: ${LOG}"
echo "========================================="

conda run -n starVLA bash examples/LIBERO/train_files/run_libero_train_janusflow_4gpu.sh 2>&1 | tee "${LOG}"
