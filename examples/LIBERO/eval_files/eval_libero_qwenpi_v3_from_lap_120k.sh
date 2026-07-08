#!/usr/bin/env bash
# Evaluate the QwenPI_v3-from-LAP 120k LIBERO checkpoint.
#
# Usage:
#   LIBERO_HOME=/path/to/LIBERO bash examples/LIBERO/eval_files/eval_libero_qwenpi_v3_from_lap_120k.sh
#
# Optional overrides:
#   CKPT=/path/to/checkpoint.pt TASK_SUITES=libero_goal,libero_object GPU_ID=0 PORT=6694 \
#   NUM_TRIALS_PER_TASK=50 bash examples/LIBERO/eval_files/eval_libero_qwenpi_v3_from_lap_120k.sh

set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-${starVLA_python:-python}}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${LIBERO_python:-python}}"
LIBERO_HOME="${LIBERO_HOME:-/home/xhy/LIBERO}"

CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/libero4in1_qwenpi_v3_from_lap/checkpoints/steps_120000_pytorch_model.pt}"
TASK_SUITES="${TASK_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
GPU_ID="${GPU_ID:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
USE_BF16="${USE_BF16:-1}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-600}"
MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-egl}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-egl}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

if [[ ! -d "${LIBERO_HOME}" ]]; then
  echo "[ERROR] LIBERO_HOME not found: ${LIBERO_HOME}" >&2
  echo "Set LIBERO_HOME=/path/to/LIBERO and rerun." >&2
  exit 1
fi

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"
export TOKENIZERS_PARALLELISM=false

MODEL_ROOT="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
FOLDER_NAME="$(echo "${CKPT}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')"
LOG_ROOT="${MODEL_ROOT}/logs/eval_libero"
mkdir -p "${LOG_ROOT}"

SERVER_LOG="${LOG_ROOT}/${FOLDER_NAME}_server_port${PORT}.log"
SERVER_CMD=(
  "${STARVLA_PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT}"
  --port "${PORT}"
  --idle_timeout -1
)

if [[ "${USE_BF16}" == "1" ]]; then
  SERVER_CMD+=(--use_bf16)
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[INFO] Stopping policy server PID=${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[INFO] Starting policy server on GPU ${GPU_ID}, port ${PORT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${SERVER_CMD[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "[INFO] Waiting for policy server to accept connections..."
"${STARVLA_PYTHON}" - "${HOST}" "${PORT}" "${WAIT_TIMEOUT}" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout = float(sys.argv[3])
deadline = time.time() + timeout

while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            sys.exit(0)
    except OSError:
        time.sleep(2)

print(f"Timed out waiting for {host}:{port}", file=sys.stderr)
sys.exit(1)
PY

IFS=',' read -ra TASK_SUITE_ARR <<< "${TASK_SUITES}"
for task_suite_name in "${TASK_SUITE_ARR[@]}"; do
  task_suite_name="${task_suite_name// /}"
  if [[ -z "${task_suite_name}" ]]; then
    continue
  fi

  VIDEO_OUT_PATH="${MODEL_ROOT}/results/${task_suite_name}/${FOLDER_NAME}"
  EVAL_LOG="${LOG_ROOT}/${FOLDER_NAME}_${task_suite_name}.log"
  mkdir -p "${VIDEO_OUT_PATH}"

  echo "[INFO] Evaluating ${task_suite_name}; videos -> ${VIDEO_OUT_PATH}"
  "${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${CKPT}" \
    --args.host "${HOST}" \
    --args.port "${PORT}" \
    --args.task-suite-name "${task_suite_name}" \
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
    --args.max-tasks "${MAX_TASKS}" \
    --args.video-out-path "${VIDEO_OUT_PATH}" \
    2>&1 | tee "${EVAL_LOG}"
done

echo "[INFO] LIBERO evaluation finished. Logs saved in ${LOG_ROOT}"
