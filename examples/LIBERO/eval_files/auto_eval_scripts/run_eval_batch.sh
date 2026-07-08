#!/usr/bin/env bash
# Batch LIBERO evaluation via eval_libero_parall.sh — all config from CLI, no script edits.
#
# Examples:
#   # Multiple model dirs, all checkpoints, 4 suites, 4 GPUs
#   bash run_eval_batch.sh \
#     --models playground/Checkpoints/libero4in1_cosmospi,playground/Checkpoints/libero4in1_qwenpi \
#     --tasks libero_goal,libero_spatial,libero_object,libero_10 \
#     --gpus 0,1,2,3 \
#     --base-port 6450
#
#   # Explicit checkpoint files only
#   bash run_eval_batch.sh \
#     --ckpts playground/Checkpoints/model_a/checkpoints/steps_50000_pytorch_model.pt \
#     --tasks libero_goal \
#     --gpus 0 \
#     --base-port 6450
#
# Environment (optional overrides):
#   LIBERO_HOME, LIBERO_python, starVLA_python, STARVLA_DIR

set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
EVAL_SCRIPT="${STARVLA_DIR}/examples/LIBERO/eval_files/auto_eval_scripts/eval_libero_parall.sh"

LIBERO_HOME="${LIBERO_HOME:-/home/xhy/LIBERO}"
LIBERO_python="${LIBERO_python:-/home/xhy/miniconda3/envs/libero/bin/python}"
starVLA_python="${starVLA_python:-/home/xhy/miniconda3/envs/starVLA/bin/python}"

MODELS=""
CKPTS=""
TASKS="libero_goal"
GPUS="0"
BASE_PORT=6450
SLEEP_BETWEEN=10
CKPT_GLOB="steps_*_pytorch_model.pt"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: run_eval_batch.sh [OPTIONS]

Evaluate multiple checkpoints × task suites across GPUs without editing repo scripts.

Options:
  --models PATH[,PATH...]   Model run dirs; scans each PATH/checkpoints/*.pt
  --ckpts  PATH[,PATH...]   Explicit .pt checkpoint files (overrides --models)
  --tasks  NAME[,NAME...]   Task suites (default: libero_goal)
                            libero_goal | libero_spatial | libero_object | libero_10
  --gpus   ID[,ID...]       GPU ids, round-robin (default: 0)
  --base-port N             Starting websocket port; each job gets base-port + job_index (default: 6450)
  --ckpt-glob GLOB          Glob under checkpoints/ when using --models (default: steps_*_pytorch_model.pt)
  --sleep-between SEC       Pause between launching jobs (default: 10)
  --dry-run                 Print planned jobs without running
  -h, --help                Show this help

Provide either --models or --ckpts (at least one checkpoint source required).

Example:
  LIBERO_HOME=/home/xhy/LIBERO \
  bash examples/LIBERO/eval_files/auto_eval_scripts/run_eval_batch.sh \
    --models playground/Checkpoints/libero4in1_cosmospi \
    --tasks libero_goal,libero_spatial \
    --gpus 0,1 \
    --base-port 6450
EOF
}

split_csv() {
  local csv="$1"
  local -n _out=$2
  _out=()
  local IFS=','
  read -ra _out <<< "${csv// /}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --ckpts) CKPTS="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --base-port) BASE_PORT="$2"; shift 2 ;;
    --ckpt-glob) CKPT_GLOB="$2"; shift 2 ;;
    --sleep-between) SLEEP_BETWEEN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -x "${LIBERO_python}" && ! -f "${LIBERO_python}" ]]; then
  echo "[WARN] LIBERO_python not found at ${LIBERO_python}; using 'python' from PATH."
  LIBERO_python="${LIBERO_python:-python}"
fi
if [[ ! -x "${starVLA_python}" && ! -f "${starVLA_python}" ]]; then
  echo "[WARN] starVLA_python not found at ${starVLA_python}; using 'python' from PATH."
  starVLA_python="${starVLA_python:-python}"
fi

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "[ERROR] LIBERO_HOME is required." >&2
  exit 1
fi

split_csv "${TASKS}" TASK_ARR
split_csv "${GPUS}" GPU_ARR

if [[ ${#TASK_ARR[@]} -eq 0 ]]; then
  echo "[ERROR] --tasks must list at least one suite." >&2
  exit 1
fi
if [[ ${#GPU_ARR[@]} -eq 0 ]]; then
  echo "[ERROR] --gpus must list at least one GPU." >&2
  exit 1
fi

CKPT_ARR=()
if [[ -n "${CKPTS}" ]]; then
  split_csv "${CKPTS}" CKPT_ARR
elif [[ -n "${MODELS}" ]]; then
  split_csv "${MODELS}" MODEL_ARR
  for model_dir in "${MODEL_ARR[@]}"; do
    ckpt_dir="${model_dir%/}/checkpoints"
    if [[ ! -d "${ckpt_dir}" ]]; then
      echo "[ERROR] checkpoints dir not found: ${ckpt_dir}" >&2
      exit 1
    fi
    shopt -s nullglob
    found=( "${ckpt_dir}"/${CKPT_GLOB} )
    shopt -u nullglob
    if [[ ${#found[@]} -eq 0 ]]; then
      echo "[ERROR] No checkpoints matching '${CKPT_GLOB}' in ${ckpt_dir}" >&2
      exit 1
    fi
    # Sort by step number in filename
    mapfile -t sorted < <(printf '%s\n' "${found[@]}" | sort -t_ -k2 -n)
    CKPT_ARR+=( "${sorted[@]}" )
  done
else
  echo "[ERROR] Provide --models or --ckpts." >&2
  usage
  exit 1
fi

for ckpt in "${CKPT_ARR[@]}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "[ERROR] Checkpoint not found: ${ckpt}" >&2
    exit 1
  fi
done

num_gpus=${#GPU_ARR[@]}
job_index=0
pids=()
gpu_job_count=()

for ((i=0; i<num_gpus; i++)); do
  gpu_job_count[$i]=0
done

echo "=========================================="
echo " run_eval_batch.sh"
echo "=========================================="
echo " Checkpoints : ${#CKPT_ARR[@]} file(s)"
printf '   %s\n' "${CKPT_ARR[@]}"
echo " Task suites : ${TASK_ARR[*]}"
echo " GPU list    : ${GPU_ARR[*]}"
echo " Base port   : ${BASE_PORT}"
echo " Parallelism : up to ${num_gpus} jobs at a time"
echo "=========================================="

for ckpt in "${CKPT_ARR[@]}"; do
  for task in "${TASK_ARR[@]}"; do
    gpu_idx=$((job_index % num_gpus))
    gpu_id=${GPU_ARR[$gpu_idx]}
    port=$((BASE_PORT + job_index))
    ckpt_name=$(basename "${ckpt}" .pt)

    echo "[Job ${job_index}] GPU=${gpu_id}  port=${port}  ckpt=${ckpt_name}  task=${task}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
      job_index=$((job_index + 1))
      gpu_job_count[$gpu_idx]=$(( gpu_job_count[$gpu_idx] + 1 ))
      continue
    fi

    # Launch a batch of num_gpus jobs in parallel; wait before next batch.
    if (( job_index > 0 && job_index % num_gpus == 0 )); then
      echo "--- Waiting for batch (jobs $((job_index - num_gpus))..$((job_index - 1))) ---"
      for pid in "${pids[@]}"; do
        wait "${pid}" || true
      done
      pids=()
    fi

    (
      export LIBERO_HOME LIBERO_python starVLA_python STARVLA_DIR
      bash "${EVAL_SCRIPT}" "${ckpt}" "${task}" "${gpu_id}" "${port}"
    ) &
    pids+=($!)

    gpu_job_count[$gpu_idx]=$(( gpu_job_count[$gpu_idx] + 1 ))
    job_index=$((job_index + 1))
    sleep "${SLEEP_BETWEEN}"
  done
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "--- Dry run complete: ${job_index} job(s) planned ---"
  exit 0
fi

if [[ ${#pids[@]} -gt 0 ]]; then
  echo "--- Waiting for final batch ---"
  for pid in "${pids[@]}"; do
    wait "${pid}" || true
  done
fi

echo "=========================================="
echo " All evaluations completed (${job_index} jobs)!"
echo " GPU job distribution:"
for ((i=0; i<num_gpus; i++)); do
  echo "   GPU ${GPU_ARR[$i]}: ${gpu_job_count[$i]} jobs"
done
echo "=========================================="
