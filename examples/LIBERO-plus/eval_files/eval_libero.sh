#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-/home/xhy/Project/LIBERO-plus}"
LIBERO_Python="${LIBERO_Python:-${LIBERO_PYTHON:-/home/xhy/miniconda3/envs/libero/bin/python}}"
ABot_python="${ABot_python:-${ABOT_PYTHON:-/home/xhy/miniconda3/envs/starVLA/bin/python}}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
host="${host:-127.0.0.1}"
your_ckpt="${your_ckpt:-/home/xhy/starVLA/playground/Checkpoints/libero4in1_qwenpi_v3/checkpoints/steps_160000_pytorch_model.pt}"
output_dir="${output_dir:-${STARVLA_DIR}/results/libero_plus_4gpu_eval}"
base_port="${base_port:-9883}"
server_warmup_seconds="${server_warmup_seconds:-60}"
start_servers="${start_servers:-1}"
num_trials_per_task="${num_trials_per_task:-1}"
category_fraction="${category_fraction:-1.0}"
log_every_tasks="${log_every_tasks:-100}"

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required."
  exit 1
fi

cd "${STARVLA_DIR}"
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export MUJOCO_GL
export PYOPENGL_PLATFORM
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"

run_id="$(date +"%Y%m%d_%H%M%S")"
run_dir="${output_dir}/${run_id}"
log_dir="${run_dir}/logs"
video_dir="${run_dir}/videos"
mkdir -p "${log_dir}" "${video_dir}"

suites=("libero_10" "libero_goal" "libero_object" "libero_spatial")
gpus=(0 1 2 3)
ports=($((base_port + 0)) $((base_port + 1)) $((base_port + 2)) $((base_port + 3)))
server_pids=()
eval_pids=()

cleanup() {
  if [[ "${start_servers}" == "1" && ${#server_pids[@]} -gt 0 ]]; then
    echo "Stopping policy servers: ${server_pids[*]}"
    kill "${server_pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Run directory: ${run_dir}"
echo "LIBERO_HOME: ${LIBERO_HOME}"
echo "MUJOCO_GL=${MUJOCO_GL}, PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}"
echo "Checkpoint: ${your_ckpt}"
echo "Category fraction: ${category_fraction}"
echo "Log every tasks: ${log_every_tasks}"
echo "LIBERO_Python: ${LIBERO_Python}"
echo "ABot_python: ${ABot_python}"

"${LIBERO_Python}" -c "import robosuite; import mujoco; from libero.libero import benchmark; print('LIBERO-plus eval deps ok')"
"${ABot_python}" -c "import pandas; import torch; print('StarVLA server deps ok')"

if [[ "${start_servers}" == "1" ]]; then
  for i in "${!suites[@]}"; do
    gpu="${gpus[$i]}"
    port="${ports[$i]}"
    server_log="${log_dir}/server_gpu${gpu}_port${port}.log"
    echo "Starting policy server on GPU ${gpu}, port ${port} -> ${server_log}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
      your_ckpt="${your_ckpt}" \
      base_port="${port}" \
      gpu_id="${gpu}" \
      ABot_python="${ABot_python}" \
      PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}" \
      bash "${STARVLA_DIR}/examples/LIBERO-plus/eval_files/run_policy_server.sh" \
      > "${server_log}" 2>&1 &
    server_pids+=("$!")
  done
  echo "Waiting ${server_warmup_seconds}s for policy servers to warm up..."
  sleep "${server_warmup_seconds}"
fi

start_time="$(date +%s)"
for i in "${!suites[@]}"; do
  suite="${suites[$i]}"
  gpu="${gpus[$i]}"
  port="${ports[$i]}"
  suite_video_dir="${video_dir}/${suite}"
  eval_log="${log_dir}/eval_${suite}_gpu${gpu}_port${port}.log"
  mkdir -p "${suite_video_dir}"
  echo "Starting ${suite} eval on GPU ${gpu}, port ${port} -> ${eval_log}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    "${LIBERO_Python}" "${STARVLA_DIR}/examples/LIBERO-plus/eval_files/eval_libero.py" \
    --args.host "${host}" \
    --args.port "${port}" \
    --args.task-suite-name "${suite}" \
    --args.num-trials-per-task "${num_trials_per_task}" \
    --args.category-fraction "${category_fraction}" \
    --args.log-every-tasks "${log_every_tasks}" \
    --args.start-idx 0 \
    --args.end-idx -1 \
    --args.video-out-path "${suite_video_dir}" \
    --args.log-path "${log_dir}" \
    > "${eval_log}" 2>&1 &
  eval_pids+=("$!")
done

echo "Waiting for evaluation jobs: ${eval_pids[*]}"
for pid in "${eval_pids[@]}"; do
  wait "${pid}"
done

end_time="$(date +%s)"
elapsed_seconds=$((end_time - start_time))
echo "elapsed_seconds=${elapsed_seconds}" | tee "${log_dir}/elapsed.txt"

LOG_DIR="${log_dir}" "${LIBERO_Python}" "${STARVLA_DIR}/examples/LIBERO-plus/eval_files/aggregate_results.py"
echo "Aggregated results: ${log_dir}/overall_results.json"
