#!/usr/bin/env bash
# QwenLAP (Language-Action Pre-training) training on LIBERO.
# Stage 1: VLM-only — Qwen3-VL learns to speak actions as language (CE loss).
# Stage 2 (future): attach a flow-matching action expert on top of the VLM.
#
# Usage:
#   bash examples/LIBERO/train_files/run_libero_train_lap.sh            # defaults (4 GPU, 30k steps)
#   bash examples/LIBERO/train_files/run_libero_train_lap.sh 8 100000   # 8 GPU, 100k steps

set -euo pipefail

# ── NCCL / distributed stability ─────────────────────────────────────────────
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export WANDB_MODE=${WANDB_MODE:-online}

###############################################################################
# === Configuration ============================================================
framework_name=QwenLAP
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_lap_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=libero_lap_v1

# ── QwenLAP-specific knobs ───────────────────────────────────────────────────
# action_horizon: chunk length fed to textualize_action().
# NOTE: must match data_config.py action_indices=list(range(N)).
# Default keeps in sync with data_config range(8); set both to 32 together.
action_horizon=8
max_new_tokens=64   # inference-only; does not affect training loss

# ── Training knobs ───────────────────────────────────────────────────────────
# LAP trains the full VLM (freeze_modules='').
# Conservative lr to preserve VLM pre-training representations.
learning_rate_base=2.0e-05
num_processes=${1:-4}
max_train_steps=${2:-80000}   # 30k for initial validation; 100k+ for full run
save_interval=10000
per_device_batch_size=8
# === End of configuration ====================================================
###############################################################################

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"   # archive the exact script that produced this run

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.action_model.action_horizon "${action_horizon}" \
  --framework.action_model.max_new_tokens "${max_new_tokens}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.learning_rate.base "${learning_rate_base}" \
  --trainer.loss_scale.vla 1.0 \
  --trainer.loss_scale.vlm 0.1 \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.is_resume true \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Libero_QwenLAP \
  --wandb_entity haoyuxiong \
  2>&1 | tee "${output_dir}/train.log"
