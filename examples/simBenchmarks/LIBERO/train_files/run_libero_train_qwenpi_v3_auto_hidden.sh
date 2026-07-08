#!/usr/bin/env bash
# QwenPI_v3 training on LIBERO with Action DiT hidden inferred from Qwen.
# Usage: bash examples/LIBERO/train_files/run_libero_train_qwenpi_v3_auto_hidden.sh

set -euo pipefail

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export WANDB_MODE=${WANDB_MODE:-online}

###############################################################################
# === Paths and run identity ===
Framework_name=QwenPI_v3
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=libero4in1_qwenpi_v3_auto_hidden

# QwenPI_v3 will infer Action DiT hidden from the loaded Qwen model when
# action_dit_hidden_dim is not provided. For Qwen3-VL-4B this is expected to
# be 2560, so this run is intentionally heavier than the 1024-hidden script.

# QwenPI_v3 currently reads repeated diffusion steps from trainer.*, not
# framework.action_model.*. Keep this explicit to avoid the code default of 16.
repeated_diffusion_steps=2

# This variant is heavier because Action DiT hidden is not compressed.
per_device_batch_size=8
num_processes=4
# === End of configuration ===
###############################################################################

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules 'qwen_vl_interface' \
  --trainer.repeated_diffusion_steps "${repeated_diffusion_steps}" \
  --trainer.learning_rate.base 2.5e-05 \
  --trainer.learning_rate.action_model 1.0e-04 \
  --trainer.loss_scale.vla 1.0 \
  --trainer.max_train_steps 160000 \
  --trainer.is_resume true \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Libero_QwenPI_v3 \
  --wandb_entity haoyuxiong \
  2>&1 | tee "${output_dir}/train.log"
