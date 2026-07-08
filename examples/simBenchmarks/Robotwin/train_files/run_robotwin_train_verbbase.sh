#!/usr/bin/env bash
# QwenPI_v3 training on RoboTwin-Clean-verb-base (21 tasks, 50 demos each, clean only).
# Usage: bash examples/Robotwin/train_files/run_robotwin_train_verbbase.sh

set -euo pipefail

# cluster-only NCCL settings, disabled on single machine
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

# export WANDB_MODE=disabled
# Reduce fragmentation OOM on optimizer.step (optional but helpful on 80GB single-GPU).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenPI_v3
# Freeze VLM: only train action head (~0.5B). Full finetune needs 4+ GPUs or ZeRO-3 offload.
freeze_module_list='qwen_vl_interface'
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
data_root_dir=/home/xhy/data/StarVLA/RoboTwin-Clean-verb-base
config_yaml=./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml
run_root_dir=./results/Checkpoints
data_mix=robotwin_clean_verb_base
run_id=0601_${data_mix}_qwen3PIv3

# QwenPI_v3-specific: compress VLM hidden (2048 for Qwen3-VL-4B) to this dim before Action DiT
# action_dit_hidden_dim=1024

# QwenPI_v3 reads repeated_diffusion_steps from trainer.*, not framework.action_model.*
repeated_diffusion_steps=1

num_processes=4   
per_device_batch_size=1
gradient_accumulation_steps=4
# === End of environment variable configuration ===
###########################################################################################

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
  --framework.action_model.action_model_type LayerwiseFM \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.video_backend pyav \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.repeated_diffusion_steps "${repeated_diffusion_steps}" \
  --trainer.learning_rate.base 2.5e-05 \
  --trainer.learning_rate.action_model 1.0e-04 \
  --trainer.is_resume false \
  --trainer.gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Robotwin \
  --wandb_entity haoyuxiong \
  2>&1 | tee "${output_dir}/train.log"
  # --is_debug True
