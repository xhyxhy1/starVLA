#!/usr/bin/env bash
# WM4A (CosmoPredict2GR00T) training on LIBERO — single-node DeepSpeed
# Usage: bash examples/LIBERO/train_files/run_libero_train_wm4a_deepspeed.sh

# 可按需限制可见 GPU，例如 0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL/Torch 分布式诊断与容错
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export WANDB_MODE=online

###############################################################################
# === 按需修改以下路径与参数 ===
Framework_name=CosmoPredict2GR00T
base_wm=./playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=wm4a_cosmo_groot_libero_4gpu_deepspeed
num_processes=4
wandb_project=starVLA_Libero_WM4A
# 优先读取环境变量 WANDB_ENTITY；如需固定可直接改为具体字符串
wandb_entity=haoyuxiong
# === 结束 ===
###############################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

wandb_args=(--wandb_project "${wandb_project}")
if [ -n "${wandb_entity}" ]; then
  wandb_args+=(--wandb_entity "${wandb_entity}")
fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.world_model.base_wm "${base_wm}" \
  --framework.qwenvl.base_vlm "${base_wm}" \
  --framework.qwenvl.vl_hidden_dim 2048 \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.learning_rate.base 2.5e-05 \
  --trainer.learning_rate.action_model 1.0e-04 \
  --trainer.loss_scale.vla 1.0 \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  "${wandb_args[@]}" \
  2>&1 | tee "${output_dir}/train.log"
