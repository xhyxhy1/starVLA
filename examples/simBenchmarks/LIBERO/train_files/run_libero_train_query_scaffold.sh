#!/usr/bin/env bash
# QwenGR00TQueryScaffold training on LIBERO — single-node 4-GPU, no DeepSpeed
# Usage: bash examples/LIBERO/train_files/run_libero_train_query_scaffold.sh

# 可按需限制可见 GPU，例如 0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# 不强绑网卡/IB，避免设备名不匹配导致 NCCL 初始化失败
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export WANDB_MODE=online

###############################################################################
# === 按需修改以下路径 ===
Framework_name=QwenGR00TQueryScaffold
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=qwengroot_query_scaffold_libero_4gpu_nodeepspeed
# === 结束 ===
###############################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.vl_hidden_dim 2048 \
  --framework.query_scaffold.auto_add_tokens true \
  --framework.query_scaffold.use_query_in_action true \
  --framework.query_scaffold.query_gate_init -5.0 \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
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
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero_QueryScaffold \
  --wandb_entity haoyuxiong \
  2>&1 | tee "${output_dir}/train.log"
