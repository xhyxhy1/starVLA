#!/bin/bash
# Small-scale test training script for JanusFlow on LIBERO subset
# Use this to verify training pipeline before full-scale training

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

###########################################################################################
# === Test Configuration (Small Scale) ===
Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=playground/Pretrained_models/JanusFlow-1.3B
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero_janusflow.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_10
run_root_dir=./results/Checkpoints
run_id=test_janusflow_small
# === End of configuration ===
###########################################################################################

# Disable wandb for quick testing (optional, comment out to enable)
# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

echo "========================================="
echo "Small-Scale Test Training"
echo "Data: 379 episodes (subset)"
echo "Steps: 100 (quick validation)"
echo "========================================="

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100 \
  --trainer.save_interval 50 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 50 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA \
  --wandb_entity 3167962078-zhongguancun \
  --is_debug False

echo "========================================="
echo "Test training completed!"
echo "Check results at: ${output_dir}"
echo "========================================="
