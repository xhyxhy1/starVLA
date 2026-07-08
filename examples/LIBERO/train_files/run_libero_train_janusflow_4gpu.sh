#!/bin/bash
# 4-GPU full training on complete LIBERO (libero_all), JanusFlow backbone

# NCCL: use lo for single-node multi-GPU if bond0/IB unavailable
export NCCL_SOCKET_IFNAME=lo
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

###########################################################################################
Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=playground/Pretrained_models/JanusFlow-1.3B
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero_janusflow.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=libero_janusflow_4gpu_full
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" ${output_dir}/

echo "========================================="
echo "4-GPU Full Training | libero_all | 80k steps"
echo "========================================="

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA \
  --wandb_entity 3167962078-zhongguancun \
  --is_debug False

echo "========================================="
echo "Training finished. Check: ${output_dir}"
echo "========================================="
