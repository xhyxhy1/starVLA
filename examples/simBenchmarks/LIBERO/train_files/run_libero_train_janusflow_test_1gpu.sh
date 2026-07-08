#!/bin/bash
# Single-GPU test training (avoids NCCL multi-node setup)

Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=playground/Pretrained_models/JanusFlow-1.3B
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero_janusflow.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_10
run_root_dir=./results/Checkpoints
run_id=test_janusflow_small_1gpu

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" ${output_dir}/

echo "========================================="
echo "Single-GPU Test Training"
echo "Data: 379 episodes | Steps: 100"
echo "========================================="

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 4 \
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
echo "Test training completed! Check: ${output_dir}"
echo "========================================="
