#!/usr/bin/env bash
# WM4A (CosmoPredict2GR00T) single-GPU training on LIBERO
# Usage: bash examples/LIBERO/train_files/run_libero_train_wm4a_1gpu.sh

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

###############################################################################
# === 按需修改 ===
Framework_name=CosmoPredict2GR00T
base_wm=./playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=wm4a_cosmo_groot_libero_1gpu
# === 结束 ===
###############################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

python starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.world_model.base_wm ${base_wm} \
  --framework.qwenvl.base_vlm ${base_wm} \
  --framework.qwenvl.vl_hidden_dim 2048 \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.learning_rate.base 2.5e-05 \
  --trainer.learning_rate.action_model 1.0e-04 \
  --trainer.loss_scale.vla 1.0 \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  2>&1 | tee "${output_dir}/train.log"
