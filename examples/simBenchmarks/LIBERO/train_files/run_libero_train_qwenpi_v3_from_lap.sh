#!/usr/bin/env bash
# QwenPI_v3 Stage 2 training from a QwenLAP-trained VLM.
#
# Stage 1 trains QwenLAP so the VLM learns LAP-style language-action grounding.
# Stage 2 builds QwenPI_v3, loads only qwen_vl_interface from the LAP checkpoint,
# freezes that VLM, and trains the flow-matching action expert plus projectors.
#
# Usage:
#   bash examples/LIBERO/train_files/run_libero_train_qwenpi_v3_from_lap.sh
#   LAP_CKPT=./playground/Checkpoints/libero_lap_v1/checkpoints/steps_80000_pytorch_model.pt \
#     bash examples/LIBERO/train_files/run_libero_train_qwenpi_v3_from_lap.sh 4 160000

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
framework_name=QwenPI_v3
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=libero4in1_qwenpi_v3_from_lap

# The LAP checkpoint must come from the same Qwen base_vlm architecture.
lap_checkpoint=${LAP_CKPT:-./playground/Checkpoints/libero_lap_v1/checkpoints/steps_80000_pytorch_model.pt}

# === QwenPI_v3-specific knobs ===
action_dit_hidden_dim=1024
repeated_diffusion_steps=2

# === Training knobs ===
num_processes=${1:-4}
max_train_steps=${2:-160000}
per_device_batch_size=8
save_interval=20000
logging_frequency=100

# Train only the newly attached action expert and per-layer projectors.
learning_rate_base=2.5e-05
learning_rate_action_model=1.0e-04
learning_rate_project_layers=1.0e-04
###############################################################################

if [[ ! -f "${lap_checkpoint}" ]]; then
  echo "LAP checkpoint not found: ${lap_checkpoint}" >&2
  echo "Set LAP_CKPT to a QwenLAP checkpoint path." >&2
  exit 1
fi

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.action_model.diffusion_model_cfg.action_dit_hidden_dim "${action_dit_hidden_dim}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.pretrained_checkpoint "${lap_checkpoint}" \
  --trainer.reload_modules qwen_vl_interface \
  --trainer.freeze_modules qwen_vl_interface \
  --trainer.repeated_diffusion_steps "${repeated_diffusion_steps}" \
  --trainer.learning_rate.base "${learning_rate_base}" \
  --trainer.learning_rate.action_model "${learning_rate_action_model}" \
  --trainer.learning_rate.project_layers "${learning_rate_project_layers}" \
  --trainer.loss_scale.vla 1.0 \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.is_resume true \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --trainer.eval_interval 100 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Libero_QwenPI_v3_From_LAP \
  --wandb_entity haoyuxiong \
  2>&1 | tee "${output_dir}/train.log"
