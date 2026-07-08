#!/bin/bash
# LIBERO evaluation for CosmoPredict2PI-trained checkpoint (same flow as eval_libero.sh).
# Run from repo root. Use LIBERO env. Start run_policy_server_cosmospi.sh first (starVLA env).

cd /home/xhy/starVLA
# conda activate libero  # ensure LIBERO env is active

###########################################################################################
# === Modify paths to match your environment ===
export LIBERO_HOME=/home/xhy/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/home/xhy/miniconda3/envs/libero/bin/python

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
###########################################################################################

RUN_DIR=/home/xhy/starVLA/playground/Checkpoints/libero4in1_cosmospi
your_ckpt=${RUN_DIR}/checkpoints/steps_50000_pytorch_model.pt

host="127.0.0.1"
base_port=6695
unnorm_key="franka"

# Match run_policy_server_cosmospi.sh port and ckpt
folder_name="libero4in1_cosmospi_steps_50000"

LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Task suite: libero_spatial | libero_object | libero_goal | libero_10
task_suite_name=libero_goal
num_trials_per_task=50
video_out_path="${RUN_DIR}/videos/${task_suite_name}/${folder_name}"
mkdir -p "${video_out_path}"

# Dataloader uses 224x224; keep eval aligned (EVAL_RESIZE_SIZE=224).
export EVAL_RESIZE_SIZE=224
${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${your_ckpt}" \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task $num_trials_per_task \
    --args.video-out-path "$video_out_path" \
    2>&1 | tee "${LOG_DIR}/eval_${task_suite_name}_${folder_name}.log"

echo "Done. Videos: ${video_out_path}, Log: ${LOG_DIR}/eval_${task_suite_name}_${folder_name}.log"
