#!/bin/bash
# Policy server for JanusFlow-trained LIBERO checkpoint (same flow as run_policy_server.sh).
# Run from repo root. Use starVLA env. Port must match eval_libero_janusflow.sh.

export PYTHONPATH=$(pwd):${PYTHONPATH}
export star_vla_python=/home/xhy/miniconda3/envs/starVLA/bin/python

# JanusFlow checkpoint (change to steps_5000 / steps_10000 / steps_15000 as needed)
RUN_DIR=/home/xhy/starVLA/results/Checkpoints/libero_janusflow_4gpu_full
your_ckpt=${RUN_DIR}/checkpoints/steps_5000_pytorch_model.pt

gpu_id=0
port=5695

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port ${port} \
    --use_bf16
