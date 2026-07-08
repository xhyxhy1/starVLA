#!/bin/bash
# Policy server for CosmoPredict2PI-trained LIBERO checkpoint (same flow as run_policy_server.sh).
# Run from repo root. Use starVLA env (needs diffusers + Cosmos deps). Port must match eval_libero_cosmospi.sh.

export PYTHONPATH=$(pwd):${PYTHONPATH}
export star_vla_python=/home/xhy/miniconda3/envs/starVLA/bin/python

# CosmoPredict2PI checkpoint (change to steps_10000 ... steps_50000 as needed)
RUN_DIR=/home/xhy/starVLA/playground/Checkpoints/libero4in1_cosmospi
your_ckpt=${RUN_DIR}/checkpoints/steps_50000_pytorch_model.pt

gpu_id=0
port=6695

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port ${port} \
    --use_bf16
