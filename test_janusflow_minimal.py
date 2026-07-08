#!/usr/bin/env python3
"""
Minimal test script for JanusFlow VLM wrapper integration.
Based on official usage from /home/xhy/Janus/inference.py
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

# Add starVLA to path
sys.path.insert(0, "/home/xhy/starVLA")

print("=" * 80)
print("JanusFlow VLM Wrapper Minimal Test")
print("=" * 80)

# Load config
cfg_path = "./examples/LIBERO/train_files/starvla_cotrain_libero_janusflow.yaml"
print(f"\n[1/5] Loading config: {cfg_path}")
cfg = OmegaConf.load(cfg_path)

# Override to use local model
local_model = "./playground/Pretrained_models/JanusFlow-1.3B"
if os.path.exists(local_model):
    print(f"[INFO] Using local model: {local_model}")
    cfg.framework.qwenvl.base_vlm = local_model
else:
    print(f"[INFO] Using HuggingFace model: {cfg.framework.qwenvl.base_vlm}")

# Import framework
print("\n[2/5] Importing framework...")
from starVLA.model.framework.QwenGR00T import Qwen_GR00T

# Initialize model
print("\n[3/5] Initializing model...")
model = Qwen_GR00T(cfg)

# Create fake sample (384x384 to match JanusFlow default)
print("\n[4/5] Creating test sample...")
image = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
sample = {
    "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float32),
    "image": [image],
    "lang": "Pick up the red block and place it into the blue box.",
}

batch = [sample]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Moving model to {device}...")
model = model.to(device)

# Test forward pass
print("\n[5/5] Running forward pass...")
try:
    with torch.no_grad():
        out = model(batch)
    print(f"✓ Forward pass successful!")
    print(f"  - action_loss: {out['action_loss'].item():.6f}")
except Exception as e:
    print(f"✗ Forward pass failed!")
    print(f"  - Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test predict_action
print("\nRunning predict_action...")
try:
    with torch.no_grad():
        pred = model.predict_action(examples=batch)
    print(f"✓ Predict successful!")
    print(f"  - normalized_actions shape: {np.array(pred['normalized_actions']).shape}")
except Exception as e:
    print(f"✗ Predict failed!")
    print(f"  - Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ All tests passed! JanusFlow VLM wrapper is working correctly.")
print("=" * 80)
