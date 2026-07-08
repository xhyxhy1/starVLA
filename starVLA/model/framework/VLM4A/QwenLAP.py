# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
Qwen-LAP Framework

Language-Action Pre-training (LAP, arxiv 2602.10556) style: train the VLM to
*speak* low-level robot actions as natural language, supervised by cross-entropy.

Key points:
  - Qwen3-VL backbone (no special action tokens required).
  - Actions are converted to LAP-style text on the fly (textualize_action),
    e.g. "move forward 9.8 cm, rotate 72.0 degrees, close gripper".
  - Trained via next-token prediction on the assistant reply (role masking).
  - This stage trains the VLM ONLY. Continuous control is expected to come from
    a separate action expert (AE) attached later; predict_action here just emits
    the generated language action for inspection and is NOT decoded back into
    continuous actions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.lap_textualize import textualize_action_batch
from starVLA.model.modules.vlm import get_vlm_model


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenLAP
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenLAPDefaultConfig:
    """QwenLAP framework default parameters.

    Natural-language action generation via cross-entropy supervision.
    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.
    """

    # --- Registry identifier ---
    name: str = "QwenLAP"

    # === VLM backbone (Qwen3-VL; plain instruct model, no action tokens) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
        }
    )

    # === Action textualization settings ===
    action_model: dict = field(
        default_factory=lambda: {
            # Tag only (no learnable action head at this stage).
            "action_model_type": "LAP_text",
            # Dimensionality of each action vector (6-DoF + gripper).
            "action_dim": 7,
            # Canonical chunk length used to build the language action.
            "action_horizon": 16,
            # Max new tokens to generate at inference (LAP sentences are short).
            "max_new_tokens": 64,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenLAP")
class Qwenvl_LAP(baseframework):
    """
    Vision-language-action model (LAP variant, VLM-only training stage).

    Components:
      - Qwen3-VL backbone for fused language/vision token modeling.
      - On-the-fly action -> language textualization for CE supervision.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenLAPDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.max_new_tokens = int(self.config.framework.action_model.get("max_new_tokens", 64))

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward: next-token prediction over the LAP-style language action.

        Flow:
          1. Convert each action chunk to LAP text via textualize_action.
          2. Build QwenVL inputs with the text as assistant solution (role masking).
          3. Run VLM and return its cross-entropy loss.

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]

        Returns:
            dict: {"action_loss": torch.Tensor scalar (cross-entropy)}.
        """
        batch_images = [example["image"] for example in examples]  # [B, [PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        # Take the last action_horizon steps as the supervised chunk.
        actions = [np.asarray(example["action"])[-self.action_horizon :] for example in examples]

        # Step 0: action -> LAP-style language
        language_actions = textualize_action_batch(actions)  # List[str]

        # Step 1: build QwenVL inputs with role-based label masking
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            solutions=language_actions,
            mask_mode="role",
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

        ce_loss = qwenvl_outputs.loss
        if ce_loss is None or torch.isnan(ce_loss):
            ce_loss = torch.tensor(0.0, device=self.qwen_vl_interface.model.device)

        return {"action_loss": ce_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> dict:
        """
        Inference: generate the LAP-style language action (for inspection).

        NOTE: This stage does NOT decode language back into continuous actions.
        Continuous control is expected from a separate action expert added later.

        Returns:
            dict: {"language_actions": List[str]}.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  # [B, [PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        input_len = qwen_inputs["input_ids"].shape[1]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.qwen_vl_interface.model.generate(
                **qwen_inputs,
                max_new_tokens=self.max_new_tokens,
            )

        # Decode only the newly generated tokens (strip the prompt).
        new_tokens = generated_ids[:, input_len:]
        language_actions = self.qwen_vl_interface.processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )
        return {"language_actions": language_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_lap_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model = Qwenvl_LAP(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float32),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action (CE) Loss: {action_loss.item()}")

    predict_output = model.predict_action([sample])
    print(f"Language actions: {predict_output['language_actions']}")

    print("Finished")
