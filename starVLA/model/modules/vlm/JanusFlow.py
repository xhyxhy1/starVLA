# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import sys
from pathlib import Path
import torch
from typing import Optional, List
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn as nn

# Add Janus repo to path
janus_path = Path("/home/xhy/Janus")
if str(janus_path) not in sys.path:
    sys.path.insert(0, str(janus_path))

from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor

from accelerate.logging import get_logger

logger = get_logger(__name__)


class _JanusFlow_VL_Interface(nn.Module):
    """
    Lightweight wrapper around JanusFlow-1.3B (MultiModalityCausalLM).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.
    
    Official reference: /home/xhy/Janus/inference.py
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the JanusFlow wrapper following official usage.
        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "deepseek-ai/JanusFlow-1.3B")

        # Load processor (official way)
        vl_chat_processor = VLChatProcessor.from_pretrained(model_id)
        tokenizer = vl_chat_processor.tokenizer
        
        # Override padding side for batch inference
        tokenizer.padding_side = "left"

        # Load model (official way with trust_remote_code)
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Move to appropriate dtype and device
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        vl_gpt = vl_gpt.to(torch_dtype)

        self.model = vl_gpt
        self.processor = vl_chat_processor
        self.tokenizer = tokenizer
        self.config = config

        # Align hidden size with other backbones
        if hasattr(vl_gpt.config, "language_config"):
            self.model.config.hidden_size = vl_gpt.config.language_config.hidden_size
        else:
            logger.warning("Could not find language_config.hidden_size, using default 2048")
            self.model.config.hidden_size = 2048

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to the underlying language model.
        Expects inputs_embeds + attention_mask from build_qwenvl_inputs.
        """
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        attention_mask = kwargs.pop("attention_mask", None)
        output_attentions = kwargs.pop("output_attentions", False)
        output_hidden_states = kwargs.pop("output_hidden_states", True)
        return_dict = kwargs.pop("return_dict", True)

        language_model = getattr(self.model, "language_model", self.model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        return outputs

    def generate(self, **kwargs):
        """
        High-level generation interface (auto-regressive decoding).
        """
        language_model = getattr(self.model, "language_model", self.model)
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = language_model.generate(**kwargs)
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, **kwargs):
        """
        Build model inputs from raw data (images + instructions) using VLChatProcessor.
        
        Following official usage from /home/xhy/Janus/inference.py:
        - conversation format with "role", "content", "images"
        - images should be PIL.Image objects (already provided)
        - use VLChatProcessor with force_batchify=True for batching
        - call prepare_inputs_embeds to get fused embeddings
        """
        assert len(images) == len(instructions), "Images and instructions must have the same length"

        prepare_list = []
        for imgs, instruction in zip(images, instructions):
            # Apply CoT prompt if configured
            if "CoT_prompt" in self.config.datasets.vla_data:
                cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = cot_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            # Official conversation format
            # Note: JanusFlow expects "<image_placeholder>" in content, but VLChatProcessor
            # will handle the mapping automatically based on the images list
            conversation = [
                {"role": "User", "content": prompt, "images": imgs},
                {"role": "Assistant", "content": ""},
            ]

            # Process single conversation (force_batchify=False for accumulation)
            prepare = self.processor(
                conversations=conversation,
                images=imgs,  # PIL images already
                force_batchify=False,
            )
            prepare_list.append(prepare)

        # Batch all conversations together
        prepare_inputs = self.processor.batchify(prepare_list)
        
        # Move to device
        prepare_inputs = prepare_inputs.to(
            self.model.device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        # Prepare fused embeddings (official API)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": prepare_inputs.attention_mask,
        }

