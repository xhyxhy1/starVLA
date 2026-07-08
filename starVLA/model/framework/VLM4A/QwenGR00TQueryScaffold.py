# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
QwenGR00T query scaffold.

This framework keeps the QwenGR00T action objective unchanged while adding
special query tokens that can be monitored or gated into the action head.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenGR00TQueryScaffoldDefaultConfig(QwenGR00TDefaultConfig):
    """QwenGR00T with gated semantic query scaffold."""

    name: str = "QwenGR00TQueryScaffold"
    query_scaffold: dict = field(
        default_factory=lambda: {
            "tokens": ["<|skill_query|>", "<|object_query|>", "<|progress_query|>"],
            "auto_add_tokens": True,
            "use_query_in_action": True,
            "query_gate_init": -5.0,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00TQueryScaffold")
class Qwen_GR00T_QueryScaffold(baseframework):
    """
    QwenGR00T variant with special query tokens and near-zero gated action injection.

    The scaffold intentionally adds no auxiliary semantic loss. Query states are
    trained only through the original action loss when ``use_query_in_action`` is enabled.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenGR00TQueryScaffoldDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        query_cfg = self.config.framework.get("query_scaffold", {})
        self.query_token_names = list(
            query_cfg.get("tokens", ["<|skill_query|>", "<|object_query|>", "<|progress_query|>"])
        )
        if len(self.query_token_names) == 0:
            raise ValueError("query_scaffold.tokens must contain at least one query token.")

        self.query_tokens = "".join(self.query_token_names)
        self.query_token_ids = None
        self.auto_add_query_tokens = bool(query_cfg.get("auto_add_tokens", True))
        self.use_query_in_action = bool(query_cfg.get("use_query_in_action", True))
        self.query_gate = torch.nn.Parameter(torch.tensor(float(query_cfg.get("query_gate_init", -5.0))))

        self._ensure_query_tokens_in_tokenizer()

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    # ---------------------------------------------------------------------
    # Query token helpers
    # ---------------------------------------------------------------------
    def _ensure_query_tokens_in_tokenizer(self) -> None:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        if not self.auto_add_query_tokens:
            return

        added = tokenizer.add_special_tokens({"additional_special_tokens": self.query_token_names})
        if added > 0 and hasattr(self.qwen_vl_interface.model, "resize_token_embeddings"):
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))

    def _ensure_query_token_ids(self, tokenizer):
        if self.query_token_ids is not None:
            return

        token_ids = []
        for token in self.query_token_names:
            token_id = tokenizer.convert_tokens_to_ids(token)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if token_id is None or (unk_id is not None and int(token_id) == int(unk_id)):
                raise ValueError(
                    f"Query token {token!r} is not available in the tokenizer. "
                    "Enable query_scaffold.auto_add_tokens or add it before training."
                )
            token_ids.append(int(token_id))

        if len(set(token_ids)) != len(token_ids):
            raise ValueError(f"Query tokens must map to distinct token ids, got {token_ids}.")
        self.query_token_ids = token_ids

    def _extract_query_hidden_states(
        self,
        hidden_states: torch.Tensor,  # [B, S, H]
        input_ids: torch.Tensor,  # [B, S]
        tokenizer,
    ) -> torch.Tensor:
        self._ensure_query_token_ids(tokenizer)

        out = []
        for b in range(hidden_states.shape[0]):
            sample_queries = []
            for token_id in self.query_token_ids:
                pos = (input_ids[b] == int(token_id)).nonzero(as_tuple=True)[0]
                if pos.numel() == 0:
                    raise AssertionError(f"Missing query token id {token_id} in sample {b}.")
                sample_queries.append(hidden_states[b, int(pos[0].item()), :])
            out.append(torch.stack(sample_queries, dim=0))
        return torch.stack(out, dim=0)  # [B, Q, H]

    def _build_action_condition(self, last_hidden: torch.Tensor, query_hidden: torch.Tensor) -> torch.Tensor:
        if not self.use_query_in_action:
            return last_hidden
        gate = torch.sigmoid(self.query_gate).to(device=query_hidden.device, dtype=query_hidden.dtype)
        return torch.cat([last_hidden, gate * query_hidden], dim=1)

    def _query_diagnostics(self, query_hidden: torch.Tensor) -> dict:
        q = query_hidden.detach().float()
        q_norm = q.norm(dim=-1).mean()
        gate = torch.sigmoid(self.query_gate.detach()).float()

        if q.shape[1] < 2:
            pairwise_cosine = torch.tensor(0.0, device=q.device)
        else:
            q_unit = F.normalize(q, dim=-1)
            sims = []
            for i in range(q.shape[1]):
                for j in range(i + 1, q.shape[1]):
                    sims.append((q_unit[:, i] * q_unit[:, j]).sum(dim=-1).mean())
            pairwise_cosine = torch.stack(sims).mean()

        return {
            "query_gate": gate,
            "query_hidden_norm": q_norm,
            "query_pairwise_cosine": pairwise_cosine,
        }

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] + self.query_tokens for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            query_hidden = self._extract_query_hidden_states(
                last_hidden, qwen_inputs["input_ids"], self.qwen_vl_interface.processor.tokenizer
            )
            action_condition = self._build_action_condition(last_hidden, query_hidden)

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            action_condition_repeated = action_condition.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(action_condition_repeated, actions_target_repeated, state_repeated)

        out = {"action_loss": action_loss}
        out.update(self._query_diagnostics(query_hidden))
        return out

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]

        batch_images = []
        for example in examples:
            imgs = example["image"]
            if isinstance(imgs, list):
                batch_images.append([to_pil_preserve(image) for image in imgs])
            else:
                batch_images.append([to_pil_preserve(imgs)])

        instructions = [example["lang"] + self.query_tokens for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            query_hidden = self._extract_query_hidden_states(
                last_hidden, qwen_inputs["input_ids"], self.qwen_vl_interface.processor.tokenizer
            )
            action_condition = self._build_action_condition(last_hidden, query_hidden)

        state = (
            torch.from_numpy(np.array(state)).to(action_condition.device, dtype=action_condition.dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(action_condition, state)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.name = "QwenGR00TQueryScaffold"

    model: Qwen_GR00T_QueryScaffold = Qwen_GR00T_QueryScaffold(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }
    batch = [sample, {**sample, "lang": "Another fake instruction for testing."}]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    print("Action Loss:", forward_output["action_loss"].item())
    print("Query Gate:", forward_output["query_gate"].item())

    predict_output = model.predict_action(examples=[sample])
    print("Pred shape:", predict_output["normalized_actions"].shape)
