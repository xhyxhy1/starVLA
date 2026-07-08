import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm

from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.action_model.lap_textualize import (
    language_action_to_interpolated_action,
    textualize_action,
)
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


def build_eval_row(
    sample_id: int,
    instruction: str,
    pred_language_action: str,
    gt_action: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    """Build one JSON-serializable LAP checkpoint eval row."""
    gt_action = np.asarray(gt_action, dtype=np.float32)[-horizon:]
    gt_language_action = textualize_action(gt_action)

    row = {
        "sample_id": int(sample_id),
        "instruction": instruction,
        "gt_language_action": gt_language_action,
        "pred_language_action": pred_language_action,
        "lap_interp_mse": None,
        "gripper_match": None,
        "parse_ok": False,
    }

    if not isinstance(pred_language_action, str):
        return row

    try:
        pred_action = language_action_to_interpolated_action(
            pred_language_action,
            horizon=horizon,
            action_dim=gt_action.shape[-1],
        )
    except Exception as exc:  # Keep eval robust; bad generations are part of the signal.
        row["parse_error"] = str(exc)
        return row

    if not np.isfinite(pred_action).all():
        row["parse_error"] = "parsed action contains non-finite values"
        return row

    score = TrainerUtils.euclidean_distance(pred_action[None], gt_action[None])
    row["lap_interp_mse"] = float(score / np.prod(gt_action[None].shape))
    row["gripper_match"] = bool((pred_action[-1, 6] > 0.5) == (gt_action[-1, 6] > 0.5))
    row["parse_ok"] = True
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_mse = [row["lap_interp_mse"] for row in rows if row.get("lap_interp_mse") is not None]
    valid_gripper = [row["gripper_match"] for row in rows if row.get("gripper_match") is not None]
    return {
        "num_samples": len(rows),
        "parse_ok_rate": sum(bool(row.get("parse_ok")) for row in rows) / max(len(rows), 1),
        "mean_lap_interp_mse": float(np.mean(valid_mse)) if valid_mse else None,
        "median_lap_interp_mse": float(np.median(valid_mse)) if valid_mse else None,
        "gripper_match_rate": sum(valid_gripper) / len(valid_gripper) if valid_gripper else None,
    }


def resolve_config_path(checkpoint: Path, config: str | None) -> Path:
    if config:
        return Path(config)
    return checkpoint.parent.parent / "config.yaml"


def ensure_single_process_dist() -> None:
    """Initialize a rank-0 process group for dataloader code that calls dist.get_rank()."""
    if dist.is_available() and not dist.is_initialized():
        init_file = tempfile.NamedTemporaryFile(prefix="starvla_lap_eval_dist_", delete=True)
        init_file.close()
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file.name}",
            rank=0,
            world_size=1,
        )


def load_model(checkpoint: Path, device: str, use_bf16: bool):
    model = baseframework.from_pretrained(str(checkpoint))
    if use_bf16:
        model = model.to(torch.bfloat16)
    return model.to(device).eval()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    config_path = resolve_config_path(checkpoint, args.config)
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")

    cfg = OmegaConf.load(config_path)
    ensure_single_process_dist()

    device = args.device
    use_bf16 = args.bf16 and device != "cpu"
    model = load_model(checkpoint, device=device, use_bf16=use_bf16)
    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    horizon = int(args.horizon or cfg.framework.action_model.get("action_horizon", 8))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    sample_id = 0
    with torch.inference_mode(), output_path.open("w", encoding="utf-8") as f:
        for examples in tqdm(dataloader, desc="Evaluating LAP checkpoint"):
            pred = model.predict_action(examples=examples)
            pred_texts = pred.get("language_actions")
            if pred_texts is None:
                raise KeyError("model.predict_action() did not return 'language_actions'")

            for example, pred_text in zip(examples, pred_texts, strict=True):
                row = build_eval_row(
                    sample_id=sample_id,
                    instruction=example["lang"],
                    pred_language_action=pred_text,
                    gt_action=example["action"],
                    horizon=horizon,
                )
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                sample_id += 1
                if sample_id >= args.num_samples:
                    summary = summarize_rows(rows)
                    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
                    return summary

    summary = summarize_rows(rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline proxy eval for QwenLAP checkpoints.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint .pt/.safetensors file.")
    parser.add_argument("--config", default=None, help="Optional config path. Defaults to <run_dir>/config.yaml.")
    parser.add_argument("--output", required=True, help="JSONL output path for per-sample records.")
    parser.add_argument("--summary", default=None, help="Optional summary JSON path.")
    parser.add_argument("--num-samples", type=int, default=128, help="Number of dataloader samples to evaluate.")
    parser.add_argument("--horizon", type=int, default=None, help="Override action horizon. Defaults to config value.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="Run model in bfloat16.")
    return parser.parse_args()


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
