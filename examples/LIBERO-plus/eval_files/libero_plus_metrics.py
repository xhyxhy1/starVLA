import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


COUNT_KEYS = ("total_count", "success_count")


def resolve_libero_home() -> Path:
    libero_home = os.environ.get("LIBERO_HOME")
    if libero_home:
        return Path(libero_home)
    config_path = os.environ.get("LIBERO_CONFIG_PATH")
    if config_path:
        return Path(config_path).parent
    raise RuntimeError(
        "LIBERO_HOME is not set. Export it before eval, e.g.\n"
        "  export LIBERO_HOME=/home/xhy/Project/LIBERO-plus"
    )


def load_task_mapping(task_suite_name: str, libero_home: str | Path | None = None) -> list[dict]:
    root = Path(libero_home) if libero_home is not None else resolve_libero_home()
    task_cls_path = root / "libero/libero/benchmark/task_classification.json"
    with open(task_cls_path, encoding="utf-8") as f:
        mappings = json.load(f)
    return mappings[task_suite_name]


def safe_filename_part(value) -> str:
    text = str(value).strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"



def uniform_sample_ids(ids: list[int], keep_count: int) -> list[int]:
    if keep_count <= 0:
        return []
    if keep_count >= len(ids):
        return list(ids)
    if keep_count == 1:
        return [ids[0]]

    positions = [round(i * (len(ids) - 1) / (keep_count - 1)) for i in range(keep_count)]
    selected_positions = []
    seen = set()
    for position in positions:
        if position not in seen:
            selected_positions.append(position)
            seen.add(position)

    if len(selected_positions) < keep_count:
        for position in range(len(ids)):
            if position not in seen:
                selected_positions.append(position)
                seen.add(position)
                if len(selected_positions) == keep_count:
                    break

    return [ids[position] for position in sorted(selected_positions)]


def select_task_ids_by_category_fraction(
    task_mapping: Iterable[Mapping],
    start_idx: int,
    end_idx: int,
    category_fraction: float,
) -> list[int]:
    if not (0.0 < category_fraction <= 1.0):
        raise ValueError(f"category_fraction must be in (0, 1], got {category_fraction}")

    selected_by_category: dict[str, list[int]] = {}
    for item in task_mapping:
        task_id = int(item["id"]) - 1
        if start_idx <= task_id < end_idx:
            selected_by_category.setdefault(str(item["category"]), []).append(task_id)

    task_ids = []
    for category in sorted(selected_by_category):
        ids = selected_by_category[category]
        keep_count = max(1, math.ceil(len(ids) * category_fraction))
        task_ids.extend(uniform_sample_ids(ids, keep_count))

    return sorted(task_ids)


def count_task_ids_by_category(task_mapping: Iterable[Mapping], task_ids: Iterable[int]) -> dict[str, int]:
    selected = set(task_ids)
    counts = {}
    for item in task_mapping:
        task_id = int(item["id"]) - 1
        if task_id in selected:
            category = str(item["category"])
            counts[category] = counts.get(category, 0) + 1
    return counts

def _empty_counts() -> dict[str, int]:
    return {"total_count": 0, "success_count": 0}


def _with_success_rate(counts: Mapping[str, int]) -> dict:
    total = int(counts["total_count"])
    success = int(counts["success_count"])
    success_rate = float(success) / float(total) if total else 0.0
    return {"total_count": total, "success_count": success, "success_rate": success_rate}


def _normalize_difficulty_level(value) -> str:
    return "unknown" if value is None else str(value)


class LiberoPlusMetricRecorder:
    def __init__(self, suite: str, task_mapping: Iterable[Mapping], task_range: tuple[int, int] | None = None) -> None:
        self.suite = suite
        self.task_range = list(task_range) if task_range is not None else None
        self.task_info_by_zero_based_id = {
            int(item["id"]) - 1: {
                "name": item["name"],
                "category": item["category"],
                "difficulty_level": _normalize_difficulty_level(item.get("difficulty_level")),
            }
            for item in task_mapping
        }
        self.overall = _empty_counts()
        self.by_category = defaultdict(_empty_counts)
        self.by_difficulty_level = defaultdict(_empty_counts)

    def task_meta(self, task_id: int) -> dict:
        return self.task_info_by_zero_based_id[task_id]

    def task_name(self, task_id: int) -> str:
        return self.task_meta(task_id)["name"]

    def record_episode(self, task_id: int, success: bool) -> None:
        info = self.task_info_by_zero_based_id[task_id]
        self._add_counts(self.overall, success)
        self._add_counts(self.by_category[info["category"]], success)
        self._add_counts(self.by_difficulty_level[info["difficulty_level"]], success)

    @staticmethod
    def _add_counts(bucket: dict[str, int], success: bool) -> None:
        bucket["total_count"] += 1
        if success:
            bucket["success_count"] += 1

    def to_dict(self) -> dict:
        result = {
            "suite": self.suite,
            "overall": _with_success_rate(self.overall),
            "by_category": {
                key: _with_success_rate(value) for key, value in sorted(self.by_category.items())
            },
            "by_difficulty_level": {
                key: _with_success_rate(value) for key, value in sorted(self.by_difficulty_level.items())
            },
        }
        if self.task_range is not None:
            result["task_range"] = self.task_range
        return result


def merge_metric_results(results: Iterable[Mapping]) -> dict:
    merged = {
        "overall": _empty_counts(),
        "by_category": defaultdict(_empty_counts),
        "by_difficulty_level": defaultdict(_empty_counts),
    }
    for result in results:
        _merge_counts(merged["overall"], result.get("overall", {}))
        for group_name in ("by_category", "by_difficulty_level"):
            for key, counts in result.get(group_name, {}).items():
                _merge_counts(merged[group_name][key], counts)

    return {
        "overall": _with_success_rate(merged["overall"]),
        "by_category": {
            key: _with_success_rate(value) for key, value in sorted(merged["by_category"].items())
        },
        "by_difficulty_level": {
            key: _with_success_rate(value) for key, value in sorted(merged["by_difficulty_level"].items())
        },
    }


def _merge_counts(target: dict[str, int], source: Mapping) -> None:
    for key in COUNT_KEYS:
        target[key] += int(source.get(key, 0))
