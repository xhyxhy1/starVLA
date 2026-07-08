import json
import os
from pathlib import Path

from libero_plus_metrics import merge_metric_results


def main() -> None:
    log_dir = os.environ.get("LOG_DIR")
    if not log_dir:
        raise RuntimeError("LOG_DIR is required")

    root = Path(log_dir)
    task_suites = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
    results = []
    for task_suite in task_suites:
        for path in sorted(root.glob(f"{task_suite}_*_to_*.json")):
            with open(path, encoding="utf-8") as f:
                results.append(json.load(f))

    if not results:
        raise RuntimeError(f"No LIBERO-plus result json files found in {root}")

    overall_results = merge_metric_results(results)
    with open(root / "overall_results.json", "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2)


if __name__ == "__main__":
    main()
