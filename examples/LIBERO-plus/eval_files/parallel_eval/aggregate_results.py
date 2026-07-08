import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libero_plus_metrics import merge_metric_results


def main() -> None:
    parser = argparse.ArgumentParser(description="aggregate LIBERO-plus results")
    parser.add_argument("--root_path", required=True)
    args = parser.parse_args()

    log_dir = Path(args.root_path)
    task_suites = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
    results = []
    for task_suite in task_suites:
        cur_root = log_dir / "logs" / task_suite
        json_files = sorted(glob.glob(os.path.join(cur_root, "*.json")))
        for file in json_files:
            with open(file, encoding="utf-8") as f:
                results.append(json.load(f))

    overall_results = merge_metric_results(results)
    with open(log_dir / "overall_results.json", "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2)


if __name__ == "__main__":
    main()
