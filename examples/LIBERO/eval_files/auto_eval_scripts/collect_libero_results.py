#!/usr/bin/env python3
"""
LIBERO Benchmark Results Collector

Collects success rates from evaluation logs and outputs:
1. Terminal table
2. Excel file (saved to current directory as libero_results.xlsx)

Usage:
    python collect_libero_results.py [checkpoint_root_dir]

Examples:
    # Collect all results
    python collect_libero_results.py playground/Checkpoints

    # Collect specific model
    python collect_libero_results.py playground/Checkpoints/libero4in1_qwenpi_v3
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Excel output will be skipped.")


# Suites to collect (matching directory names)
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
STEPS_PER_EPOCH = 2360  # Adjust if your training has different steps-per-epoch


def extract_success_rate(log_file: Path) -> Optional[float]:
    """Extract success rate from log file."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            # Find all success rate lines
            matches = re.findall(r'>> Total success rate: ([\d.]+)', content)
            if matches:
                return float(matches[-1])  # Use the last one
    except Exception:
        pass
    return None


def extract_step_from_filename(filename: str) -> Optional[int]:
    """Extract step number from filename: *_steps_{N}_pytorch_model.pt.log"""
    match = re.search(r'steps_(\d+)_pytorch_model\.pt\.log', filename)
    return int(match.group(1)) if match else None


def collect_results(checkpoint_root: str) -> List[Dict]:
    """Collect results from all model directories."""
    results = []
    checkpoint_path = Path(checkpoint_root)

    if not checkpoint_path.exists():
        print(f"Error: Path does not exist: {checkpoint_root}")
        return results

    # Find all model directories
    for model_dir in checkpoint_path.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        logs_dir = model_dir / "logs"

        if not logs_dir.exists():
            continue

        # Collect all unique step numbers across all suites
        seen_steps = set()

        for suite in SUITES:
            suite_log_dir = logs_dir / suite
            if suite_log_dir.exists():
                for log_file in suite_log_dir.glob("*.log"):
                    step = extract_step_from_filename(log_file.name)
                    if step:
                        seen_steps.add(step)

        # For each unique step, collect all suite results
        for step in sorted(seen_steps):
            suite_results = {}

            for suite in SUITES:
                suite_log_dir = logs_dir / suite
                if not suite_log_dir.exists():
                    continue

                # Find the log file for this step
                log_pattern = f"*steps_{step}_pytorch_model.pt.log"
                log_files = list(suite_log_dir.glob(log_pattern))

                if log_files:
                    sr = extract_success_rate(log_files[0])
                    if sr is not None:
                        suite_results[suite] = sr

            # Skip if no results
            if not suite_results:
                continue

            # Calculate epoch
            epoch = round(step / STEPS_PER_EPOCH, 1)

            # Build result row
            row = {
                "Model": model_name,
                "Steps": step,
                "Epoch": epoch,
                "Spatial": suite_results.get("libero_spatial", "N/A"),
                "Object": suite_results.get("libero_object", "N/A"),
                "Goal": suite_results.get("libero_goal", "N/A"),
                "Long": suite_results.get("libero_10", "N/A"),  # libero_10 = Long
            }

            # Calculate average (skip N/A values)
            values = [v for v in suite_results.values() if isinstance(v, float)]
            if values:
                row["Avg"] = round(sum(values) / len(values), 4)
            else:
                row["Avg"] = "N/A"

            results.append(row)

    # Sort by model name, then by step
    results.sort(key=lambda x: (x["Model"], x["Steps"]))
    return results


def print_table(results: List[Dict]):
    """Print results as a formatted table."""
    if not results:
        print("No results found.")
        return

    # Column widths
    cols = ["Model", "Steps", "Epoch", "Spatial", "Object", "Goal", "Long", "Avg"]
    widths = [32, 10, 8, 9, 9, 8, 8, 8]

    # Print header
    header = " | ".join(col.ljust(w) for col, w in zip(cols, widths))
    separator = "=" * len(header)
    print(separator)
    print(header)
    print(separator)

    # Print rows
    for r in results:
        row = " | ".join(str(r[col]).ljust(w) for col, w in zip(cols, widths))
        print(row)

    print(separator)
    print(f"\nTotal results: {len(results)}")


def save_excel(results: List[Dict], output_path: str = "libero_results.xlsx"):
    """Save results to Excel file."""
    if not HAS_PANDAS:
        print("Skip Excel output (pandas not installed)")
        return

    if not results:
        print("No results to save.")
        return

    # Column order
    cols = ["Model", "Steps", "Epoch", "Spatial", "Object", "Goal", "Long", "Avg"]
    df = pd.DataFrame(results, columns=cols)

    # Save to Excel
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\nExcel saved to: {output_path}")


def main():
    import sys

    checkpoint_root = sys.argv[1] if len(sys.argv) > 1 else "playground/Checkpoints"

    print(f"Scanning checkpoint logs in: {checkpoint_root}\n")

    results = collect_results(checkpoint_root)
    print_table(results)

    if HAS_PANDAS:
        save_excel(results)


if __name__ == "__main__":
    main()
