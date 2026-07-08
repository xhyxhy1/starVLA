#!/usr/bin/env bash
# ============================================================================
# LIBERO Benchmark Results Collector
# ============================================================================
# Collects success rates from evaluation logs and outputs a formatted table.
#
# Usage:
#   bash collect_libero_results.sh [checkpoint_root_dir]
#
# Examples:
#   # Collect all results
#   bash collect_libero_results.sh playground/Checkpoints
#
#   # Collect specific model
#   bash collect_libero_results.sh playground/Checkpoints/libero4in1_qwenpi_v3
#
# Output format: Model | Steps | Epoch | Spatial | Object | Goal | Long | Avg
# ============================================================================

# Default checkpoint root directory
CHECKPOINT_ROOT="${1:-playground/Checkpoints}"

# Temporary file for storing parsed results
TMP_RESULTS=$(mktemp)
trap "rm -f $TMP_RESULTS" EXIT

# Suites to collect (in order)
# Note: libero_10 is the "Long" suite
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

echo "Scanning checkpoint logs in: $CHECKPOINT_ROOT"
echo

# Find all model directories
for model_dir in "$CHECKPOINT_ROOT"/*/; do
    if [[ ! -d "$model_dir" ]]; then
        continue
    fi

    model_dir=$(realpath "$model_dir")
    model_name=$(basename "$model_dir")

    # Skip if no logs directory
    logs_dir="$model_dir/logs"
    if [[ ! -d "$logs_dir" ]]; then
        continue
    fi

    # Collect all unique step numbers across all suites
    declare -A seen_steps

    for suite in "${SUITES[@]}"; do
        suite_log_dir="$logs_dir/$suite"
        if [[ ! -d "$suite_log_dir" ]]; then
            continue
        fi

        while IFS= read -r log_file; do
            # Extract step number from filename
            if [[ $log_file =~ steps_([0-9]+)_pytorch_model\.pt\.log$ ]]; then
                step="${BASH_REMATCH[1]}"
                seen_steps["$step"]=1
            fi
        done < <(find "$suite_log_dir" -name "*.log" -type f 2>/dev/null || true)
    done

    # For each unique step, collect all suite results
    for step in $(printf "%s\n" "${!seen_steps[@]}" | sort -n); do
        spatial="N/A"
        object="N/A"
        goal="N/A"
        long="N/A"

        # Extract success rate for each suite
        for suite in "${SUITES[@]}"; do
            suite_log_dir="$logs_dir/$suite"
            if [[ ! -d "$suite_log_dir" ]]; then
                continue
            fi

            # Find the log file for this step
            log_pattern="*steps_${step}_pytorch_model.pt.log"
            log_file=$(find "$suite_log_dir" -name "$log_pattern" -type f 2>/dev/null | head -1)

            if [[ -n "$log_file" ]]; then
                sr=$(grep ">> Total success rate:" "$log_file" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" || true)
                if [[ -n "$sr" ]]; then
                    case "$suite" in
                        "libero_spatial") spatial="$sr" ;;
                        "libero_object") object="$sr" ;;
                        "libero_goal") goal="$sr" ;;
                        "libero_10") long="$sr" ;;
                    esac
                fi
            fi
        done

        # Calculate average (skip N/A values)
        avg_val=0
        count=0
        for val in "$spatial" "$object" "$goal" "$long"; do
            if [[ "$val" != "N/A" ]]; then
                avg_val=$(awk "BEGIN {print $avg_val + $val}")
                ((count++))
            fi
        done
        if [[ $count -gt 0 ]]; then
            avg=$(awk "BEGIN {printf \"%.4f\", $avg_val / $count}")
        else
            avg="N/A"
        fi

        # Epoch calculation (assuming ~2360 steps per epoch for LIBERO 4in1)
        epoch=$(awk "BEGIN {printf \"%.1f\", $step / 2360}")

        # Only output if at least one suite has results
        if [[ "$spatial" != "N/A" || "$object" != "N/A" || "$goal" != "N/A" || "$long" != "N/A" ]]; then
            echo "$model_name|$step|$epoch|$spatial|$object|$goal|$long|$avg" >> "$TMP_RESULTS"
        fi
    done
done

# Sort by model name, then by step
sort -t '|' -k1,1 -k2,2n "$TMP_RESULTS" -o "$TMP_RESULTS"

# Print header
printf "+ %-30s | %8s | %6s | %7s | %7s | %6s | %6s | %6s\n" \
    "Model" "Steps" "Epoch" "Spatial" "Object" "Goal" "Long" "Avg"
printf "+-%.0s-+-%.0s-+-%.0s-+-%.0s-+-%.0s-+-%.0s-+-%.0s-+-%.0s-+\n" \
    {1..8} | tr '-' '='

# Print data
while IFS='|' read -r model step epoch spatial object goal long avg; do
    printf "%-32s | %8s | %6s | %7s | %7s | %6s | %6s | %6s\n" \
        "$model" "$step" "$epoch" "$spatial" "$object" "$goal" "$long" "$avg"
done < "$TMP_RESULTS"

# Summary count
total_results=$(wc -l < "$TMP_RESULTS" 2>/dev/null || echo "0")
echo
echo "Total results: $total_results"
