#!/usr/bin/env python3
"""
Extract specific metric values from an evaluation log file.
"""
import re
import sys
from pathlib import Path


def extract_metrics(log_file_path, metrics=None):
    """
    Extract the specified metric values from a log file.

    Args:
        log_file_path: Path to the log file.
        metrics: List of metrics to extract. Defaults to
            ['eval/gt_err', 'eval/gt_err_max'].

    Returns:
        dict: A dictionary containing the extracted metric values.
    """
    if metrics is None:
        metrics = ['eval/gt_err', 'eval/gt_err_max', 'eval/power', 'eval/tracking_success_rate_0.2', 'eval/tracking_success_rate_0.5']
    
    results = {metric: [] for metric in metrics}
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                for metric in metrics:
                    # Match patterns like "eval/gt_err: 0.1996".
                    pattern = rf'{re.escape(metric)}:\s+([\d.]+)'
                    match = re.search(pattern, line)
                    if match:
                        value = float(match.group(1))
                        results[metric].append(value)
    
    except FileNotFoundError:
        print(f"Error: File not found: {log_file_path}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    return results


def print_results(results, show_all=False):
    """
    Print the extracted results.

    Args:
        results: Dictionary of extracted metrics.
        show_all: Whether to show all occurrences. By default, only the last
            occurrence is displayed.
    """
    if not results:
        return
    
    print("\nExtracted metric values:")
    print("=" * 50)
    
    for metric, values in results.items():
        if not values:
            print(f"{metric}: Not found")
        elif show_all:
            print(f"{metric}:")
            for i, val in enumerate(values, 1):
                print(f"  Occurrence {i}: {val}")
        else:
            # Only show the last occurrence.
            print(f"{metric}: {values[-1]}")
    
    print("=" * 50)


def main():
    """Main function."""
    # Default log file path.
    default_log_path = "save/omnicontrol_smpl_fix_eval_debug/eval_0/deepmimic_output/eval_agent.log"

    # If a command-line argument is provided, use it as the log file path.
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = default_log_path

    # Check whether all occurrences should be displayed.
    show_all = '--all' in sys.argv
    
    print(f"Reading log file: {log_path}")
    
    # Extract metrics.
    results = extract_metrics(log_path)
    
    if results:
        print_results(results, show_all=show_all)
        
        # Show a hint if multiple evaluation runs are found.
        max_occurrences = max(len(values) for values in results.values() if values)
        if max_occurrences > 1 and not show_all:
            print(f"\nNote: Found {max_occurrences} evaluation runs in the log")
            print("Use --all to view the results from all runs")


if __name__ == "__main__":
    main()
