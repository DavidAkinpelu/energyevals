#!/usr/bin/env python3
"""Update evaluation summary.csv with failure flags and token breakdown from traces.

Usage:
    python scripts/update_summary_from_traces.py \
        --traces-dir benchmark_traces/with_tools_final/anthropic_claude-sonnet-4-6 \
        --eval-dir evaluation_results/new_evaluations/anthropic_claude-sonnet-4-6
"""

import argparse
import sys
from pathlib import Path

EVAL_MODULE_DIR = Path(__file__).parent.parent / "energyevals" / "evaluation"
sys.path.insert(0, str(EVAL_MODULE_DIR))

from summary_postprocess import update_summary_from_traces


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update evaluation summary.csv with Failed + token columns derived "
            "from benchmark trace files."
        )
    )
    parser.add_argument(
        "--traces-dir",
        required=True,
        type=Path,
        help="Path to model trace folder (contains trace_qN_*.json files).",
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        type=Path,
        help="Path to model evaluation folder containing summary.csv.",
    )
    parser.add_argument(
        "--summary-file",
        default="summary.csv",
        help="Summary CSV filename inside --eval-dir (default: summary.csv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print stats without writing file.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    traces_dir = args.traces_dir
    eval_dir = args.eval_dir
    summary_path = eval_dir / args.summary_file

    try:
        stats = update_summary_from_traces(
            traces_dir=traces_dir,
            eval_dir=eval_dir,
            summary_file=args.summary_file,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"[dry-run] summary path: {summary_path}")
        print(f"[dry-run] rows processed: {stats['rows_processed']}")
        print(f"[dry-run] failed rows: {stats['failed_rows']}")
        print(f"[dry-run] missing traces: {stats['missing_traces']}")
        return 0

    print(f"Updated summary: {summary_path}")
    print(f"Rows processed: {stats['rows_processed']}")
    print(f"Failed rows: {stats['failed_rows']}")
    print(f"Missing traces: {stats['missing_traces']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
