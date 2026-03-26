#!/usr/bin/env python3
"""
Process evaluation summary files and compute per-model aggregate metrics.

Features:
1) Class-balanced (macro across question categories) metrics for:
   - Accuracy (%)
   - Approach
   - Source validity
   - Failure rate (%)
2) Simple per-task averages for:
   - Tokens
   - Tool calls
3) Cost per task based on input/output/cached token usage, with editable
   per-model pricing in this script.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Pricing configuration (EDIT THESE DIRECTLY AS NEEDED)
# Prices are in USD per 1M tokens.
# ---------------------------------------------------------------------------

DEFAULT_PRICING_PER_MTOKEN = {
    "input": 0.0,
    "output": 0.0,
    "cached": 0.0,
}

# Optional model-specific overrides.
# Keys should match model folder names under the results directory.
MODEL_PRICING_PER_MTOKEN: dict[str, dict[str, float]] = {
    # Example:
    "openai_gpt-5.2": {"input": 1.75, "output": 14.0, "cached": 0.175},
    "openai_gpt-5-mini": {"input": 0.25, "output": 2.0, "cached": 0.025},
    "anthropic_claude-sonnet-4-6": {"input": 3, "output": 15.0, "cached": 0.3},
    "anthropic_claude-sonnet-4-6-new": {"input": 3, "output": 15, "cached": 0.3},
    "deepinfra_deepseek-ai_DeepSeek-V3.2": {"input": 0.26, "output": 0.38, "cached": 0.13},
    "deepinfra_moonshotai_Kimi-K2.5": {"input": 0.45, "output": 2.25, "cached": 0.07},
    "deepinfra_Qwen_Qwen3-Max-Thinking": {"input": 1.2, "output": 6.0, "cached": 0.24},
    "google_gemini-3.1-pro-preview": {"input": 2, "output": 12, "cached": 0.2}
}


@dataclass
class RowMetrics:
    question_id: int
    category: str
    approach: float
    accuracy: float
    sources: float
    failed: bool
    tokens: float
    tool_calls: float
    iterations: float
    input_tokens: float
    output_tokens: float
    cached_tokens: float


def _to_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _to_bool(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _read_summary_rows(summary_path: Path) -> list[RowMetrics]:
    rows: list[RowMetrics] = []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            qid_raw = (r.get("question_id") or "").strip()
            if not qid_raw:
                continue
            try:
                question_id = int(float(qid_raw))
            except ValueError:
                continue
            category = (r.get("category") or "").strip()
            rows.append(
                RowMetrics(
                    question_id=question_id,
                    category=category,
                    approach=_to_float(r.get("approach_mean")),
                    accuracy=_to_float(r.get("accuracy_mean")),
                    sources=_to_float(r.get("sources_mean")),
                    failed=_to_bool(r.get("Failed")),
                    tokens=_to_float(r.get("tokens")),
                    tool_calls=_to_float(r.get("tool_calls")),
                    iterations=_to_float(r.get("iterations")),
                    input_tokens=_to_float(
                        r.get("total_input_tokens") or r.get("input_tokens")
                    ),
                    output_tokens=_to_float(
                        r.get("total_output_tokens") or r.get("output_tokens")
                    ),
                    cached_tokens=_to_float(
                        r.get("total_cached_tokens") or r.get("cached_tokens")
                    ),
                )
            )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _class_balanced(rows: list[RowMetrics]) -> tuple[float, float, float, float, int]:
    # Macro-average across non-empty categories.
    by_cat: dict[str, list[RowMetrics]] = {}
    for row in rows:
        if not row.category:
            continue
        by_cat.setdefault(row.category, []).append(row)

    if not by_cat:
        return 0.0, 0.0, 0.0, 0.0, 0

    cat_approach = []
    cat_accuracy = []
    cat_sources = []
    cat_failure = []

    for cat_rows in by_cat.values():
        cat_approach.append(_mean([r.approach for r in cat_rows]))
        cat_accuracy.append(_mean([r.accuracy for r in cat_rows]))
        cat_sources.append(_mean([r.sources for r in cat_rows]))
        cat_failure.append(_mean([1.0 if r.failed else 0.0 for r in cat_rows]))

    return (
        _mean(cat_approach),
        _mean(cat_accuracy),
        _mean(cat_sources),
        _mean(cat_failure),
        len(by_cat),
    )


def _pricing_for_model(model_name: str) -> dict[str, float]:
    price = dict(DEFAULT_PRICING_PER_MTOKEN)
    price.update(MODEL_PRICING_PER_MTOKEN.get(model_name, {}))
    return price


def _cost_usd(row: RowMetrics, pricing_per_mtoken: dict[str, float]) -> float:
    # Input token counts include cached tokens. Bill non-cached input separately
    # so cached tokens are not charged twice.
    billable_input_tokens = max(row.input_tokens - row.cached_tokens, 0.0)
    return (
        (billable_input_tokens / 1_000_000.0) * pricing_per_mtoken["input"]
        + (row.output_tokens / 1_000_000.0) * pricing_per_mtoken["output"]
        + (row.cached_tokens / 1_000_000.0) * pricing_per_mtoken["cached"]
    )


def process_results(
    results_dir: Path,
    excludes: set[str],
    question_ids: set[int] | None = None,
) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []

    model_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    for model_dir in model_dirs:
        if model_dir.name in excludes:
            continue

        summary_path = model_dir / "summary.csv"
        if not summary_path.exists():
            continue

        rows = _read_summary_rows(summary_path)
        if question_ids is not None:
            rows = [r for r in rows if r.question_id in question_ids]
        if not rows:
            continue

        class_bal_approach, class_bal_accuracy, class_bal_sources, class_bal_failure, n_categories = _class_balanced(rows)

        avg_tokens = _mean([r.tokens for r in rows])
        avg_tool_calls = _mean([r.tool_calls for r in rows])

        pricing = _pricing_for_model(model_dir.name)
        per_task_costs = [_cost_usd(r, pricing) for r in rows]
        total_cost = sum(per_task_costs)
        # Explicit simple average over tasks (not class-balanced).
        avg_cost_per_task = total_cost / len(rows)

        output_rows.append(
            {
                "model": model_dir.name,
                "tasks": str(len(rows)),
                "categories": str(n_categories),
                "class_bal_accuracy_pct": f"{class_bal_accuracy * 100.0:.4f}",
                "class_bal_approach": f"{class_bal_approach:.4f}",
                "class_bal_source_validity": f"{class_bal_sources:.4f}",
                "class_bal_failure_rate_pct": f"{class_bal_failure * 100.0:.4f}",
                "avg_tokens_per_task": f"{avg_tokens:.2f}",
                "avg_tool_calls_per_task": f"{avg_tool_calls:.2f}",
                "avg_cost_per_task_usd": f"{avg_cost_per_task:.6f}",
                "total_cost_usd": f"{total_cost:.6f}",
                "pricing_input_per_mtoken": f"{pricing['input']:.6f}",
                "pricing_output_per_mtoken": f"{pricing['output']:.6f}",
                "pricing_cached_per_mtoken": f"{pricing['cached']:.6f}",
            }
        )

    return output_rows


def _print_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No model summary rows found.")
        return

    headers = list(rows[0].keys())
    widths = {h: max(len(h), *(len(r[h]) for r in rows)) for h in headers}

    def line(vals: list[str]) -> str:
        return " | ".join(v.ljust(widths[h]) for v, h in zip(vals, headers))

    print(line(headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for r in rows:
        print(line([r[h] for h in headers]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Process evaluation summary.csv files across model folders.")
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path containing model subfolders with summary.csv files.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Model folder names to exclude (space-separated).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for aggregated model metrics.",
    )
    parser.add_argument(
        "--question-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of question IDs to include (space-separated).",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {args.results_dir}")

    qids = set(args.question_ids) if args.question_ids else None
    rows = process_results(args.results_dir, set(args.exclude), question_ids=qids)
    _print_table(rows)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            with args.out_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            with args.out_csv.open("w", encoding="utf-8", newline="") as f:
                f.write("")
        print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
