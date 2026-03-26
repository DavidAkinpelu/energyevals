import csv
import json
import re
from pathlib import Path
from typing import Any

from .models import (
    BenchmarkResultEntry,
    CostEstimate,
    GroundTruth,
    LatencyBreakdown,
    MetricScore,
)

_REQUIRED_EVAL_COLUMNS = {"S/N", "Question", "Answer", "Approach", "Category"}

# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_eval_data(csv_path: Path | str) -> list[dict[str, Any]]:
    """Load the evaluation dataset CSV into a list of row dicts.

    Expected columns: S/N, Category, Question type, Difficulty level, Question,
    Answer, Approach.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    if rows:
        missing = _REQUIRED_EVAL_COLUMNS - set(rows[0].keys())
        if missing:
            raise ValueError(
                f"Dataset CSV '{csv_path}' is missing required columns: {sorted(missing)}. "
                f"Use eval_samples_with_answers.csv, not eval_samples.csv."
            )

    return rows


def load_ground_truth(csv_path: Path | str) -> dict[int, GroundTruth]:
    """Extract ground truth per question number from the dataset CSV.

    Returns:
        Mapping of question number (S/N) to GroundTruth.
    """
    rows = load_eval_data(csv_path)
    ground_truths: dict[int, GroundTruth] = {}
    for row in rows:
        qnum = int(row["S/N"])
        ground_truths[qnum] = GroundTruth(
            answer=row.get("Answer", ""),
            approach=row.get("Approach", ""),
            question_type=row.get("Question type", ""),
            category=row.get("Category", ""),
        )
    return ground_truths


# ---------------------------------------------------------------------------
# Trial discovery
# ---------------------------------------------------------------------------

def discover_trials(model_trace_path: Path) -> list[int | None]:
    """Scan *model_trace_path* for ``trial_N/`` subdirectories.

    Returns:
        Sorted list of trial numbers found, or ``[None]`` when no trial
        directories exist (single-trial backward compatibility).
    """
    if not model_trace_path.is_dir():
        return [None]

    trial_dirs: list[int] = []
    for child in model_trace_path.iterdir():
        if child.is_dir():
            match = re.fullmatch(r"trial_(\d+)", child.name)
            if match:
                trial_dirs.append(int(match.group(1)))

    if not trial_dirs:
        return [None]
    return sorted(trial_dirs)


# ---------------------------------------------------------------------------
# Single trace loading
# ---------------------------------------------------------------------------

def _resolve_trace_file(trace_dir: Path, question_num: int) -> Path:
    """Find the trace file for a question number inside *trace_dir*."""
    pattern = f"trace_q{question_num}_*.json"
    matches = list(trace_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No trace file matching '{pattern}' in {trace_dir}"
        )
    if len(matches) > 1:
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _parse_latency_breakdown(steps: list[dict]) -> LatencyBreakdown:
    """Compute latency breakdown from trace steps.

    LLM thinking time is read from ``thought`` steps (one per iteration).
    Tool execution time is read from ``observation`` steps.
    Final answer latency is included in ``llm_thinking_ms``.
    """
    llm_ms = 0.0
    tool_ms = 0.0
    per_tool: dict[str, float] = {}

    for step in steps:
        lat = step.get("latency_ms") or 0.0
        stype = step.get("step_type", "")

        if stype == "thought":
            llm_ms += lat
        elif stype == "observation":
            tool_ms += lat
            tool_name = step.get("tool_name") or "unknown"
            per_tool[tool_name] = per_tool.get(tool_name, 0.0) + lat
        elif stype == "answer":
            llm_ms += lat

    return LatencyBreakdown(
        wall_clock_ms=llm_ms + tool_ms,
        llm_thinking_ms=llm_ms,
        tool_execution_ms=tool_ms,
        per_tool_ms=per_tool,
    )


def _build_steps_trace(steps: list[dict]) -> str:
    """Build a string representation of agent action steps for judge input."""
    actions = [
        {
            "index": step.get("index"),
            "timestamp": step.get("timestamp"),
            "content": step.get("content"),
            "tool_name": step.get("tool_name"),
            "tool_input": step.get("tool_input"),
        }
        for step in steps
        if step.get("step_type") == "action"
    ]
    return str(actions)


def load_benchmark_result(
    trace_base_path: Path,
    question_num: int,
    trial: int | None,
) -> BenchmarkResultEntry:
    """Load one trace file and return a structured result entry.

    Args:
        trace_base_path: Model-level trace directory (e.g.
            ``benchmark_traces/eval_samples/openai_gpt-4.1``).
        question_num: Question number (matches ``trace_q{N}_*.json``).
        trial: Trial number.  When an ``int``, reads from
            ``trial_{trial}/trace_q{N}_*.json``.  When ``None``, reads from
            the flat directory.

    Returns:
        Parsed BenchmarkResultEntry.
    """
    if trial is not None:
        trace_dir = trace_base_path / f"trial_{trial}"
    else:
        trace_dir = trace_base_path

    trace_path = _resolve_trace_file(trace_dir, question_num)

    with open(trace_path, encoding="utf-8") as f:
        data: dict = json.load(f)

    answer = data.get("final_answer")
    steps = data.get("steps", [])
    metrics_raw = data.get("metrics", {})

    latency = _parse_latency_breakdown(steps)
    steps_trace = _build_steps_trace(steps)

    cost = CostEstimate(
        input_tokens=metrics_raw.get("total_input_tokens", 0),
        output_tokens=metrics_raw.get("total_output_tokens", 0),
        cached_tokens=metrics_raw.get("total_cached_tokens", 0),
        reasoning_tokens=metrics_raw.get("total_reasoning_tokens", 0),
    )

    metric_score = MetricScore(
        tool_calls=metrics_raw.get("tool_calls_count", 0),
        iterations=metrics_raw.get("iterations", 0),
        total_tokens=metrics_raw.get("total_tokens", 0),
        duration_seconds=metrics_raw.get("duration_seconds", 0.0),
        latency=latency,
        cost=cost,
    )

    return BenchmarkResultEntry(
        answer=answer,
        steps_trace=steps_trace,
        metrics=metric_score,
    )


def load_benchmark_results(
    trace_base_path: Path,
    question_nums: list[int],
    trial: int | None,
) -> dict[int, BenchmarkResultEntry]:
    """Load trace results for multiple questions in one trial.

    Returns:
        Mapping of question number to BenchmarkResultEntry.
    """
    results: dict[int, BenchmarkResultEntry] = {}
    for qnum in question_nums:
        try:
            results[qnum] = load_benchmark_result(trace_base_path, qnum, trial)
        except FileNotFoundError:
            continue
    return results
