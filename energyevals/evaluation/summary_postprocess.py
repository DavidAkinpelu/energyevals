from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


CLARIFICATION_PATTERNS = [
    r"\bcan you clarify\b",
    r"\bcould you clarify\b",
    r"\bplease clarify\b",
    r"\bcan you provide more (details|information|context)\b",
    r"\bcould you provide more (details|information|context)\b",
    r"\bplease provide (more )?(details|information|context)\b",
    r"\bdo you mean\b",
    r"\bwhat (exactly|specifically) do you\b",
    r"\bcould you specify\b",
    r"\bplease specify\b",
    r"\bneed (more|additional) (details|information|context)\b",
    r"\bi need (more|additional) (details|information|context)\b",
]

QUESTION_PROMPT_PATTERNS = [
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bwould you\b",
    r"\bdo you\b",
    r"\bplease\b",
    r"\bwhich (one|option|region|market|dataset)\b",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at root in {path}")
    return data


def _find_trace_file(traces_dir: Path, question_id: int) -> Path | None:
    pattern = f"trace_q{question_id}_*.json"
    matches = list(traces_dir.glob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _looks_like_clarification_request(final_answer: str | None) -> bool:
    if not final_answer:
        return True

    text = final_answer.strip()
    if not text:
        return True

    lowered = text.lower()
    for pat in CLARIFICATION_PATTERNS:
        if re.search(pat, lowered):
            return True

    # Treat as clarification only when the final response itself is a
    # user-directed question, not when question-like phrases appear in prose.
    sentence_count = max(1, len(re.findall(r"[.!?]+", text)))
    word_count = len(re.findall(r"\S+", text))
    asks_question = text.endswith("?")
    if not asks_question:
        return False

    if any(re.search(pat, lowered) for pat in QUESTION_PROMPT_PATTERNS):
        return word_count <= 120 and sentence_count <= 5

    return word_count <= 25 and sentence_count <= 2


def _trace_metrics(trace_data: dict[str, Any]) -> tuple[int, int, int, int]:
    metrics = trace_data.get("metrics")
    if not isinstance(metrics, dict):
        return 0, 0, 0, 0
    return (
        int(metrics.get("total_input_tokens", 0) or 0),
        int(metrics.get("total_output_tokens", 0) or 0),
        int(metrics.get("total_cached_tokens", 0) or 0),
        int(metrics.get("iterations", 0) or 0),
    )


def _trace_failed(trace_data: dict[str, Any]) -> bool:
    has_error = trace_data.get("error") is not None
    clarification = _looks_like_clarification_request(trace_data.get("final_answer"))
    return has_error or clarification


def _parse_question_id(row: dict[str, str]) -> int:
    raw = (row.get("question_id") or "").strip()
    if not raw:
        raise ValueError("Row has empty question_id")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid question_id value: '{raw}'") from exc


def _read_summary(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ValueError(f"CSV appears empty or missing header: {path}")
    if "question_id" not in fieldnames:
        raise ValueError(f"summary.csv missing required column 'question_id': {path}")
    return rows, fieldnames


def _write_summary(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_summary_from_traces(
    traces_dir: Path,
    eval_dir: Path,
    *,
    summary_file: str = "summary.csv",
    dry_run: bool = False,
) -> dict[str, int]:
    """Update per-question summary.csv with failure and token columns from traces."""
    summary_path = eval_dir / summary_file
    if not traces_dir.is_dir():
        raise FileNotFoundError(f"Trace dir not found: {traces_dir}")
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    rows, fieldnames = _read_summary(summary_path)
    extra_cols = [
        "Failed",
        "iterations",
        "total_input_tokens",
        "total_output_tokens",
        "total_cached_tokens",
    ]
    for col in extra_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    updated = 0
    missing_traces = 0
    failed_count = 0

    for row in rows:
        qid = _parse_question_id(row)
        trace_file = _find_trace_file(traces_dir, qid)
        if trace_file is None:
            row["Failed"] = "True"
            row["iterations"] = ""
            row["total_input_tokens"] = ""
            row["total_output_tokens"] = ""
            row["total_cached_tokens"] = ""
            missing_traces += 1
            failed_count += 1
            updated += 1
            continue

        trace_data = _load_json(trace_file)
        failed = _trace_failed(trace_data)
        input_tok, output_tok, cached_tok, iterations = _trace_metrics(trace_data)

        row["Failed"] = "True" if failed else "False"
        row["iterations"] = str(iterations)
        row["total_input_tokens"] = str(input_tok)
        row["total_output_tokens"] = str(output_tok)
        row["total_cached_tokens"] = str(cached_tok)

        if failed:
            failed_count += 1
        updated += 1

    if not dry_run:
        _write_summary(summary_path, rows, fieldnames)

    return {
        "rows_processed": updated,
        "failed_rows": failed_count,
        "missing_traces": missing_traces,
    }
