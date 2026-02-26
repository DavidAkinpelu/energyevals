import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st

TRACES_ROOT = Path("benchmark_traces")
FILENAME_RE = re.compile(
    r"trace_q(\d+)_(\d{8}_\d{6})_([a-f0-9]{8})\.json$"
)
TOOL_OUTPUT_MAX_CHARS = 8_000


@dataclass
class TraceRef:
    path: Path
    run_name: str
    model_key: str
    trial: int | None
    question_id: int
    timestamp_str: str
    uuid8: str
    success: bool | None = None
    duration_seconds: float | None = None
    category: str | None = None
    difficulty: str | None = None
    error: str | None = None
    total_tokens: int | None = None


# ---------------------------------------------------------------------------
# Scanning / loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def list_runs(traces_root: str = str(TRACES_ROOT)) -> list[str]:
    root = Path(traces_root)
    if not root.exists():
        return []
    return sorted(
        d.name for d in root.iterdir() if d.is_dir()
    )


@st.cache_data(ttl=300)
def load_run_index(
    run_name: str,
    traces_root: str = str(TRACES_ROOT),
) -> tuple[list[TraceRef], int]:
    """Return (trace_refs, corrupt_count) for all traces in run_name."""
    root = Path(traces_root) / run_name
    refs: list[TraceRef] = []
    corrupt = 0

    if not root.exists():
        return refs, corrupt

    # Walk model_key dirs
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_key = model_dir.name

        # Collect (path, trial) tuples to process
        file_pairs: list[tuple[Path, int | None]] = []

        for item in sorted(model_dir.iterdir()):
            if item.is_dir() and item.name.startswith("trial_"):
                try:
                    trial_num = int(item.name.split("_", 1)[1])
                except ValueError:
                    trial_num = None
                for f in sorted(item.iterdir()):
                    if f.suffix == ".json":
                        file_pairs.append((f, trial_num))
            elif item.is_file() and item.suffix == ".json":
                file_pairs.append((item, None))

        for fpath, trial in file_pairs:
            m = FILENAME_RE.search(fpath.name)
            if not m:
                continue
            q_id = int(m.group(1))
            ts_str = m.group(2)
            uuid8 = m.group(3)

            try:
                with fpath.open() as fh:
                    data = json.load(fh)
                meta = data.get("metadata") or {}
                metrics = data.get("metrics") or {}
                ref = TraceRef(
                    path=fpath,
                    run_name=run_name,
                    model_key=model_key,
                    trial=trial,
                    question_id=q_id,
                    timestamp_str=ts_str,
                    uuid8=uuid8,
                    success=data.get("success"),
                    duration_seconds=metrics.get("duration_seconds"),
                    category=meta.get("category"),
                    difficulty=meta.get("difficulty"),
                    error=data.get("error"),
                    total_tokens=metrics.get("total_tokens"),
                )
                refs.append(ref)
            except Exception:
                corrupt += 1

    return refs, corrupt


def load_trace(ref: TraceRef) -> dict[str, Any] | None:
    """Load the full trace JSON. No cache (called on demand)."""
    try:
        with ref.path.open() as fh:
            return json.load(fh)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def get_model_keys(index: list[TraceRef]) -> list[str]:
    return sorted({r.model_key for r in index})


def get_trials(index: list[TraceRef], model_key: str) -> list[int]:
    trials = {r.trial for r in index if r.model_key == model_key and r.trial is not None}
    return sorted(trials)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_traces(
    index: list[TraceRef],
    model_keys: list[str] | None = None,
    trials: list[int] | None = None,
    difficulties: list[str] | None = None,
    categories: list[str] | None = None,
    q_id_search: str = "",
    success_only: bool = False,
    failed_only: bool = False,
) -> list[TraceRef]:
    out = index
    if model_keys:
        out = [r for r in out if r.model_key in model_keys]
    if trials:
        out = [r for r in out if r.trial in trials]
    if difficulties:
        out = [r for r in out if r.difficulty in difficulties]
    if categories:
        out = [r for r in out if r.category in categories]
    if q_id_search.strip():
        try:
            qid = int(q_id_search.strip())
            out = [r for r in out if r.question_id == qid]
        except ValueError:
            pass
    if success_only:
        out = [r for r in out if r.success is True]
    if failed_only:
        out = [r for r in out if r.success is False]
    return out


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_run_stats(index: list[TraceRef]) -> dict[str, Any]:
    total = len(index)
    passed = sum(1 for r in index if r.success is True)
    failed = sum(1 for r in index if r.success is False)
    unknown = total - passed - failed

    durations = [r.duration_seconds for r in index if r.duration_seconds is not None]
    tokens = [r.total_tokens for r in index if r.total_tokens is not None]
    success_rate = (passed / total * 100) if total else 0.0
    avg_duration = (sum(durations) / len(durations)) if durations else None
    avg_tokens = (sum(tokens) / len(tokens)) if tokens else None

    def breakdown(key: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[TraceRef]] = {}
        for r in index:
            val = getattr(r, key) or "Unknown"
            groups.setdefault(val, []).append(r)
        result = {}
        for val, refs in sorted(groups.items()):
            n = len(refs)
            p = sum(1 for r in refs if r.success is True)
            f = sum(1 for r in refs if r.success is False)
            durs = [r.duration_seconds for r in refs if r.duration_seconds is not None]
            toks = [r.total_tokens for r in refs if r.total_tokens is not None]
            result[val] = {
                "total": n,
                "passed": p,
                "failed": f,
                "success_rate": (p / n * 100) if n else 0.0,
                "avg_duration": (sum(durs) / len(durs)) if durs else None,
                "avg_tokens": (sum(toks) / len(toks)) if toks else None,
            }
        return result

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "unknown": unknown,
        "success_rate": success_rate,
        "avg_duration": avg_duration,
        "avg_tokens": avg_tokens,
        "by_model": breakdown("model_key"),
        "by_difficulty": breakdown("difficulty"),
        "by_category": breakdown("category"),
        "durations": durations,
        "tokens": tokens,
    }


# ---------------------------------------------------------------------------
# Step timeline enrichment
# ---------------------------------------------------------------------------

def compute_step_timeline(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich raw step dicts with display helpers."""
    if not steps:
        return []

    first_ts = steps[0].get("timestamp_unix")
    enriched = []
    for step in steps:
        s = dict(step)
        step_type = s.get("step_type", "")
        tool_name = s.get("tool_name") or ""

        # Display label
        if step_type == "action" and tool_name:
            s["display_label"] = f"action → {tool_name}"
        elif step_type == "observation" and tool_name:
            s["display_label"] = f"observation ← {tool_name}"
        else:
            s["display_label"] = step_type

        # Relative time
        ts_unix = s.get("timestamp_unix")
        if ts_unix is not None and first_ts is not None:
            s["relative_time_s"] = round(ts_unix - first_ts, 3)
        else:
            s["relative_time_s"] = None

        # Tool output truncation
        raw_output = s.get("tool_output") or s.get("content") or ""
        if step_type == "observation":
            raw_output = s.get("tool_output") or ""
        s["tool_output_full_len"] = len(raw_output)
        s["tool_output_truncated"] = raw_output[:TOOL_OUTPUT_MAX_CHARS]
        s["tool_output_is_truncated"] = len(raw_output) > TOOL_OUTPUT_MAX_CHARS

        # Try to parse as JSON
        to_parse = s.get("tool_output") or ""
        if step_type == "observation" and to_parse:
            try:
                s["tool_output_parsed"] = json.loads(to_parse)
                s["tool_output_is_json"] = True
            except Exception:
                s["tool_output_parsed"] = None
                s["tool_output_is_json"] = False
        else:
            s["tool_output_parsed"] = None
            s["tool_output_is_json"] = False

        enriched.append(s)

    return enriched
