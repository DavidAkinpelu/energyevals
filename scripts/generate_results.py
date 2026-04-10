#!/usr/bin/env python3
"""
scripts/generate_results.py

Auto-discover models from trace/eval directories, compute per-model,
per-category, and per-difficulty aggregates, and output publication-quality
PDF figures plus LaTeX table snippets for the paper.

Usage:
    .venv/bin/python scripts/generate_results.py \
        --trace-dir benchmark_traces \
        --eval-dir  evaluation_results \
        --out-dir   latex/figures \
        --tables-out latex/tables.tex

Output:
    latex/figures/fig_main_results.pdf
    latex/figures/fig_with_vs_without.pdf
    latex/figures/fig_by_category.pdf
    latex/figures/fig_by_difficulty.pdf
    latex/figures/fig_tool_usage.pdf
    latex/figures/fig_efficiency.pdf
    latex/tables.tex
    results/summary.csv

Directory structure expected
────────────────────────────
benchmark_traces/
  {condition}/
    {model_key}/
      trace_q{N}_{ts}_{uuid}.json   (metadata.question_id, .category, .difficulty)

evaluation_results/
  {condition}/
    {model_key}/
      q{N}.json                     (approach/accuracy/sources scores)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

DISPLAY_NAMES: dict[str, str] = {
    "anthropic_claude-sonnet-4-6": "Claude Sonnet 4.6",
    "openai_gpt-5.2": "GPT-5.2",
    "openai_gpt-5-mini": "GPT-5-mini",
    "deepinfra_Qwen_Qwen3-Max-Thinking": "Qwen3-Max",
    "moonshot_kimi-k2-5": "Kimi-K2.5",
    "deepseek_deepseek-v3-2": "DeepSeek V3.2",
}

# 6-colour colorblind-safe palette (Wong 2011).
COLORS: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # teal
    "#CC79A7",  # mauve
    "#E69F00",  # amber
    "#56B4E9",  # sky blue
]

CONDITION_COLORS: dict[str, str] = {
    "with_tools": "#0072B2",
    "without_tools": "#D55E00",
}

CONDITION_LABELS: dict[str, str] = {
    "with_tools": "With Tools",
    "without_tools": "Without Tools",
}

CATEGORY_SHORT: dict[str, str] = {
    "Knowledge retrieval and interpretation": "Knowledge\nRetrieval",
    "Markets data retrieval and analysis": "Markets\nAnalysis",
    "Markets data retrieval only": "Markets\nRetrieval",
    "Advanced quantitative modeling and decision analytics": "Adv.\nModeling",
}

DIFFICULTIES: list[str] = ["Easy", "Medium", "Hard"]

# (df_column, y-axis label, (y_min, y_max))
DIMS: list[tuple[str, str, tuple[float, float]]] = [
    ("approach_mean", "Approach Correctness (1–5)", (0.0, 5.5)),
    ("accuracy_mean", "Answer Accuracy (0–1)", (0.0, 1.1)),
    ("sources_mean", "Source Validity (1–5)", (0.0, 5.5)),
]


# ── Display name helper ────────────────────────────────────────────────────────

def make_display_name(model_key: str) -> str:
    if model_key in DISPLAY_NAMES:
        return DISPLAY_NAMES[model_key]
    parts = model_key.split("_", 1)
    name = parts[1] if len(parts) == 2 else model_key
    return name.replace("_", " ").replace("-", " ").title()


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_models(base_dir: Path, condition: str) -> list[str]:
    """Return sorted model_key directory names present under base_dir/condition."""
    d = base_dir / condition
    if not d.exists():
        return []
    return sorted(
        p.name for p in d.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_traces_by_qid(
    trace_dir: Path, condition: str, model_key: str
) -> dict[int, dict]:
    """
    Glob all trace_q*.json files and return a dict keyed by question_id.
    question_id is read from metadata.question_id (preferred) or the filename.
    """
    model_dir = trace_dir / condition / model_key
    if not model_dir.exists():
        return {}

    by_qid: dict[int, dict] = {}
    for f in sorted(model_dir.rglob("trace_q*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception as exc:
            print(f"  [warn] Failed to load {f}: {exc}", file=sys.stderr)
            continue

        # Prefer metadata field; fall back to filename digit after 'trace_q'
        qid = data.get("metadata", {}).get("question_id")
        if qid is None:
            stem = f.stem  # trace_q{N}_{ts}_{uuid}
            try:
                qid = int(stem.split("_")[1][1:])  # strip leading 'q'
            except (IndexError, ValueError):
                print(f"  [warn] Cannot determine question_id for {f}", file=sys.stderr)
                continue

        by_qid[int(qid)] = data

    return by_qid


def load_eval_by_qid(
    eval_dir: Path, condition: str, model_key: str
) -> dict[int, dict]:
    """
    Load all q{N}.json eval files and return a dict keyed by question_id (N).
    """
    model_dir = eval_dir / condition / model_key
    if not model_dir.exists():
        return {}

    by_qid: dict[int, dict] = {}
    for f in model_dir.glob("q*.json"):
        try:
            qid = int(f.stem[1:])  # 'q123' → 123
            by_qid[qid] = json.loads(f.read_text())
        except Exception as exc:
            print(f"  [warn] Failed to load {f}: {exc}", file=sys.stderr)

    return by_qid


def join_question_data(
    eval_by_qid: dict[int, dict],
    traces_by_qid: dict[int, dict],
) -> list[dict]:
    """
    Join eval scores and trace execution data by question_id.

    - Scores (approach/accuracy/sources) come from eval files.
    - category, difficulty, success, iterations come from trace files.
    - Questions present only in eval (no trace) still appear; category/difficulty
      will be 'Unknown' and execution metrics will be NaN.
    - Questions present only in traces (not yet evaluated) are silently skipped.
    """
    nan = float("nan")
    records: list[dict] = []

    for qid in sorted(eval_by_qid.keys()):
        eq = eval_by_qid[qid]
        tr = traces_by_qid.get(qid, {})

        tr_meta = tr.get("metadata", {})
        tr_metrics = tr.get("metrics", {})
        eq_metrics = eq.get("metrics", {})

        records.append(
            {
                "question_id": qid,
                "category": tr_meta.get("category", "Unknown"),
                "difficulty": tr_meta.get("difficulty", "Unknown"),
                # Scores from eval file
                "approach": eq.get("approach", {}).get("score", nan),
                "accuracy": eq.get("accuracy", {}).get("score", nan),
                "sources": eq.get("sources", {}).get("score", nan),
                # Execution metrics: prefer trace, fall back to eval metrics
                "iterations": tr_metrics.get("iterations", nan),
                "total_tokens": tr_metrics.get(
                    "total_tokens", eq_metrics.get("total_tokens", nan)
                ),
                "tool_calls_count": tr_metrics.get(
                    "tool_calls_count", eq_metrics.get("tool_calls", nan)
                ),
                "success": tr.get("success", False),
                # Tool names for frequency chart
                "tool_names": tr.get("step_summary", {}).get("tool_calls", []),
            }
        )

    return records


# ── Matched aggregation (shared question IDs across conditions) ───────────────

def build_matched_agg_df(
    all_records: dict[tuple[str, str], list[dict]],
    conditions: list[str],
) -> pd.DataFrame:
    """
    For the with-vs-without comparison only: restrict both conditions to the
    intersection of question IDs that were evaluated in *all* conditions for
    each model.  This ensures apples-to-apples averages when one condition
    has fewer questions than the other.

    Returns a DataFrame in the same shape as agg_df.
    """
    all_model_keys = sorted({mk for (_, mk) in all_records})
    rows: list[dict] = []

    for mk in all_model_keys:
        # Collect qid sets per condition for this model
        qid_sets: dict[str, set[int]] = {}
        for cond in conditions:
            recs = all_records.get((cond, mk))
            if recs:
                qid_sets[cond] = {r["question_id"] for r in recs}

        if len(qid_sets) < 2:
            # Only one condition has data — nothing to match
            continue

        common_qids = set.intersection(*qid_sets.values())
        if not common_qids:
            continue

        for cond in conditions:
            recs = all_records.get((cond, mk))
            if not recs:
                continue
            subset = [r for r in recs if r["question_id"] in common_qids]
            if subset:
                agg = build_model_agg(subset, mk, cond)
                rows.append(vars(agg))

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Per-model aggregation ─────────────────────────────────────────────────────

@dataclass
class ModelAgg:
    model_key: str
    display_name: str
    condition: str
    n_questions: int
    approach_mean: float
    accuracy_mean: float
    sources_mean: float
    avg_tool_calls: float
    avg_tokens: float
    avg_iterations: float
    success_rate: float


def _nanmean(vals: list[float]) -> float:
    arr = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(arr)) if arr else float("nan")


def build_model_agg(
    records: list[dict],
    model_key: str,
    condition: str,
) -> ModelAgg:
    nan = float("nan")
    if not records:
        return ModelAgg(
            model_key=model_key,
            display_name=make_display_name(model_key),
            condition=condition,
            n_questions=0,
            approach_mean=nan, accuracy_mean=nan, sources_mean=nan,
            avg_tool_calls=nan, avg_tokens=nan,
            avg_iterations=nan, success_rate=nan,
        )

    return ModelAgg(
        model_key=model_key,
        display_name=make_display_name(model_key),
        condition=condition,
        n_questions=len(records),
        approach_mean=_nanmean([r["approach"] for r in records]),
        accuracy_mean=_nanmean([r["accuracy"] for r in records]),
        sources_mean=_nanmean([r["sources"] for r in records]),
        avg_tool_calls=_nanmean([r["tool_calls_count"] for r in records]),
        avg_tokens=_nanmean([r["total_tokens"] for r in records]),
        avg_iterations=_nanmean([r["iterations"] for r in records]),
        success_rate=_nanmean([1.0 if r["success"] else 0.0 for r in records]),
    )


def compute_tool_usage(records: list[dict]) -> pd.Series:
    """Count total calls per tool name across all question records."""
    all_tools: list[str] = []
    for r in records:
        all_tools.extend(r.get("tool_names", []))
    if not all_tools:
        return pd.Series(dtype=int)
    return pd.Series(all_tools).value_counts()


# ── Per-category / per-difficulty breakdowns ──────────────────────────────────

def build_category_df(
    records: list[dict],
    model_key: str,
    display_name: str,
    condition: str,
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    rows = [
        {
            "model_key": model_key,
            "display_name": display_name,
            "condition": condition,
            "category": r["category"],
            "approach": r["approach"],
            "accuracy": r["accuracy"],
            "sources": r["sources"],
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    return (
        df.groupby(["model_key", "display_name", "condition", "category"])
        .agg(
            approach_mean=("approach", "mean"),
            accuracy_mean=("accuracy", "mean"),
            sources_mean=("sources", "mean"),
            n=("approach", "count"),
        )
        .reset_index()
    )


def build_difficulty_df(
    records: list[dict],
    model_key: str,
    display_name: str,
    condition: str,
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    rows = [
        {
            "model_key": model_key,
            "display_name": display_name,
            "condition": condition,
            "difficulty": r["difficulty"],
            "approach": r["approach"],
            "accuracy": r["accuracy"],
            "sources": r["sources"],
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    return (
        df.groupby(["model_key", "display_name", "condition", "difficulty"])
        .agg(
            approach_mean=("approach", "mean"),
            accuracy_mean=("accuracy", "mean"),
            sources_mean=("sources", "mean"),
            n=("approach", "count"),
        )
        .reset_index()
    )


# ── Matplotlib setup ──────────────────────────────────────────────────────────

def setup_mpl() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi, format="pdf")
    plt.close(fig)
    print(f"  Saved: {path}")


def _val_label(
    ax: plt.Axes,
    bar: mpl.patches.Rectangle,
    val: float,
    ylim_max: float,
) -> None:
    if math.isnan(val):
        return
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02 * ylim_max,
        f"{val:.2f}",
        ha="center", va="bottom", fontsize=6.5,
    )


# ── Figure 1: Main results (with-tools) ──────────────────────────────────────

def plot_main_results(agg_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    df = agg_df[agg_df["condition"] == "with_tools"].copy()
    if df.empty:
        print("  [skip] No with_tools data for main results plot.")
        return

    models = df["display_name"].tolist()
    n = len(models)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.suptitle("Main Results — With Tools", fontsize=11, y=1.02)

    for ax, (col, label, (ylo, yhi)) in zip(axes, DIMS):
        colors = [COLORS[i % len(COLORS)] for i in range(n)]
        bars = ax.bar(x, df[col].values, color=colors,
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=7)
        ax.set_ylabel(label, fontsize=8)
        ax.set_ylim(ylo, yhi)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, df[col].values):
            _val_label(ax, bar, val, yhi)

    fig.tight_layout()
    _save(fig, out_dir / "fig_main_results.pdf", dpi)


# ── Figure 2: With vs. without tools ─────────────────────────────────────────

def plot_with_vs_without(agg_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    conditions_present = [
        c for c in ["with_tools", "without_tools"]
        if c in agg_df["condition"].values
    ]
    if len(conditions_present) < 2:
        print("  [skip] Both conditions required for with_vs_without plot.")
        return

    all_keys = sorted(agg_df["model_key"].unique())
    all_display = [
        agg_df[agg_df["model_key"] == mk]["display_name"].iloc[0]
        for mk in all_keys
    ]
    n = len(all_keys)
    x = np.arange(n)
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.suptitle("With Tools vs. Without Tools", fontsize=11, y=1.02)

    for ax, (col, label, (ylo, yhi)) in zip(axes, DIMS):
        for j, cond in enumerate(["with_tools", "without_tools"]):
            cdf = agg_df[agg_df["condition"] == cond].set_index("model_key")
            vals = [
                cdf.loc[mk, col] if mk in cdf.index else float("nan")
                for mk in all_keys
            ]
            offset = (j - 0.5) * width
            ax.bar(
                x + offset, vals, width=width,
                color=CONDITION_COLORS[cond],
                label=CONDITION_LABELS[cond],
                edgecolor="white", linewidth=0.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(all_display, rotation=20, ha="right", fontsize=7)
        ax.set_ylabel(label, fontsize=8)
        ax.set_ylim(ylo, yhi)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    seen: dict[str, mpl.patches.Patch] = {}
    for h, lbl in zip(handles, labels):
        seen.setdefault(lbl, h)
    fig.legend(
        list(seen.values()), list(seen.keys()),
        loc="upper right", bbox_to_anchor=(1.01, 1.0), fontsize=8,
    )
    fig.tight_layout()
    _save(fig, out_dir / "fig_with_vs_without.pdf", dpi)


# ── Figure 3: By category ─────────────────────────────────────────────────────

def plot_by_category(
    cat_df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    condition: str = "with_tools",
) -> None:
    df = cat_df[cat_df["condition"] == condition].copy()
    if df.empty:
        print(f"  [skip] No {condition} data for category plot.")
        return

    categories = sorted(df["category"].unique())
    all_keys = sorted(df["model_key"].unique())
    display_map = (
        df.drop_duplicates("model_key")
        .set_index("model_key")["display_name"]
        .to_dict()
    )
    n_cats = len(categories)
    n_models = len(all_keys)
    x = np.arange(n_cats)
    width = 0.8 / max(n_models, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    cond_label = CONDITION_LABELS.get(condition, condition)
    fig.suptitle(f"Scores by Question Category ({cond_label})", fontsize=11, y=1.02)

    for ax, (col, label, (ylo, yhi)) in zip(axes, DIMS):
        for j, mk in enumerate(all_keys):
            mdf = df[df["model_key"] == mk].set_index("category")
            vals = [
                mdf.loc[c, col] if c in mdf.index else float("nan")
                for c in categories
            ]
            offset = (j - n_models / 2 + 0.5) * width
            ax.bar(
                x + offset, vals, width=width * 0.9,
                color=COLORS[j % len(COLORS)],
                label=display_map.get(mk, mk),
                edgecolor="white", linewidth=0.3,
            )
        cat_labels = [CATEGORY_SHORT.get(c, c) for c in categories]
        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, fontsize=7)
        ax.set_ylabel(label, fontsize=8)
        ax.set_ylim(ylo, yhi)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right",
               bbox_to_anchor=(1.01, 1.0), fontsize=7)
    fig.tight_layout()
    _save(fig, out_dir / "fig_by_category.pdf", dpi)


# ── Figure 4: By difficulty ───────────────────────────────────────────────────

def plot_by_difficulty(
    diff_df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    condition: str = "with_tools",
) -> None:
    df = diff_df[diff_df["condition"] == condition].copy()
    if df.empty:
        print(f"  [skip] No {condition} data for difficulty plot.")
        return

    difficulties = [d for d in DIFFICULTIES if d in df["difficulty"].unique()]
    all_keys = sorted(df["model_key"].unique())
    display_map = (
        df.drop_duplicates("model_key")
        .set_index("model_key")["display_name"]
        .to_dict()
    )
    n_diffs = len(difficulties)
    n_models = len(all_keys)
    x = np.arange(n_diffs)
    width = 0.8 / max(n_models, 1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    cond_label = CONDITION_LABELS.get(condition, condition)
    fig.suptitle(f"Scores by Question Difficulty ({cond_label})", fontsize=11, y=1.02)

    for ax, (col, label, (ylo, yhi)) in zip(axes, DIMS):
        for j, mk in enumerate(all_keys):
            mdf = df[df["model_key"] == mk].set_index("difficulty")
            vals = [
                mdf.loc[d, col] if d in mdf.index else float("nan")
                for d in difficulties
            ]
            offset = (j - n_models / 2 + 0.5) * width
            ax.bar(
                x + offset, vals, width=width * 0.9,
                color=COLORS[j % len(COLORS)],
                label=display_map.get(mk, mk),
                edgecolor="white", linewidth=0.3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(difficulties, fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_ylim(ylo, yhi)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right",
               bbox_to_anchor=(1.01, 1.0), fontsize=7)
    fig.tight_layout()
    _save(fig, out_dir / "fig_by_difficulty.pdf", dpi)


# ── Figure 5: Tool usage ──────────────────────────────────────────────────────

def plot_tool_usage(tool_series: pd.Series, out_dir: Path, dpi: int) -> None:
    if tool_series.empty:
        print("  [skip] No tool usage data.")
        return

    ts = tool_series.sort_values(ascending=True)
    n = len(ts)
    fig, ax = plt.subplots(figsize=(7, max(3.0, n * 0.38)))

    y = np.arange(n)
    ax.barh(y, ts.values, color=COLORS[0], edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(ts.index.tolist(), fontsize=8)
    ax.set_xlabel("Total Tool Calls (all models, with-tools condition)", fontsize=9)
    ax.set_title("Tool Usage Frequency", fontsize=10)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    x_max = float(ts.values.max())
    for i, val in enumerate(ts.values):
        ax.text(val + 0.01 * x_max, i, f"{int(val):,}", va="center", fontsize=7)

    ax.set_xlim(0, x_max * 1.12)
    fig.tight_layout()
    _save(fig, out_dir / "fig_tool_usage.pdf", dpi)


# ── Figure 6: Efficiency ──────────────────────────────────────────────────────

def plot_efficiency(agg_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    conditions_present = [
        c for c in ["with_tools", "without_tools"]
        if c in agg_df["condition"].values
    ]
    if not conditions_present:
        print("  [skip] No data for efficiency plot.")
        return

    all_keys = sorted(agg_df["model_key"].unique())
    all_display = [
        agg_df[agg_df["model_key"] == mk]["display_name"].iloc[0]
        for mk in all_keys
    ]
    n = len(all_keys)
    x = np.arange(n)
    width = 0.8 / max(len(conditions_present), 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Efficiency Metrics by Model", fontsize=11, y=1.02)

    panels = [
        ("avg_iterations", "Avg. Iterations per Question", False),
        ("avg_tokens", "Avg. Total Tokens per Question", True),
    ]

    for ax, (col, label, use_log) in zip(axes, panels):
        for j, cond in enumerate(conditions_present):
            cdf = agg_df[agg_df["condition"] == cond].set_index("model_key")
            vals = [
                cdf.loc[mk, col] if mk in cdf.index else float("nan")
                for mk in all_keys
            ]
            offset = (j - len(conditions_present) / 2 + 0.5) * width
            ax.bar(
                x + offset, vals, width=width,
                color=CONDITION_COLORS[cond],
                label=CONDITION_LABELS[cond],
                edgecolor="white", linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(all_display, rotation=20, ha="right", fontsize=7)
        ax.set_ylabel(label, fontsize=8)
        if use_log:
            non_nan = [
                v for v in agg_df[col].tolist()
                if isinstance(v, (int, float)) and not math.isnan(v) and v > 0
            ]
            if non_nan and max(non_nan) / max(min(non_nan), 1) > 50:
                ax.set_yscale("log")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    seen: dict[str, mpl.patches.Patch] = {}
    for h, lbl in zip(handles, labels):
        seen.setdefault(lbl, h)
    fig.legend(list(seen.values()), list(seen.keys()),
               loc="upper right", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / "fig_efficiency.pdf", dpi)


# ── LaTeX tables ──────────────────────────────────────────────────────────────

def _fmt(val: float, decimals: int = 2) -> str:
    if isinstance(val, float) and math.isnan(val):
        return "--"
    return f"{val:.{decimals}f}"


def _row_vals(sub: pd.DataFrame) -> tuple[str, str, str]:
    if sub.empty:
        return "--", "--", "--"
    r = sub.iloc[0]
    return _fmt(r["approach_mean"]), _fmt(r["accuracy_mean"]), _fmt(r["sources_mean"])


def latex_main_table(agg_df: pd.DataFrame) -> str:
    lines: list[str] = [
        r"% === Main Results Table ===",
        r"\begin{tabular}{l ccc ccc}",
        r"\toprule",
        (r" & \multicolumn{3}{c}{With Tools} "
         r"& \multicolumn{3}{c}{Without Tools} \\"),
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        (r"Model & Approach & Accuracy & Sources "
         r"& Approach & Accuracy & Sources \\"),
        (r"      & (1--5)   & (0--1)   & (1--5)  "
         r"& (1--5)   & (0--1)   & (1--5)  \\"),
        r"\midrule",
    ]
    for mk in sorted(agg_df["model_key"].unique()):
        wt = agg_df[(agg_df["model_key"] == mk) & (agg_df["condition"] == "with_tools")]
        wot = agg_df[(agg_df["model_key"] == mk) & (agg_df["condition"] == "without_tools")]
        display = make_display_name(mk)
        ap_wt, ac_wt, so_wt = _row_vals(wt)
        ap_wot, ac_wot, so_wot = _row_vals(wot)
        lines.append(
            f"{display} & {ap_wt} & {ac_wt} & {so_wt} "
            f"& {ap_wot} & {ac_wot} & {so_wot} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def latex_efficiency_table(agg_df: pd.DataFrame) -> str:
    lines: list[str] = [
        r"% === Efficiency Table (With Tools) ===",
        r"% Note: latency excluded — network round-trip times vary per provider",
        r"% and are not a fair model-capability metric.",
        r"\begin{tabular}{l ccc}",
        r"\toprule",
        r"Model & Avg Iterations & Avg Tokens & Success Rate \\",
        r"\midrule",
    ]
    df_wt = agg_df[agg_df["condition"] == "with_tools"]
    for mk in sorted(agg_df["model_key"].unique()):
        display = make_display_name(mk)
        sub = df_wt[df_wt["model_key"] == mk]
        if sub.empty:
            lines.append(f"{display} & -- & -- & -- \\\\")
            continue
        r = sub.iloc[0]
        iters = _fmt(r["avg_iterations"], 1)
        tok = (
            f"{r['avg_tokens']:,.0f}"
            if not math.isnan(r["avg_tokens"]) else "--"
        )
        succ = (
            f"{r['success_rate'] * 100:.1f}\\%"
            if not math.isnan(r["success_rate"]) else "--"
        )
        lines.append(f"{display} & {iters} & {tok} & {succ} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def write_tables(tables: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(tables) + "\n")
    print(f"  Saved: {out_path}")


# ── Summary CSV ───────────────────────────────────────────────────────────────

def write_summary_csv(agg_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate results figures and LaTeX tables from "
            "benchmark traces and per-question eval files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--trace-dir", type=Path, default=Path("benchmark_traces"),
        help="Root dir: {condition}/{model_key}/trace_q*.json",
    )
    p.add_argument(
        "--eval-dir", type=Path, default=Path("evaluation_results"),
        help="Root dir: {condition}/{model_key}/q{N}.json",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("latex/figures"),
        help="Output directory for PDF figures.",
    )
    p.add_argument(
        "--tables-out", type=Path, default=Path("latex/tables.tex"),
        help="Output path for LaTeX table snippets.",
    )
    p.add_argument(
        "--csv-out", type=Path, default=Path("results/summary.csv"),
        help="Output path for the flat summary CSV.",
    )
    p.add_argument(
        "--conditions", nargs="+",
        default=["with_tools", "without_tools"],
        help="Experimental conditions to include.",
    )
    p.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for saved PDF figures.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_mpl()

    # ── 1. Discovery + loading ─────────────────────────────────────────────────
    print("\n=== Discovery ===")

    all_aggs: list[ModelAgg] = []
    all_cat_dfs: list[pd.DataFrame] = []
    all_diff_dfs: list[pd.DataFrame] = []
    tool_series_list: list[pd.Series] = []
    # Store per-(condition, model_key) records for matched comparison
    all_records: dict[tuple[str, str], list[dict]] = {}

    # Discover model keys from eval dir (primary) and trace dir (secondary).
    all_model_keys: set[str] = set()
    for condition in args.conditions:
        for base in (args.eval_dir, args.trace_dir):
            all_model_keys.update(discover_models(base, condition))

    for condition in args.conditions:
        models = sorted(
            mk for mk in all_model_keys
            if (args.eval_dir / condition / mk).exists()
            or (args.trace_dir / condition / mk).exists()
        )
        print(f"\n  Condition : {condition}")
        if not models:
            print("    (no model directories found)")
            continue
        print(f"  Models    : {', '.join(models)}")

        for mk in models:
            print(f"\n    [{mk}]")

            eval_qs = load_eval_by_qid(args.eval_dir, condition, mk)
            traces = load_traces_by_qid(args.trace_dir, condition, mk)

            print(
                f"      eval files: {len(eval_qs)}, "
                f"trace files: {len(traces)}"
            )

            if not eval_qs:
                print(f"      [skip] No eval files — skipping {mk}/{condition}")
                continue

            records = join_question_data(eval_qs, traces)
            print(f"      joined records: {len(records)}")
            all_records[(condition, mk)] = records

            agg = build_model_agg(records, mk, condition)
            all_aggs.append(agg)

            cat_df = build_category_df(records, mk, agg.display_name, condition)
            if not cat_df.empty:
                all_cat_dfs.append(cat_df)

            diff_df = build_difficulty_df(records, mk, agg.display_name, condition)
            if not diff_df.empty:
                all_diff_dfs.append(diff_df)

            if condition == "with_tools":
                ts = compute_tool_usage(records)
                if not ts.empty:
                    tool_series_list.append(ts)

    if not all_aggs:
        print(
            "\n[ERROR] No data loaded. "
            "Check --trace-dir and --eval-dir paths.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 2. Combine DataFrames ──────────────────────────────────────────────────
    agg_df = pd.DataFrame([vars(a) for a in all_aggs])
    cat_df = (
        pd.concat(all_cat_dfs, ignore_index=True)
        if all_cat_dfs else pd.DataFrame()
    )
    diff_df = (
        pd.concat(all_diff_dfs, ignore_index=True)
        if all_diff_dfs else pd.DataFrame()
    )
    combined_tools = (
        pd.concat(tool_series_list).groupby(level=0).sum().sort_values(ascending=False)
        if tool_series_list else pd.Series(dtype=int)
    )

    # ── 3. Print summary ───────────────────────────────────────────────────────
    print("\n=== Aggregate Summary (all questions per condition) ===")
    cols = ["condition", "model_key", "n_questions",
            "approach_mean", "accuracy_mean", "sources_mean", "success_rate"]
    print(agg_df[[c for c in cols if c in agg_df.columns]].to_string(index=False))
    print("\n  Note: fig_with_vs_without.pdf uses matched question IDs only "
          "(intersection across conditions per model).")

    # ── 4. Figures ─────────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Generating Figures → {args.out_dir} ===")

    plot_main_results(agg_df, args.out_dir, args.dpi)

    # Build matched aggregates (intersection of question IDs across conditions)
    # so the with-vs-without comparison is always apples-to-apples.
    matched_agg_df = build_matched_agg_df(all_records, args.conditions)
    if matched_agg_df.empty:
        print("  [warn] No overlapping question IDs across conditions; "
              "falling back to unmatched aggregates for with_vs_without plot.")
        matched_agg_df = agg_df
    else:
        n_matched = matched_agg_df["n_questions"].min() if "n_questions" in matched_agg_df.columns else "?"
        print(f"  Matched comparison restricted to {n_matched} shared questions per model.")
    plot_with_vs_without(matched_agg_df, args.out_dir, args.dpi)

    primary = args.conditions[0]
    plot_by_category(cat_df, args.out_dir, args.dpi, condition=primary)
    plot_by_difficulty(diff_df, args.out_dir, args.dpi, condition=primary)

    plot_tool_usage(combined_tools, args.out_dir, args.dpi)
    plot_efficiency(agg_df, args.out_dir, args.dpi)

    # ── 5. Tables ──────────────────────────────────────────────────────────────
    print(f"\n=== Generating Tables → {args.tables_out} ===")
    write_tables(
        [latex_main_table(agg_df), latex_efficiency_table(agg_df)],
        args.tables_out,
    )

    # ── 6. CSV ─────────────────────────────────────────────────────────────────
    print(f"\n=== Exporting CSV → {args.csv_out} ===")
    write_summary_csv(agg_df, args.csv_out)

    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
