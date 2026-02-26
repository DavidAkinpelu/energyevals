import json
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.loader import (
    TRACES_ROOT,
    TraceRef,
    compute_run_stats,
    compute_step_timeline,
    filter_traces,
    get_model_keys,
    get_trials,
    list_runs,
    load_run_index,
    load_trace,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="energBench Traces",
    page_icon="⚡",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

STEP_COLORS = {
    "thought": "#3B82F6",
    "action": "#F59E0B",
    "observation": "#10B981",
    "answer": "#14B8A6",
    "error": "#EF4444",
}

BADGE_CSS_MAP = {
    True: ("OK", "#16a34a", "#dcfce7"),
    False: ("FAIL", "#b91c1c", "#fee2e2"),
    None: ("?", "#6b7280", "#f3f4f6"),
}


def inject_custom_css() -> None:
    step_rules = "\n".join(
        f"""
        .step-{stype} {{
            border-left: 4px solid {color};
            padding-left: 0.6rem;
            margin-bottom: 0.4rem;
        }}
        .badge-{stype} {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            color: {color};
            background: {color}22;
            margin-right: 4px;
        }}
        """
        for stype, color in STEP_COLORS.items()
    )
    st.markdown(
        f"""
        <style>
        {step_rules}
        .trace-ok   {{ color: #16a34a; font-weight: 700; }}
        .trace-fail {{ color: #b91c1c; font-weight: 700; }}
        .trace-unk  {{ color: #6b7280; font-weight: 700; }}
        .metric-label {{ font-size: 0.8rem; color: #6b7280; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "selected_run": None,
    "selected_trace_path": None,
    "filter_models": [],
    "filter_trials": [],
    "filter_difficulties": [],
    "filter_categories": [],
    "filter_q_search": "",
    "show_failed_only": False,
    "show_success_only": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _reset_trace() -> None:
    st.session_state["selected_trace_path"] = None


# ---------------------------------------------------------------------------
# Helper: success badge HTML
# ---------------------------------------------------------------------------

def _badge_html(success: bool | None) -> str:
    label, fg, bg = BADGE_CSS_MAP.get(success, ("?", "#6b7280", "#f3f4f6"))
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f"font-size:0.75rem;font-weight:700;color:{fg};background:{bg};"
        f'">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(index: list[TraceRef], corrupt: int) -> list[TraceRef]:
    with st.sidebar:
        st.title("⚡ energBench")

        runs = list_runs(str(TRACES_ROOT))
        if not runs:
            st.warning("No runs found in `benchmark_traces/`.")
            st.stop()

        selected_run = st.selectbox(
            "Run",
            runs,
            index=runs.index(st.session_state["selected_run"])
            if st.session_state["selected_run"] in runs
            else 0,
            on_change=_reset_trace,
            key="selected_run",
        )

        if corrupt:
            st.caption(f"⚠️ {corrupt} corrupt file(s) skipped")

        all_models = get_model_keys(index)
        filter_models = st.multiselect(
            "Models",
            all_models,
            key="filter_models",
        )

        # --- Filters ---
        with st.expander("Filters", expanded=False):
            # Trials — gather across all or selected models
            model_scope = filter_models if filter_models else all_models
            all_trials: list[int] = []
            for mk in model_scope:
                for t in get_trials(index, mk):
                    if t not in all_trials:
                        all_trials.append(t)
            all_trials.sort()
            # Drop any stale trial values that no longer exist in current options
            valid_trials = [t for t in st.session_state["filter_trials"] if t in all_trials]
            if valid_trials != st.session_state["filter_trials"]:
                st.session_state["filter_trials"] = valid_trials
            filter_trials = st.multiselect(
                "Trials",
                all_trials,
                key="filter_trials",
            )

            all_difficulties = sorted({r.difficulty for r in index if r.difficulty})
            filter_difficulties = st.multiselect(
                "Difficulty",
                all_difficulties,
                key="filter_difficulties",
            )

            all_categories = sorted({r.category for r in index if r.category})
            filter_categories = st.multiselect(
                "Category",
                all_categories,
                key="filter_categories",
            )

            filter_q_search = st.text_input(
                "Question ID",
                placeholder="e.g. 42",
                key="filter_q_search",
            )

            col1, col2 = st.columns(2)
            show_success_only = col1.checkbox(
                "Success only",
                key="show_success_only",
            )
            show_failed_only = col2.checkbox(
                "Failed only",
                key="show_failed_only",
            )

        filtered = filter_traces(
            index,
            model_keys=filter_models,
            trials=filter_trials,
            difficulties=filter_difficulties,
            categories=filter_categories,
            q_id_search=filter_q_search,
            success_only=show_success_only,
            failed_only=show_failed_only,
        )

        passed_n = sum(1 for r in filtered if r.success is True)
        failed_n = sum(1 for r in filtered if r.success is False)
        st.caption(f"{len(filtered)} traces · {passed_n} passed · {failed_n} failed")

        st.divider()

        if st.button("Model Overview", use_container_width=True):
            st.session_state["selected_trace_path"] = None

        st.subheader("Traces")
        for ref in sorted(filtered, key=lambda r: r.question_id):
            label, fg, bg = BADGE_CSS_MAP.get(ref.success, ("?", "#6b7280", "#f3f4f6"))
            dur = f"{ref.duration_seconds:.0f}s" if ref.duration_seconds is not None else "—"
            btn_label = f"Q{ref.question_id} · {dur}  [{label}]"
            is_selected = (
                st.session_state["selected_trace_path"] == str(ref.path)
            )
            if st.button(
                btn_label,
                key=f"btn_{ref.path}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state["selected_trace_path"] = str(ref.path)

    return filtered


# ---------------------------------------------------------------------------
# Run Overview
# ---------------------------------------------------------------------------

def render_overview(index: list[TraceRef]) -> None:
    stats = compute_run_stats(index)

    selected_models: list[str] = st.session_state.get("filter_models") or []
    if selected_models:
        model_label = ", ".join(selected_models)
    else:
        model_label = "All Models"
    st.title(f"Model Overview — {model_label}")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Traces", stats["total"])
    c2.metric("Success Rate", f"{stats['success_rate']:.1f}%")
    avg_dur = stats["avg_duration"]
    c3.metric("Avg Duration", f"{avg_dur:.1f}s" if avg_dur is not None else "—")
    avg_tok = stats["avg_tokens"]
    c4.metric("Avg Tokens", f"{int(avg_tok):,}" if avg_tok is not None else "—")

    st.divider()

    # Charts row 1
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Success Rate by Model")
        by_model = stats["by_model"]
        if by_model:
            df_model = pd.DataFrame(
                [
                    {
                        "Model": k,
                        "Success Rate (%)": v["success_rate"],
                        "Color": "Good" if v["success_rate"] >= 90 else "Needs Improvement",
                    }
                    for k, v in by_model.items()
                ]
            )
            chart = (
                alt.Chart(df_model)
                .mark_bar()
                .encode(
                    x=alt.X("Success Rate (%):Q", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Model:N", sort="-x"),
                    color=alt.Color(
                        "Color:N",
                        scale=alt.Scale(
                            domain=["Good", "Needs Improvement"],
                            range=["#16a34a", "#d97706"],
                        ),
                        legend=None,
                    ),
                    tooltip=["Model", "Success Rate (%)"],
                )
                .properties(height=max(120, len(df_model) * 40))
            )
            st.altair_chart(chart, use_container_width=True)

    with col_right:
        st.subheader("Success / Fail by Difficulty")
        by_diff = stats["by_difficulty"]
        if by_diff:
            rows = []
            for diff, v in by_diff.items():
                rows.append({"Difficulty": diff, "Count": v["passed"], "Status": "Passed"})
                rows.append({"Difficulty": diff, "Count": v["failed"], "Status": "Failed"})
            df_diff = pd.DataFrame(rows)
            chart = (
                alt.Chart(df_diff)
                .mark_bar()
                .encode(
                    x=alt.X("Difficulty:N"),
                    y=alt.Y("Count:Q"),
                    color=alt.Color(
                        "Status:N",
                        scale=alt.Scale(
                            domain=["Passed", "Failed"],
                            range=["#16a34a", "#dc2626"],
                        ),
                    ),
                    tooltip=["Difficulty", "Status", "Count"],
                )
                .properties(height=250)
            )
            st.altair_chart(chart, use_container_width=True)

    # Charts row 2
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.subheader("Token Distribution")
        if stats["tokens"]:
            df_tok = pd.DataFrame({"Total Tokens": stats["tokens"]})
            chart = (
                alt.Chart(df_tok)
                .mark_bar(color="#6366f1")
                .encode(
                    x=alt.X("Total Tokens:Q", bin=alt.Bin(maxbins=30)),
                    y=alt.Y("count():Q", title="Count"),
                    tooltip=["count()"],
                )
                .properties(height=220)
            )
            st.altair_chart(chart, use_container_width=True)

    with col_r2:
        st.subheader("Duration Distribution")
        if stats["durations"]:
            df_dur = pd.DataFrame({"Duration (s)": stats["durations"]})
            chart = (
                alt.Chart(df_dur)
                .mark_bar(color="#0891b2")
                .encode(
                    x=alt.X("Duration (s):Q", bin=alt.Bin(maxbins=30)),
                    y=alt.Y("count():Q", title="Count"),
                    tooltip=["count()"],
                )
                .properties(height=220)
            )
            st.altair_chart(chart, use_container_width=True)

    st.divider()

    # Tables
    t1, t2, t3 = st.tabs(["By Model", "By Difficulty", "By Category"])

    def _stats_df(breakdown: dict[str, dict]) -> pd.DataFrame:
        rows = []
        for k, v in breakdown.items():
            rows.append(
                {
                    "Name": k,
                    "Total": v["total"],
                    "Passed": v["passed"],
                    "Failed": v["failed"],
                    "Success %": round(v["success_rate"], 1),
                    "Avg Duration (s)": round(v["avg_duration"], 1)
                    if v["avg_duration"]
                    else None,
                    "Avg Tokens": int(v["avg_tokens"]) if v["avg_tokens"] else None,
                }
            )
        return pd.DataFrame(rows)

    with t1:
        st.dataframe(_stats_df(stats["by_model"]), use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(_stats_df(stats["by_difficulty"]), use_container_width=True, hide_index=True)
    with t3:
        st.dataframe(_stats_df(stats["by_category"]), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Trace Detail
# ---------------------------------------------------------------------------

def render_trace_detail(trace_path_str: str, index: list[TraceRef]) -> None:
    # Find the ref by path
    ref: TraceRef | None = next(
        (r for r in index if str(r.path) == trace_path_str), None
    )
    if ref is None:
        # Try to construct a minimal ref from path alone
        path = Path(trace_path_str)
        ref_fallback = True
    else:
        ref_fallback = False

    data = load_trace(ref) if ref else None
    if data is None:
        path = Path(trace_path_str)
        try:
            with path.open() as fh:
                data = json.load(fh)
        except Exception:
            st.error("Could not load trace file.")
            return

    meta = data.get("metadata") or {}
    metrics = data.get("metrics") or {}
    step_summary = data.get("step_summary") or {}

    q_id = meta.get("question_id") or (ref.question_id if ref else "?")
    category = meta.get("category") or (ref.category if ref else "?")
    difficulty = meta.get("difficulty") or (ref.difficulty if ref else "?")
    provider = meta.get("provider", "?")
    model = meta.get("model", "?")
    success = data.get("success")

    # Header
    label, fg, bg = BADGE_CSS_MAP.get(success, ("?", "#6b7280", "#f3f4f6"))
    badge_html = (
        f'<span style="padding:3px 10px;border-radius:4px;font-size:0.8rem;'
        f"font-weight:700;color:{fg};background:{bg};\">{label}</span>"
    )
    st.markdown(
        f"## Q{q_id} — {category} &nbsp; {badge_html}",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Difficulty: **{difficulty}** · Provider: **{provider}** · Model: **{model}**"
    )

    if data.get("error"):
        st.error(f"Error: {data['error']}")

    # Query / Final Answer tabs
    tab_q, tab_a = st.tabs(["Query", "Final Answer"])
    with tab_q:
        st.markdown(data.get("query") or "_No query recorded._")
    with tab_a:
        final = data.get("final_answer") or "_No answer recorded._"
        st.markdown(final)

    st.divider()

    # Metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Iterations", metrics.get("iterations", "—"))
    m2.metric("Tool Calls", metrics.get("tool_calls_count", "—"))
    m3.metric("Total Tokens", f"{metrics.get('total_tokens', 0):,}")
    m4.metric("Input Tokens", f"{metrics.get('total_input_tokens', 0):,}")
    m5.metric("Output Tokens", f"{metrics.get('total_output_tokens', 0):,}")
    dur = metrics.get("duration_seconds")
    m6.metric("Duration", f"{dur:.1f}s" if dur is not None else "—")

    st.divider()

    # Step summary
    sc1, sc2 = st.columns(2)
    with sc1:
        st.subheader("Step Types")
        step_types = step_summary.get("step_types") or {}
        if step_types:
            df_st = pd.DataFrame(
                [{"Type": k, "Count": v} for k, v in step_types.items()]
            )
            st.dataframe(df_st, hide_index=True, use_container_width=True)

    with sc2:
        st.subheader("Tool Calls")
        tool_calls = step_summary.get("tool_calls") or []
        failed_tool_calls = step_summary.get("failed_tool_calls") or []
        if tool_calls:
            from collections import Counter

            counts = Counter(tool_calls)
            df_tc = pd.DataFrame(
                [{"Tool": k, "Calls": v} for k, v in counts.most_common()]
            )
            st.dataframe(df_tc, hide_index=True, use_container_width=True)
            if failed_tool_calls:
                st.warning(f"Failed tool calls: {', '.join(failed_tool_calls)}")
        else:
            st.caption("No tool calls recorded.")

    # Per-step latency chart
    raw_steps = data.get("steps") or []
    latency_data = [
        {
            "Step": s.get("index", i),
            "Type": s.get("step_type", "unknown"),
            "Latency (ms)": s.get("latency_ms", 0),
        }
        for i, s in enumerate(raw_steps)
        if s.get("latency_ms", 0) > 0
    ]
    if latency_data:
        st.subheader("Per-Step Latency")
        df_lat = pd.DataFrame(latency_data)
        color_map = list(STEP_COLORS.items())
        domain = [c[0] for c in color_map]
        rng = [c[1] for c in color_map]
        chart = (
            alt.Chart(df_lat)
            .mark_bar()
            .encode(
                x=alt.X("Step:O", title="Step Index"),
                y=alt.Y("Latency (ms):Q"),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(domain=domain, range=rng),
                ),
                tooltip=["Step", "Type", "Latency (ms)"],
            )
            .properties(height=200)
        )
        st.altair_chart(chart, use_container_width=True)

    st.divider()

    # Steps Timeline
    steps = compute_step_timeline(raw_steps)
    if steps:
        st.subheader("Steps Timeline")
        current_iter = -1

        for s in steps:
            iter_num = s.get("iteration", 0)
            step_type = s.get("step_type", "unknown")
            color = STEP_COLORS.get(step_type, "#9ca3af")

            if iter_num != current_iter:
                current_iter = iter_num
                st.markdown(
                    f'<div style="background:#f1f5f9;padding:4px 10px;'
                    f'border-radius:4px;margin:8px 0;color:#64748b;'
                    f'font-size:0.8rem;font-weight:700;">ITERATION {iter_num}</div>',
                    unsafe_allow_html=True,
                )

            # Build expander label
            rel_t = s.get("relative_time_s")
            time_str = f"+{rel_t:.1f}s" if rel_t is not None else ""
            display_label = s.get("display_label", step_type)
            expander_label = (
                f"[{step_type.upper()}] {display_label}"
                + (f"  ({time_str})" if time_str else "")
            )

            is_answer = step_type == "answer"
            with st.expander(expander_label, expanded=is_answer):
                # Left-border accent
                st.markdown(
                    f'<div class="step-{step_type}">',
                    unsafe_allow_html=True,
                )

                if step_type == "thought":
                    content = s.get("content") or ""
                    if content.strip():
                        st.markdown(content)
                    else:
                        st.markdown(
                            "_No thought content recorded._",
                        )

                elif step_type == "action":
                    tool_input = s.get("tool_input")
                    if tool_input is not None:
                        st.code(
                            json.dumps(tool_input, indent=2),
                            language="json",
                        )
                    else:
                        st.caption(s.get("content") or "_No action input recorded._")

                elif step_type == "observation":
                    is_error = s.get("is_error", False)
                    if is_error:
                        st.error(s.get("tool_output_truncated") or "Tool error")
                    elif s.get("tool_output_is_json") and s.get("tool_output_parsed"):
                        parsed = s["tool_output_parsed"]
                        # Pretty print but cap output
                        pretty = json.dumps(parsed, indent=2)
                        if len(pretty) > 8_000:
                            pretty = pretty[:8_000]
                        st.code(pretty, language="json")
                    else:
                        out = s.get("tool_output_truncated") or ""
                        if out:
                            st.text(out)
                        else:
                            st.caption("_No observation output._")

                    if s.get("tool_output_is_truncated"):
                        full_len = s.get("tool_output_full_len", 0)
                        st.caption(
                            f"Output truncated — showing first 8,000 of {full_len:,} chars."
                        )

                elif step_type == "answer":
                    content = s.get("content") or ""
                    st.markdown(content if content else "_No answer content._")

                elif step_type == "error":
                    st.error(s.get("content") or "Unknown error")

                else:
                    st.write(s.get("content") or "")

                st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    runs = list_runs(str(TRACES_ROOT))
    if not runs:
        st.warning("No benchmark runs found. Run the benchmark first.")
        st.stop()

    # Initialise selected_run
    if st.session_state["selected_run"] not in runs:
        st.session_state["selected_run"] = runs[0]

    selected_run: str = st.session_state["selected_run"]
    index, corrupt = load_run_index(selected_run, str(TRACES_ROOT))

    filtered = render_sidebar(index, corrupt)

    selected_trace_path: str | None = st.session_state["selected_trace_path"]

    if selected_trace_path is None:
        render_overview(filtered)
    else:
        render_trace_detail(selected_trace_path, index)


main()
