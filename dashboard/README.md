# EnergyEvals Trace Dashboard

A Streamlit app for browsing and inspecting benchmark traces produced by the EnergyEvals runner.

## Setup

Install all dependencies from the root of the repo (includes `streamlit` and `altair`):

```bash
.venv/bin/pip install -r requirements.txt
```

## Running

From the repo root:

```bash
.venv/bin/python -m streamlit run dashboard/app.py
```

For headless environments (CI, remote servers):

```bash
.venv/bin/python -m streamlit run dashboard/app.py \
    --server.headless=true \
    --browser.gatherUsageStats=false
```

The app will be available at `http://localhost:8501` by default.

## Usage

1. **Select a run** from the dropdown — corresponds to a subdirectory under `benchmark_traces/`.
2. **Filter by model** using the Models multiselect directly below the run selector.
3. Open **Filters** to further narrow by trial, difficulty, category, or question ID.
4. Click **Model Overview** to see aggregate KPIs, charts, and breakdowns for the selected model.
5. Click any trace in the list to open the **Trace Detail** view — full step-by-step ReAct execution with tool inputs, outputs, and per-step latency.

## Trace Data

Traces are read from:

```
benchmark_traces/{run_name}/{provider}_{model}/[trial_N/]trace_q{qid}_{timestamp}_{uuid}.json
```

The dashboard auto-refreshes its file index every 5 minutes (`ttl=300`).
