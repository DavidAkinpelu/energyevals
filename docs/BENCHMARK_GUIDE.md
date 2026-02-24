# energBench Benchmark Guide

A complete guide to installing, configuring, and running benchmarks with energBench -- an AI agent evaluation framework for energy analytics.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Environment Setup (API Keys)](#4-environment-setup-api-keys)
5. [Preparing a Benchmark](#5-preparing-a-benchmark)
6. [Benchmark Configuration Reference](#6-benchmark-configuration-reference)
7. [Running Benchmarks (CLI Reference)](#7-running-benchmarks-cli-reference)
8. [Available Tools Reference](#8-available-tools-reference)
9. [MCP Integration](#9-mcp-integration)
10. [Observability and Tracing](#10-observability-and-tracing)
11. [Expected Output -- What You Will See](#11-expected-output----what-you-will-see)
12. [Common Recipes](#12-common-recipes)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Introduction

energBench evaluates how well LLM-powered agents answer questions about energy markets, grid operations, tariffs, renewables, and regulatory proceedings. A **benchmark run** works as follows:

1. A set of energy-domain questions is loaded from a CSV file.
2. For each question, a **ReAct agent** is created with access to configurable tools (web search, grid data APIs, battery optimization solvers, regulatory docket scrapers, etc.).
3. The agent reasons through the question using a think-act-observe loop, calling tools as needed.
4. The agent's answer, token usage, tool calls, latency, and full execution trace are recorded.
5. Results are saved to JSON and, optionally, to a tracing backend (local JSON files or Langfuse).

You can run benchmarks against a single model or compare multiple models side by side in a single run.

---

## 2. Prerequisites

- **Python 3.10+**
- **At least one LLM provider API key** (OpenAI, Anthropic, Google, or DeepInfra)
- **System dependencies** (optional): The Ipopt solver is required only if you use the `battery` optimization tool. If you do not need battery optimization, you can skip the system-level install.

---

## 3. Installation

### Clone the repository

```bash
git clone <repo-url>
cd energBench
```

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

For development tools (pytest, ruff, mypy):

```bash
pip install -r requirements-dev.txt
```

### Install system solvers (optional)

The battery optimization tool uses [Pyomo](https://www.pyomo.org/) with the Ipopt nonlinear solver. The `install.sh` script builds Ipopt and its COIN-OR dependencies (ASL, Mumps) from source and registers them with Pyomo.

```bash
sudo ./install.sh
```

> **Note:** This step requires `gcc`, `gfortran`, `cmake`, and other build tools. The script installs them automatically on Debian/Ubuntu. If you are not using the `battery` tool, skip this step entirely.

---

## 4. Environment Setup (API Keys)

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials. The table below lists every supported variable:

### LLM Providers (at least one required)

| Variable | Purpose | Required |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI models (GPT-4o, o1, o3, etc.) | At least one provider |
| `ANTHROPIC_API_KEY` | Anthropic models (Claude Sonnet 4, Opus 4) | At least one provider |
| `GOOGLE_API_KEY` | Google models (Gemini 2.0 Flash, 1.5 Pro/Flash) | At least one provider |
| `DEEPINFRA_API_KEY` | DeepInfra models (Llama 3.3, 3.1) | At least one provider |

### Observability (optional)

| Variable | Purpose |
|---|---|
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key for cloud tracing |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse host URL (default: `https://cloud.langfuse.com`) |

### Tool API Keys (optional -- tools warn if missing)

| Variable | Tool | Purpose |
|---|---|---|
| `EXA_API_KEY` | SearchTool | Web search via the Exa API |
| `GRIDSTATUS_API_KEY` | GridStatusAPITool | Electricity grid data (pricing, load, generation) |
| `OPEN_EI_API_KEY` | TariffsTool | Utility tariff data from OpenEI |
| `RENEWABLES_NINJA_API_KEY` | RenewablesTool | Solar and wind generation data |
| `OPENWEATHER_API_KEY` | OpenWeatherTool | Weather data (temperature, wind, conditions) |

### MCP Server URLs (optional)

| Variable | Purpose |
|---|---|
| `RAG_SERVER_URL` | URL for a remote RAG MCP server (vector search over energy documents) |
| `DATABASE_SERVER_URL` | URL for a remote Database MCP server (PostgreSQL query interface) |

> **Tip:** You only need one LLM provider key to run benchmarks. Tool keys are optional -- if a key is missing, that tool will log a warning but the benchmark will still run with the remaining tools.

---

## 5. Preparing a Benchmark

Before running, walk through these steps to set up your benchmark.

### 5a. Prepare the Questions File

The benchmark reads questions from a **CSV file** with the following required columns:

| Column | Description | Example |
|---|---|---|
| `S/N` | Integer ID (serial number) | `1` |
| `Category` | Topic category | `Market rules retrieval` |
| `Question type` | Type classification | `Quantitative data retrieval with sources` |
| `Difficulty level` | Difficulty rating | `Easy`, `Medium`, `Hard` |
| `Question` | The full question text | `What fees are associated with...` |

**Default questions file:** The repository ships with `data/AI Evals New Questions.xlsx - Q&As.csv` containing energy-domain questions across multiple categories and difficulty levels.

**Custom questions file:** Create a CSV with the exact column headers above. Example:

```csv
S/N,Category,Question type,Difficulty level,Question
1,Market rules retrieval,Quantitative data retrieval with sources,Easy,What fees are associated with each decision point in the NYISO generation interconnection process?
2,Grid operations,Analytical reasoning,Medium,Compare the average day-ahead LMP prices across PJM and ERCOT for the past week.
3,Battery optimization,Multi-step computation,Hard,Optimize a 100MW/400MWh battery storage system for revenue maximization in ERCOT.
```

**Preview questions before running:**

```bash
python scripts/run_benchmark.py --list-questions
```

This prints every question with its ID, category, and difficulty level, so you can pick which ones to run.

### 5b. Choose Models

Decide which LLM providers and models to evaluate. Supported options:

| Provider | Models |
|---|---|
| `openai` | `gpt-4o`, `gpt-4o-mini`, `o1`, `o1-mini`, `o3`, `o3-mini`, `o4-mini` |
| `anthropic` | `claude-sonnet-4-20250514`, `claude-opus-4-20250514` |
| `google` | `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash` |
| `deepinfra` | `meta-llama/Llama-3.3-70B-Instruct-Turbo`, `meta-llama/Meta-Llama-3.1-405B-Instruct` |

Configure models in YAML:

```yaml
models:
  - provider: openai
    model: gpt-4o-mini
  - provider: anthropic
    model: claude-sonnet-4-20250514
```

**Reasoning model override:** OpenAI reasoning models (o1, o3, etc.) use different API parameters (`reasoning_effort` instead of `temperature`). The system auto-detects this based on the model name, but you can override it:

```yaml
models:
  - provider: openai
    model: o3-mini
    is_reasoning_model: true    # Force reasoning model behavior
```

### 5c. Configure Tools

Decide which tools the agent should have access to. There are four patterns:

**Pattern 1 -- All tools (default):**

```yaml
tools:
  enabled: true
  include: []    # Empty = all tools
  exclude: []
```

**Pattern 2 -- Include only specific tools:**

```yaml
tools:
  enabled: true
  include: [search, gridstatus, battery]
  exclude: []
```

**Pattern 3 -- Exclude specific tools:**

```yaml
tools:
  enabled: true
  include: []
  exclude: [docket, openweather]
```

**Pattern 4 -- No tools (reasoning only):**

```yaml
tools:
  enabled: false
```

Make sure the API keys for your chosen tools are set in `.env` (see [Section 4](#4-environment-setup-api-keys)).

**Preview available tools:**

```bash
python scripts/run_benchmark.py --list-tools
```

### 5d. Select Questions

By default, all questions in the CSV are run. To run a subset:

**In the config file:**

```yaml
# Specific IDs
questions: [1, 2, 3]

# Range
questions: 1-5
```

**On the command line:**

```bash
python scripts/run_benchmark.py --questions 1,2,3
python scripts/run_benchmark.py --questions 1-5
```

> **Tip:** Start with 2-3 questions to validate your setup before running the full benchmark. This saves time and API costs.

### 5e. Configure Observability

Choose how execution traces are stored:

| Backend | Description | Required env vars |
|---|---|---|
| `json` | Local JSON files (one per question) | None |
| `langfuse` | Cloud tracing with Langfuse UI | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` |

Set `run_name` to organize experiments into subdirectories:

```yaml
observability:
  enabled: true
  backend: json
  output_dir: ./benchmark_traces
  run_name: with_tools    # Traces saved to: ./benchmark_traces/with_tools/{model}/
```

### 5f. Write the Config File

Copy the default config and customize it:

```bash
cp configs/benchmark_config.yaml configs/my_benchmark.yaml
```

**Minimal config (single model, all defaults):**

```yaml
models:
  - provider: openai
    model: gpt-4o-mini

questions_file: data/AI Evals New Questions.xlsx - Q&As.csv
```

**Full-featured config (multi-model comparison):**

```yaml
models:
  - provider: openai
    model: gpt-4o
  - provider: anthropic
    model: claude-sonnet-4-20250514

questions_file: data/AI Evals New Questions.xlsx - Q&As.csv
questions: 1-10

tools:
  enabled: true
  include: [search, gridstatus, battery, tariffs]
  exclude: []

agent:
  max_iterations: 25
  num_trials: 3              # Run each question 3 times for statistical evaluation
  shuffle: true
  seed_mode: rotate
  seed: 12345

mcp:
  enabled: true

observability:
  enabled: true
  backend: json
  output_dir: ./benchmark_traces
  run_name: model_comparison_v1

output:
  results_dir: ./benchmark_results
  save_answers: true
```

---

## 6. Benchmark Configuration Reference

Complete reference for every field in the YAML config file.

### `models` (required)

List of models to evaluate. Each entry requires `provider` and `model`:

```yaml
models:
  - provider: openai          # Required: openai | anthropic | google | deepinfra
    model: gpt-4o-mini        # Required: model name
    is_reasoning_model: true   # Optional: override reasoning model auto-detection
```

### `questions_file`

Path to the CSV questions file, relative to the project root.

```yaml
questions_file: data/AI Evals New Questions.xlsx - Q&As.csv
```

### `questions`

Optional filter. Omit or set to `null` to run all questions.

```yaml
questions: [1, 2, 3]     # Specific IDs
questions: 1-5            # Range
questions: 1,3,5-10       # Mixed
```

### `tools`

```yaml
tools:
  enabled: true            # true (default) or false to disable all tools
  include: []              # List of tool names to include (empty = all)
  exclude: []              # List of tool names to exclude
```

Available tool names: `battery`, `docket`, `gridstatus`, `openweather`, `renewables`, `search`, `system`, `tariffs`.

### `agent`

```yaml
agent:
  max_iterations: 25       # Maximum ReAct loop iterations per question (default: 25)
  num_trials: 1            # Independent trials per question (default: 1)
  shuffle: false           # Shuffle question order within each trial
  seed_mode: rotate        # fixed | rotate | random_per_trial (ignored when shuffle=false)
  seed: null               # Base seed for fixed/rotate when seeds is not provided
  # seeds: [101, 202, 303] # Optional explicit per-trial seeds (length must equal num_trials)
```

When `num_trials` is greater than 1, each question is run N independent times per model. Traces are stored in `trial_N/` subdirectories, and the results JSON groups answers by trial. This is designed for statistical evaluation -- the downstream evaluation pipeline uses multiple trials to compute confidence intervals and significance tests between models.

When `shuffle: true`, per-trial seeds are resolved in this order:
1. `seeds` list (explicit per-trial seeds).
2. `seed_mode=fixed` uses one seed for all trials.
3. `seed_mode=rotate` uses `seed + (trial - 1)`.
4. `seed_mode=random_per_trial` draws a fresh seed every trial.

### `mcp`

```yaml
mcp:
  enabled: true            # Enable MCP tools (requires server URLs in .env)
```

### `observability`

```yaml
observability:
  enabled: true            # Enable/disable tracing
  backend: json            # json | langfuse
  output_dir: ./benchmark_traces
  run_name: experiment_1   # Optional: subdirectory for organizing runs
```

### `output`

```yaml
output:
  results_dir: ./benchmark_results   # Directory for result JSON files
  save_answers: true                 # Include full answers in results (default: true)
```

---

## 7. Running Benchmarks (CLI Reference)

The benchmark runner is at `scripts/run_benchmark.py`. All arguments are optional and override the config file.

### Basic usage

```bash
# Use default config (configs/benchmark_config.yaml)
python scripts/run_benchmark.py

# Use a custom config file
python scripts/run_benchmark.py -c configs/my_benchmark.yaml
```

### Model overrides

```bash
# Single model override
python scripts/run_benchmark.py --provider openai --model gpt-4o

# Multi-model from CLI (provider:model pairs)
python scripts/run_benchmark.py --models openai:gpt-4o anthropic:claude-sonnet-4-20250514
```

### Question selection

```bash
# Specific questions
python scripts/run_benchmark.py --questions 1,2,3

# Range of questions
python scripts/run_benchmark.py --questions 1-5
```

### Tool filtering

```bash
# Include only specific tools
python scripts/run_benchmark.py --tools search,gridstatus,battery

# Exclude specific tools
python scripts/run_benchmark.py --exclude-tools docket,tariffs
```

### Multiple trials

```bash
# Run 3 independent trials per question (for statistical evaluation)
python scripts/run_benchmark.py --num-trials 3

# Combine with multi-model
python scripts/run_benchmark.py --models openai:gpt-4o anthropic:claude-sonnet-4-20250514 --num-trials 3
```

### Feature toggles

```bash
# Disable MCP tools
python scripts/run_benchmark.py --no-mcp

# Disable observability tracing
python scripts/run_benchmark.py --no-observe

# Override reasoning model detection
python scripts/run_benchmark.py --reasoning-model true
```

### Inspection commands

```bash
# List all available questions
python scripts/run_benchmark.py --list-questions

# List all available tools (standard + MCP)
python scripts/run_benchmark.py --list-tools
```

### Full CLI reference

| Flag | Short | Description |
|---|---|---|
| `--config` | `-c` | Path to config YAML (default: `configs/benchmark_config.yaml`) |
| `--provider` | `-p` | Override provider (`openai`, `anthropic`, `google`, `deepinfra`) |
| `--model` | `-m` | Override model name (single model mode) |
| `--models` | | Space-separated `provider:model` pairs for multi-model |
| `--questions` | `-q` | Question filter (e.g. `1,2,3` or `1-5`) |
| `--list-questions` | `-l` | List questions and exit |
| `--list-tools` | | List tools and exit |
| `--no-observe` | | Disable observability |
| `--no-mcp` | | Disable MCP tools |
| `--reasoning-model` | | Override reasoning model detection (`true`, `false`, `auto`) |
| `--tools` | | Include only these tools (comma-separated) |
| `--exclude-tools` | | Exclude these tools (comma-separated) |
| `--num-trials` | | Number of independent trials per question (default: 1) |

---

## 8. Available Tools Reference

The following tools are registered by default. The agent decides which tools to call based on the question.

| Tool | Name in config | Description | API Key Required |
|---|---|---|---|
| **SearchTool** | `search` | Web search via the Exa API. Returns relevant web pages and snippets. | `EXA_API_KEY` |
| **GridStatusAPITool** | `gridstatus` | Electricity grid data from GridStatus -- real-time and historical pricing (LMP), load, generation, and interchange data for US ISOs. | `GRIDSTATUS_API_KEY` |
| **TariffsTool** | `tariffs` | Utility rate/tariff data from the OpenEI database. | `OPEN_EI_API_KEY` |
| **RenewablesTool** | `renewables` | Solar and wind generation simulation data from Renewables.ninja for any location. | `RENEWABLES_NINJA_API_KEY` |
| **BatteryOptimizationTool** | `battery` | Battery energy storage revenue optimization using Pyomo and the Ipopt nonlinear solver. | None (requires Ipopt installed via `install.sh`) |
| **OpenWeatherTool** | `openweather` | Current and forecast weather data from OpenWeather. | `OPENWEATHER_API_KEY` |
| **SystemTool** | `system` | General-purpose system/utility functions (date/time, calculations). | None |
| **Docket Tools** | `docket` | Regulatory docket search for FERC, Maryland, Texas, New York, North Carolina, South Carolina, Virginia, and DC public utility commissions. Scrape-based, no API key needed. | None |

> **Note:** If a tool's API key is missing, the tool logs a warning but does not crash the benchmark. The agent simply cannot use that tool.

---

## 9. MCP Integration

The Model Context Protocol (MCP) provides two additional tool servers:

- **RAG Server** -- Vector search over energy documents (tariffs, interconnection guides, regulatory filings).
- **Database Server** -- SQL queries against a PostgreSQL database with energy market data.

### Remote setup (recommended)

If MCP servers are deployed remotely, add their URLs to `.env`:

```bash
RAG_SERVER_URL=https://your-rag-server.com/sse
DATABASE_SERVER_URL=https://your-db-server.com/sse
```

### Local setup

Install the server packages locally:

```bash
pip install ./mcp-servers/rag-server
pip install ./mcp-servers/database-server
```

Each server has its own `.env` for database credentials and embedding model configuration. See `mcp-servers/DEPLOYMENT_GUIDE.md` for details.

### Disabling MCP

If you do not need MCP tools:

```yaml
# In config file
mcp:
  enabled: false
```

Or on the command line:

```bash
python scripts/run_benchmark.py --no-mcp
```

---

## 10. Observability and Tracing

Observability captures the full execution trace of every benchmark question -- every thought, tool call, observation, and the final answer.

### JSON backend (default)

Writes one JSON file per question to disk. No external services required.

```yaml
observability:
  enabled: true
  backend: json
  output_dir: ./benchmark_traces
```

Trace files are organized by model:

```
benchmark_traces/{provider}_{model}/trace_q{id}_{timestamp}.json
```

### Langfuse backend

Sends traces to [Langfuse](https://langfuse.com/) for a web-based UI with search, filtering, and analytics.

```yaml
observability:
  enabled: true
  backend: langfuse
```

Requires `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` in `.env`.

### Organizing experiment runs

Use `run_name` to group traces into subdirectories. This is useful when comparing different configurations:

```yaml
observability:
  run_name: with_tools       # Traces go to: benchmark_traces/with_tools/{model}/
```

Then run again with different settings:

```yaml
observability:
  run_name: no_tools         # Traces go to: benchmark_traces/no_tools/{model}/
```

### Config reproducibility

The benchmark automatically copies your `benchmark_config.yaml` into the trace output directory, so you always know which configuration produced a given set of traces.

---

## 11. Expected Output -- What You Will See

### 11a. Console Output During a Run

When you run a benchmark, the terminal shows progress in real time. Here is an example of what you will see:

```
======================================================================
  energBench Benchmark Runner
======================================================================

  Configuration:
    Models (2):
      - openai/gpt-4o-mini
      - anthropic/claude-sonnet-4-20250514
    Questions file: data/AI Evals New Questions.xlsx - Q&As.csv
    Questions: [1, 2, 3]
    MCP enabled: True
    Observability: json
    Max iterations: 25

  Loaded 3 questions
  Observability: JSONFileObserver (run: experiment_1)
  Config saved to: benchmark_traces/experiment_1/benchmark_config.yaml
  Standard tools: 15/15
  MCP tools: 4 loaded
  Total tools available: 19

======================================================================
  Running Multi-Model Benchmark
======================================================================
  Models: 2
    - openai/gpt-4o-mini
    - anthropic/claude-sonnet-4-20250514

======================================================================
  Evaluating: openai/gpt-4o-mini
======================================================================

  [1/3] Question 1 | Market rules retrieval | Easy
  What fees are associated with each decision point in the NYISO generation...

  [PASS]
  Answer: The NYISO generation interconnection process involves several fees...
  Metrics: tokens=4200, tools=3, time=12.3s
  Trace: 20260207_143012_123456

  [2/3] Question 2 | Market rules retrieval | Easy
  What fees are associated with each decision point in the PJM generation...

  [PASS]
  Answer: According to PJM Manual 14H, the interconnection process fees...
  Metrics: tokens=3800, tools=2, time=9.7s
  Trace: 20260207_143025_234567

  [3/3] Question 3 | Market rules retrieval | Easy
  What are the fees associated with each milestone in the ERCOT generation...

  [FAIL]
  Error: Max iterations reached
  Metrics: tokens=8500, tools=8, time=45.2s
  Trace: 20260207_143110_345678

======================================================================
  Evaluating: anthropic/claude-sonnet-4-20250514
======================================================================
  ...

======================================================================
  Summary
======================================================================

  openai/gpt-4o-mini:
    Questions: 3
    Passed: 2 (67%)
    Failed: 1
    Total tokens: 16,500
    Total time: 67.2s

  anthropic/claude-sonnet-4-20250514:
    Questions: 3
    Passed: 3 (100%)
    Failed: 0
    Total tokens: 12,300
    Total time: 52.1s

  Results saved: benchmark_results/benchmark_multi_20260207_143500.json
```

**Key elements in the output:**

- `[PASS]` / `[FAIL]` -- whether the agent produced an answer within the iteration limit.
- **Answer preview** -- first 300 characters of the agent's response.
- **Metrics line** -- total tokens consumed, number of tool calls, and wall-clock time for that question.
- **Trace ID** -- identifier linking to the detailed trace file.
- **Summary** -- aggregate pass/fail counts, total tokens, and total time per model.

**Multi-trial output:** When `num_trials > 1`, the console output includes trial markers and trial counts in the summary:

```
======================================================================
  Running Multi-Model Benchmark
======================================================================
  Models: 1
    - openai/gpt-4o
  Trials: 3

  --- Trial 1/3 ---

======================================================================
  Evaluating: openai/gpt-4o
======================================================================

  [1/3] Question 1 | Market rules retrieval | Easy
  ...
  [PASS]
  Metrics: tokens=4200, tools=3, time=12.3s

  --- Trial 2/3 ---

======================================================================
  Evaluating: openai/gpt-4o
======================================================================

  [1/3] Question 1 | Market rules retrieval | Easy
  ...
  [PASS]
  Metrics: tokens=3900, tools=3, time=11.8s

  --- Trial 3/3 ---

  ...

======================================================================
  Summary
======================================================================

  openai/gpt-4o:
    Questions: 3
    Trials: 3
    Passed: 9 (100%)
    Failed: 0
    Total tokens: 38,400
    Total time: 105.7s

  Results saved: benchmark_results/benchmark_openai_20260207_150000.json
```

### 11b. Results JSON File

After a benchmark completes, results are saved as JSON.

**Location:** `benchmark_results/` (configurable via `output.results_dir`).

**File naming:**

- Single model: `benchmark_{provider}_{YYYYMMDD_HHMMSS}.json`
- Multi-model: `benchmark_multi_{YYYYMMDD_HHMMSS}.json`

**JSON structure (single trial):**

When `num_trials` is 1 (the default), results are stored as a flat list per model:

```json
{
  "timestamp": "2026-02-07T14:35:00.123456",
  "config": {
    "models": [
      {"provider": "openai", "model": "gpt-4o-mini"}
    ],
    "questions_file": "data/AI Evals New Questions.xlsx - Q&As.csv",
    "mcp_enabled": true,
    "max_iterations": 25,
    "num_trials": 1,
    "shuffle": false,
    "seed": null,
    "seed_mode": "rotate",
    "seeds": null,
    "trial_seeds": {
      "trial_1": null
    }
  },
  "summary": {
    "total_questions": 3,
    "models": {
      "openai/gpt-4o-mini": {
        "num_trials": 1,
        "passed": 2,
        "failed": 1,
        "total_tokens": 16500,
        "total_duration_seconds": 67.2
      }
    }
  },
  "results_by_model": {
    "openai/gpt-4o-mini": [
      {
        "question_id": 1,
        "category": "Market rules retrieval",
        "difficulty": "Easy",
        "question": "What fees are associated with...",
        "success": true,
        "answer": "The NYISO generation interconnection process involves...",
        "error": null,
        "metrics": {
          "input_tokens": 3200,
          "output_tokens": 1000,
          "cached_tokens": 0,
          "reasoning_tokens": 0,
          "total_tokens": 4200,
          "tool_calls": 3,
          "iterations": 5,
          "duration_seconds": 12.3,
          "latency_ms": 12300
        },
        "trace_id": "20260207_143012_123456"
      }
    ]
  }
}
```

**JSON structure (multiple trials):**

When `num_trials > 1`, results are grouped by `trial_N` keys instead of a flat list:

```json
{
  "timestamp": "2026-02-07T15:00:00.123456",
  "config": {
    "models": [
      {"provider": "openai", "model": "gpt-4o"}
    ],
    "questions_file": "data/AI Evals New Questions.xlsx - Q&As.csv",
    "mcp_enabled": true,
    "max_iterations": 25,
    "num_trials": 3,
    "shuffle": true,
    "seed": 12345,
    "seed_mode": "rotate",
    "seeds": null,
    "trial_seeds": {
      "trial_1": 12345,
      "trial_2": 12346,
      "trial_3": 12347
    }
  },
  "summary": {
    "total_questions": 3,
    "models": {
      "openai/gpt-4o": {
        "num_trials": 3,
        "passed": 9,
        "failed": 0,
        "total_tokens": 38400,
        "total_duration_seconds": 105.7
      }
    }
  },
  "results_by_model": {
    "openai/gpt-4o": {
      "trial_1": [
        {
          "question_id": 1,
          "category": "Market rules retrieval",
          "difficulty": "Easy",
          "question": "What fees are associated with...",
          "success": true,
          "answer": "The NYISO generation interconnection process involves...",
          "error": null,
          "metrics": { "total_tokens": 4200, "tool_calls": 3, "..." : "..." },
          "trace_id": "20260207_150012_123456"
        }
      ],
      "trial_2": [
        {
          "question_id": 1,
          "...": "..."
        }
      ],
      "trial_3": [
        {
          "question_id": 1,
          "...": "..."
        }
      ]
    }
  }
}
```

**Metrics glossary:**

| Metric | Description |
|---|---|
| `input_tokens` | Tokens sent to the LLM (prompt, tool schemas, conversation history) |
| `output_tokens` | Tokens generated by the LLM (reasoning, tool calls, final answer) |
| `cached_tokens` | Prompt tokens served from the provider's cache (reduces cost) |
| `reasoning_tokens` | Tokens used for internal chain-of-thought by reasoning models (o1, o3, etc.). Zero for non-reasoning models. |
| `total_tokens` | `input_tokens` + `output_tokens` |
| `tool_calls` | Number of tool invocations the agent made |
| `iterations` | Number of ReAct loop cycles (think-act-observe) |
| `duration_seconds` | Wall-clock time for the entire question |
| `latency_ms` | Cumulative LLM API latency in milliseconds |

### 11c. Trace Files (Observability Output)

When the JSON observability backend is enabled, a detailed trace file is written for each question.

**Location:** `{output_dir}/{run_name}/{provider}_{model}/`

**File naming:** `trace_q{question_id}_{timestamp}.json`

**Trace structure:**

```json
{
  "trace_id": "20260207_143012_123456",
  "timestamp": "2026-02-07T14:30:12.123456",
  "start_time": "2026-02-07T14:30:00.000000",
  "end_time": "2026-02-07T14:30:12.123456",

  "query": "What fees are associated with each decision point in the NYISO...",
  "final_answer": "The NYISO generation interconnection process involves...",
  "success": true,
  "error": null,

  "metrics": {
    "iterations": 5,
    "tool_calls_count": 3,
    "total_input_tokens": 3200,
    "total_output_tokens": 1000,
    "total_cached_tokens": 0,
    "total_tokens": 4200,
    "total_latency_ms": 12300,
    "duration_seconds": 12.3
  },

  "step_summary": {
    "total_steps": 11,
    "step_types": {"thought": 5, "action": 3, "observation": 3},
    "tool_calls": ["search", "search", "gridstatus"],
    "failed_tool_calls": [],
    "errors": []
  },

  "steps": [
    {
      "index": 0,
      "step_type": "thought",
      "timestamp": "2026-02-07T14:30:00.500000",
      "content": "I need to find NYISO interconnection fees. Let me search...",
      "tool_name": null,
      "tool_input": null,
      "tool_output": null,
      "tokens_used": 150,
      "latency_ms": 500
    },
    {
      "index": 1,
      "step_type": "action",
      "timestamp": "2026-02-07T14:30:01.000000",
      "content": null,
      "tool_name": "search",
      "tool_input": {"query": "NYISO interconnection fees manual 23"},
      "tool_output": "{\"success\": true, \"data\": ...}",
      "tokens_used": null,
      "latency_ms": 2000
    },
    {
      "index": 2,
      "step_type": "observation",
      "timestamp": "2026-02-07T14:30:03.000000",
      "content": "Search returned 5 results about NYISO fees...",
      "tool_name": "search",
      "tool_output": "{\"success\": true, \"data\": ...}",
      "tool_output_length": 4500,
      "is_error": false,
      "tokens_used": null,
      "latency_ms": null
    }
  ],

  "metadata": {
    "question_id": 1,
    "category": "Market rules retrieval",
    "difficulty": "Easy",
    "provider": "openai",
    "model": "gpt-4o-mini"
  },
  "tags": ["benchmark", "Market rules retrieval", "Easy"]
}
```

Traces are invaluable for debugging. You can inspect:

- The agent's **reasoning chain** (thought steps) to understand its approach.
- Which **tools** it called, with exact inputs and outputs.
- Where it **failed** -- error steps, failed tool calls, or hitting the iteration limit.
- **Token usage** per step to identify expensive operations.

### 11d. Directory Structure After a Run

After a complete benchmark run, your project will contain:

**Without `run_name`:**

```
benchmark_results/
  benchmark_openai_20260207_143000.json

benchmark_traces/
  benchmark_config.yaml                            # Auto-copied config
  openai_gpt-4o-mini/
    trace_q1_20260207_143012_123456.json
    trace_q2_20260207_143025_234567.json
    trace_q3_20260207_143110_345678.json
```

**With `run_name: with_tools` (multi-model):**

```
benchmark_results/
  benchmark_multi_20260207_143500.json

benchmark_traces/
  with_tools/
    benchmark_config.yaml                          # Auto-copied config
    openai_gpt-4o-mini/
      trace_q1_20260207_143012_123456.json
      trace_q2_20260207_143025_234567.json
      trace_q3_20260207_143110_345678.json
    anthropic_claude-sonnet-4-20250514/
      trace_q1_20260207_144012_456789.json
      trace_q2_20260207_144030_567890.json
      trace_q3_20260207_144100_678901.json
```

**With `num_trials: 3` (multi-trial):**

When `num_trials > 1`, traces are organized into `trial_N/` subdirectories under each model:

```
benchmark_results/
  benchmark_openai_20260207_150000.json

benchmark_traces/
  eval_run/
    benchmark_config.yaml                          # Auto-copied config
    openai_gpt-4o/
      trial_1/
        trace_q1_20260207_150012_123456.json
        trace_q2_20260207_150025_234567.json
        trace_q3_20260207_150040_345678.json
      trial_2/
        trace_q1_20260207_150112_456789.json
        trace_q2_20260207_150125_567890.json
        trace_q3_20260207_150140_678901.json
      trial_3/
        trace_q1_20260207_150212_789012.json
        trace_q2_20260207_150225_890123.json
        trace_q3_20260207_150240_901234.json
```

This structure is consumed directly by the evaluation pipeline (`scripts/run_eval.py`), which discovers trials automatically and computes per-question confidence intervals across them.

---

## 12. Common Recipes

### Compare two models on all questions

```bash
python scripts/run_benchmark.py \
  --models openai:gpt-4o anthropic:claude-sonnet-4-20250514
```

### Quick validation run (3 questions, no MCP)

```bash
python scripts/run_benchmark.py \
  --questions 1,2,3 \
  --no-mcp \
  --no-observe
```

### Run with only search and grid tools

```bash
python scripts/run_benchmark.py \
  --tools search,gridstatus
```

### Full benchmark with Langfuse tracing

```yaml
# configs/langfuse_benchmark.yaml
models:
  - provider: openai
    model: gpt-4o

questions_file: data/AI Evals New Questions.xlsx - Q&As.csv

observability:
  enabled: true
  backend: langfuse

mcp:
  enabled: true

tools:
  enabled: true
  include: []
  exclude: []
```

```bash
python scripts/run_benchmark.py -c configs/langfuse_benchmark.yaml
```

### Compare with-tools vs no-tools using `run_name`

**Run 1 -- with tools:**

```bash
python scripts/run_benchmark.py \
  -c configs/benchmark_config.yaml \
  --questions 1-5
```

With the config containing:

```yaml
observability:
  run_name: with_tools
```

**Run 2 -- without tools:**

Create a separate config or override on CLI:

```bash
python scripts/run_benchmark.py \
  --questions 1-5 \
  --tools ""
```

With the config containing:

```yaml
observability:
  run_name: no_tools
tools:
  enabled: false
```

Compare traces in `benchmark_traces/with_tools/` vs `benchmark_traces/no_tools/`.

### Run multiple trials for statistical evaluation

Running multiple independent trials per question enables confidence intervals and significance testing in the evaluation pipeline.

```bash
python scripts/run_benchmark.py \
  --provider openai \
  --model gpt-4o \
  --num-trials 3 \
  --seed-mode rotate \
  --seed 12345 \
  --questions 1-10
```

Or set it in the config file:

```yaml
agent:
  max_iterations: 25
  num_trials: 3
  shuffle: true
  seed_mode: rotate
  seed: 12345

observability:
  enabled: true
  backend: json
  output_dir: ./benchmark_traces
  run_name: eval_samples
```

After the benchmark completes, run the evaluation pipeline against the traces:

```bash
python scripts/run_eval.py \
  --run-name eval_samples \
  --model openai_gpt-4o
```

### Multi-trial multi-model comparison

Combine multiple trials with multiple models for a full statistical comparison:

```bash
python scripts/run_benchmark.py \
  --models openai:gpt-4o anthropic:claude-sonnet-4-20250514 \
  --num-trials 3
```

Then evaluate and compare:

```bash
python scripts/run_eval.py \
  --run-name eval_samples \
  --compare
```

The evaluation pipeline discovers all models and trials automatically, computes per-question confidence intervals, and runs paired significance tests between models.

### Run a single model on a specific question for debugging

```bash
python scripts/run_benchmark.py \
  --provider openai \
  --model gpt-4o-mini \
  --questions 7
```

Then inspect the trace file to see the agent's full reasoning chain.

---

## 13. Troubleshooting

### Missing LLM provider API key

**Symptom:** Error on startup or when the agent tries to call the LLM.

**Fix:** Set at least one provider API key in `.env`. For example:

```bash
OPENAI_API_KEY=sk-...
```

### Tool API key missing

**Symptom:** Warning in console like `Warning: EXA_API_KEY not set`. The tool is skipped but the benchmark continues.

**Fix:** Add the missing key to `.env`, or exclude the tool:

```bash
python scripts/run_benchmark.py --exclude-tools search
```

### Ipopt solver not found

**Symptom:** Battery optimization tool fails with a solver error.

**Fix:** Run `sudo ./install.sh` to build and install Ipopt. Or exclude the battery tool:

```bash
python scripts/run_benchmark.py --exclude-tools battery
```

### MCP connection failure

**Symptom:** `Warning: MCP unavailable: ...` in console output.

**Fix:**
- Check that `RAG_SERVER_URL` and `DATABASE_SERVER_URL` are correct in `.env`.
- Verify the MCP servers are running and accessible.
- Or disable MCP: `--no-mcp`.

### Questions file not found

**Symptom:** `Error: Questions file not found: data/...`

**Fix:** The path is relative to the project root. Make sure you are running from the `energBench/` directory:

```bash
cd /path/to/energBench
python scripts/run_benchmark.py
```

### Config file not found

**Symptom:** `ConfigurationError: Config file not found: configs/my_config.yaml`

**Fix:** If you pass an explicit config path via `--config` / `-c`, the file must exist. Check the path for typos.

### ConfigurationError on startup

**Symptom:** `Invalid benchmark configuration:` followed by a list of errors.

**Common causes:**
- Unknown provider name (typo in `provider` field).
- Missing `models` list in config. The config must include a `models` list (see [Section 6](#6-benchmark-configuration-reference)).
- Invalid question IDs (non-positive integers or non-integer values).
- `max_iterations` set to less than 1.
- `num_trials` set to less than 1.
- Invalid observability backend (must be `json` or `langfuse`).

**Fix:** Check the error messages -- they describe exactly what is wrong. Correct your config file and retry.

### Wide confidence intervals in evaluation results

**Symptom:** After running the evaluation pipeline, confidence intervals are very wide or degenerate (upper == lower).

**Fix:** Run the benchmark with more trials. Confidence intervals require multiple independent observations per question. A single trial produces degenerate intervals (the CI collapses to the single score). Three or more trials are recommended:

```yaml
agent:
  num_trials: 3
```

Or on the command line:

```bash
python scripts/run_benchmark.py --num-trials 3
```

### Agent hits max iterations without answering

**Symptom:** `[FAIL]` with `Error: Max iterations reached`.

**Fix:** Increase `max_iterations` in the config:

```yaml
agent:
  max_iterations: 40   # Default is 25
```

This may happen with complex questions that require many tool calls. Keep in mind that higher iteration limits increase token costs.
