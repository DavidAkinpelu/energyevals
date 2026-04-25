# Reproducing the Paper Results

This document explains how to reproduce the benchmark experiments and evaluations reported in the paper.

## Pre-computed Artifacts

All raw outputs from our experiments are in `public_release/public_release/`. These artifacts allow you to inspect and analyze results without re-running any experiments.

### Datasets

| File | Description |
|------|-------------|
| `evals_dataset_full_30.csv` | 30 evaluation questions with ground-truth answers and approaches |
| `evals_dataset_questions_only_212.csv` | Full question bank (212 questions, no answers) |

The full benchmark uses 212 questions total: 23 questions requiring the GridStatus API (`evals_full_dataset_with_gridstatus.csv`) and 189 questions that do not (`evals_full_dataset_without_gridstatus.csv`). Of these, 30 were selected for evaluation with ground-truth answers.

### Traces

Full agent execution traces capturing every ReAct step (thought, action, observation), tool inputs/outputs, token usage, and latency. One JSON file per question per model. The public release contains traces for the 30 evaluated questions, not all 212.

| Directory | Description |
|-----------|-------------|
| `traces/with_tools_final/` | 30 evaluated questions x 7 models (210 traces) -- full tool suite |
| `traces/without_tools_final/` | 7 evaluated questions x 7 models (49 traces) -- limited tools |

Each trace directory is organized by model (e.g., `anthropic_claude-sonnet-4-6/`, `openai_gpt-5.2/`).

### Evaluations

LLM judge scores for each model's responses. Each evaluation JSON contains approach, accuracy, and source validity scores with reasoning.

| Directory | Description |
|-----------|-------------|
| `evaluations/evaluations_with_tools_final/` | Scores for with-tools experiment |
| `evaluations/evaluations_without_tools_final/` | Scores for without-tools experiment |

Each model directory also contains a `summary.csv` with aggregated results.

## Models Evaluated

| Provider | Model | Reasoning effort |
|----------|-------|-----------------|
| Anthropic | claude-sonnet-4-6 | low |
| OpenAI | gpt-5.2 | low |
| OpenAI | gpt-5-mini | low |
| Google | gemini-3.1-pro-preview | low |
| DeepInfra | deepseek-ai/DeepSeek-V3.2 | -- |
| DeepInfra | moonshotai/Kimi-K2.5 | -- |
| DeepInfra | Qwen/Qwen3-Max-Thinking | -- |

Anthropic, OpenAI, and Google models use `effort: low` to control reasoning token usage. DeepInfra models do not support this parameter.

## Setup

```bash
# 1. Install system dependencies (Ipopt solver)
sudo ./install.sh

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env with your API keys (see required keys below)
```

### Required API Keys

**LLM providers** (all four needed to run all models):
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `DEEPINFRA_API_KEY`

**Tool APIs** (needed for with-tools experiments):
- `EXA_API_KEY` (web search)
- `GRIDSTATUS_API_KEY` (energy market data)
- `OPEN_EI_API_KEY` (utility tariffs)
- `RENEWABLES_NINJA_API_KEY` (solar/wind profiles)
- `OPENWEATHER_API_KEY` (weather data)

**MCP servers** (needed for RAG and database tools):
- `RAG_SERVER_URL=https://energyevals-rag-mcp.tume.ai/sse`
- `DATABASE_SERVER_URL=https://energyevals-db-mcp.tume.ai/sse`

## Running the Experiments

There are three benchmark configurations and one evaluation configuration.

### Experiment 1: With GridStatus (23 questions)

Runs all 7 models against the 23 questions requiring the GridStatus API, with the full tool suite (37 tools including GridStatus).

```bash
python scripts/run_benchmark.py --config configs/benchmark_config_with_gridstatus.yaml
```

- **Config**: `configs/benchmark_config_with_gridstatus.yaml`
- **Dataset**: `data/evals_full_dataset_with_gridstatus.csv` (23 questions)
- **Output**: `benchmark_traces/with_tools_final/`

### Experiment 2: Without Tools (30 questions)

Runs all 7 models with only system, search, and weather tools (energy-specific tools excluded).

```bash
python scripts/run_benchmark.py --config configs/benchmark_config_without_tools.yaml
```

- **Config**: `configs/benchmark_config_without_tools.yaml`
- **Dataset**: `data/evals_full_dataset_without_tools.csv` (30 questions)
- **Output**: `benchmark_traces/without_tools_final/`

### Experiment 3: Without GridStatus (189 questions)

Runs all 7 models with the full tool suite except the GridStatus API tool. Uses the larger question set that does not depend on GridStatus data.

```bash
python scripts/run_benchmark.py --config configs/benchmark_config_without_gridstatus.yaml
```

- **Config**: `configs/benchmark_config_without_gridstatus.yaml`
- **Dataset**: `data/evals_full_dataset_without_gridstatus.csv` (189 questions)
- **Output**: `benchmark_traces/without_gridstatus/`

### Running the LLM Judge Evaluation

After benchmark traces are generated, run the evaluation pipeline to score each model's responses:

```bash
# Evaluate with-gridstatus traces (default eval_config.yaml paths)
python scripts/run_eval.py --config configs/eval_config.yaml

# Evaluate without-tools traces (update paths in eval_config.yaml first):
#   results_path: ./benchmark_traces/without_tools_final
#   dataset_path: ./data/evals_full_dataset_without_tools.csv
#   output_dir: ./evaluation_results/without_tools
python scripts/run_eval.py --config configs/eval_config.yaml
```

- **Config**: `configs/eval_config.yaml`
- **Judge model**: OpenAI gpt-5-mini (temperature=0.0, reasoning_effort=low)

## Notes on Reproducibility

- **API costs**: Running all 7 models across all experiments involves significant API calls. Budget accordingly.
- **Non-determinism**: LLM outputs are inherently stochastic. Results will vary between runs even with the same seed due to model-side sampling. The shuffle seed (`seed: 101`) controls question ordering only.
- **MCP servers**: The RAG and database MCP servers must be accessible. The public endpoints are `https://energyevals-rag-mcp.tume.ai/sse` (document retrieval) and `https://energyevals-db-mcp.tume.ai/sse` (SQL database). These provide the document retrieval and SQL database tools used in the with-tools experiments.
- **Tool API availability**: Some tool APIs (GridStatus, Renewables.ninja) have rate limits and may return different data over time as their underlying datasets update.
