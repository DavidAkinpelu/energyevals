# Evaluation Pipeline Testing Guide

A hands-on guide for manually verifying that the LLM-judge evaluation pipeline is functional and produces correct results. Work through each section in order — later sections depend on earlier ones passing.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Import Smoke Test](#2-import-smoke-test)
3. [Config Loading](#3-config-loading)
4. [Data Loading](#4-data-loading)
5. [Strategy Routing](#5-strategy-routing)
6. [Single-Question Evaluation](#6-single-question-evaluation)
7. [Multi-Question Evaluation](#7-multi-question-evaluation)
8. [Multi-Trial Evaluation](#8-multi-trial-evaluation)
9. [Cross-Model Comparison](#9-cross-model-comparison)
10. [Output Validation](#10-output-validation)
11. [Score Sanity Checks](#11-score-sanity-checks)
12. [Judge Model Override](#12-judge-model-override)
13. [Edge Cases](#13-edge-cases)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Prerequisites

**Goal:** Confirm everything the evaluation pipeline needs is in place before running any evaluations.

### What you need

- **Benchmark traces** must already exist. These are produced by `scripts/run_benchmark.py` and live under `benchmark_traces/{model}/` (or `benchmark_traces/{run_name}/{model}/` when `run_name` is configured in the benchmark config). Each model directory contains files like `trace_q1_20260220_180033_e33e3a7d.json`. If you have multi-trial runs (`num_trials > 1`), traces will be under `trial_1/`, `trial_2/`, etc.
- **`OPENAI_API_KEY`** must be set in your `.env` file. All four judges (approach, accuracy, sources, attributes) use OpenAI structured outputs via `client.responses.parse()`.
- **Ground truth CSV** at `data/eval_samples_with_answers.csv` with columns: `S/N`, `Category`, `Question type`, `Difficulty level`, `Question`, `Answer`, `Approach`.
- **Python dependencies** installed, including `scipy` (used for statistical tests and confidence intervals).

### Steps

```bash
# Verify your virtual environment is active
source .venv/bin/activate

# Check dependencies are installed
python -c "import openai, scipy, pydantic, yaml; print('Dependencies OK')"

# Verify API key is set
python -c "import os; from dotenv import load_dotenv; load_dotenv(); assert os.getenv('OPENAI_API_KEY'), 'OPENAI_API_KEY not set'; print('API key OK')"

# Check ground truth CSV exists
ls -la data/eval_samples_with_answers.csv

# Check you have benchmark traces (replace with your actual run name and model)
ls benchmark_traces/
```

### What to check

- [ ] All four imports succeed without errors.
- [ ] `OPENAI_API_KEY` is set and non-empty.
- [ ] `data/eval_samples_with_answers.csv` exists and is non-empty.
- [ ] At least one model directory exists under `benchmark_traces/`.

---

## 2. Import Smoke Test

**Goal:** Verify the evaluation module and all its submodules import cleanly.

### Steps

```bash
python -c "
from energbench.evaluation import (
    run_evaluation,
    load_eval_config,
    load_eval_data,
    load_ground_truth,
    judge_approach,
    judge_accuracy,
    judge_sources,
    judge_attributes,
    compute_score_statistics,
    compare_models_paired,
    get_strategy,
)
print('All evaluation imports OK')
"
```

### What to check

- [ ] Prints `All evaluation imports OK` with no `ModuleNotFoundError` or `ImportError`.

---

## 3. Config Loading

**Goal:** Verify config parsing works for both defaults and YAML files.

### Steps

```bash
# 3a. Load default config (no file)
python -c "
from energbench.evaluation.config import load_eval_config
c = load_eval_config()
print(f'Judge model:       {c.judge.model}')
print(f'Judge provider:    {c.judge.provider}')
print(f'Judge temperature: {c.judge.temperature}')
print(f'Results path:      {c.results_path}')
print(f'Dataset path:      {c.dataset_path}')
print(f'Output dir:        {c.output_dir}')
print(f'Abs tolerance:     {c.abs_tol}')
print(f'Rel tolerance:     {c.rel_tol}')
print(f'Confidence level:  {c.confidence_level}')
print(f'Sig alpha:         {c.significance_alpha}')
print(f'Compare:           {c.compare}')
print(f'Category strategies: {c.category_strategies}')
print(f'Default strategy:    {c.default_strategy}')
"

# 3b. Load from YAML config
python -c "
from pathlib import Path
from energbench.evaluation.config import load_eval_config
c = load_eval_config('configs/eval_config.yaml', base_path=Path('.'))
print(f'Judge model:         {c.judge.model}')
print(f'Results path:        {c.results_path}')
print(f'Dataset path:        {c.dataset_path}')
print(f'Abs tolerance:       {c.abs_tol}')
print(f'Rel tolerance:       {c.rel_tol}')
print(f'Category strategies: {c.category_strategies}')
print(f'Default strategy:    {c.default_strategy}')
"
```

### What to check

- [ ] Default config: judge model is `gpt-4o`, provider is `openai`, temperature is `0.0`.
- [ ] Default config: `abs_tol=0.01`, `rel_tol=0.5`, `confidence_level=0.95`.
- [ ] Default config: `category_strategies` is `{}` (empty dict), `default_strategy` is `"attributes"`.
- [ ] YAML config: paths resolve relative to the base path (not as raw strings like `./benchmark_traces`).
- [ ] YAML config: tolerances and judge settings match `configs/eval_config.yaml`.
- [ ] YAML config: `category_strategies` matches the `strategy.categories` section in the YAML (e.g. `{"Market data retrieval and analysis": "accuracy"}`).
- [ ] YAML config: `default_strategy` is `"attributes"`.

---

## 4. Data Loading

**Goal:** Verify the CSV dataset and benchmark trace files load correctly.

### 4a. Load evaluation dataset

```bash
python -c "
from energbench.evaluation.data_loader import load_eval_data
rows = load_eval_data('data/eval_samples_with_answers.csv')
print(f'Loaded {len(rows)} questions')
print(f'Columns: {list(rows[0].keys())}')
print(f'Q1 category: {rows[0][\"Category\"]}')
print(f'Q1 has answer: {bool(rows[0].get(\"Answer\"))}')
print(f'Q1 has approach: {bool(rows[0].get(\"Approach\"))}')
"
```

### What to check

- [ ] Number of questions matches the CSV row count.
- [ ] Columns include `S/N`, `Category`, `Question type`, `Difficulty level`, `Question`, `Answer`, `Approach`.
- [ ] At least one row has non-empty `Answer` and `Approach` fields.

### 4b. Load ground truth

```bash
python -c "
from energbench.evaluation.data_loader import load_ground_truth
gt = load_ground_truth('data/eval_samples_with_answers.csv')
print(f'Ground truths loaded: {len(gt)}')
for qnum in sorted(list(gt.keys())[:3]):
    g = gt[qnum]
    print(f'  Q{qnum}: category={g.category}, answer_len={len(g.answer)}, approach_len={len(g.approach)}')
"
```

### What to check

- [ ] Ground truth dict is keyed by integer question numbers (matching `S/N`).
- [ ] Each entry has non-empty `answer`, `approach`, and `category`.

### 4c. Load a benchmark trace file

Replace the path below with an actual trace file from your benchmark run:

```bash
python -c "
from pathlib import Path
from energbench.evaluation.data_loader import load_benchmark_result

# Adjust these to match your actual trace directory and question number.
# When run_name is configured: Path('benchmark_traces/{run_name}/{model}')
# When run_name is NOT configured: Path('benchmark_traces/{model}')
trace_base = Path('benchmark_traces/openai_gpt-4o-mini')
question_num = 1
trial = 1  # Use an int matching trial_N/ subdirectory, or None when no trial dirs exist

entry = load_benchmark_result(trace_base, question_num, trial)
print(f'Answer length:    {len(entry.answer) if entry.answer else 0}')
print(f'Steps trace len:  {len(entry.steps_trace)}')
print(f'Tool calls:       {entry.metrics.tool_calls}')
print(f'Total tokens:     {entry.metrics.total_tokens}')
print(f'Duration (s):     {entry.metrics.duration_seconds}')
print(f'LLM latency (ms): {entry.metrics.latency.llm_thinking_ms}')
print(f'Tool latency (ms):{entry.metrics.latency.tool_execution_ms}')
"
```

### What to check

- [ ] `answer` is a non-empty string (the agent's final answer).
- [ ] `steps_trace` is a non-empty string representation of the agent's action steps.
- [ ] `metrics.tool_calls` is a positive integer (agent used at least one tool).
- [ ] `metrics.total_tokens` is a positive integer.
- [ ] `metrics.duration_seconds` is a positive number.

---

## 5. Strategy Routing

**Goal:** Verify the config-driven category-to-judge-strategy mapping works correctly. Strategy routing no longer uses a hardcoded map -- it is fully determined by the `strategy` section in the YAML config (or the `category_strategies` / `default_strategy` fields on `EvalConfig`).

### Steps

```bash
python -c "
from energbench.evaluation.strategy import get_strategy, has_strategy

# Simulate the category map a user would set in their config
categories = {'Market data retrieval and analysis': 'accuracy'}

# Category present in the map -> uses mapped strategy
print(get_strategy('Market data retrieval and analysis', categories))  # should print: accuracy
print(has_strategy('Market data retrieval and analysis', categories))  # should print: True

# Categories NOT in the map -> fall back to default strategy
print(get_strategy('Market rules retrieval', categories))               # should print: attributes
print(get_strategy('Policy and regulatory analysis', categories))       # should print: attributes
print(get_strategy('Project and asset development analysis', categories))  # should print: attributes
print(has_strategy('Market rules retrieval', categories))               # should print: False

# Empty map -> everything uses the default
print(get_strategy('Market data retrieval and analysis', {}))           # should print: attributes

# Custom category added to the map
custom = {'My new category': 'accuracy'}
print(get_strategy('My new category', custom))                          # should print: accuracy
print(get_strategy('Other category', custom))                           # should print: attributes

# Custom default strategy
print(get_strategy('Anything', {}, 'accuracy'))                         # should print: accuracy
"
```

### What to check

- [ ] Categories present in the map return the mapped strategy value.
- [ ] Categories absent from the map return the default strategy (`"attributes"` unless overridden).
- [ ] `has_strategy` returns `True` only for categories explicitly in the map.
- [ ] An empty map causes all categories to use the default.
- [ ] A custom `default_strategy` argument is respected when provided.

---

## 6. Single-Question Evaluation

**Goal:** Run the full evaluation pipeline for one question against one model and verify it produces correct output.

### Steps

```bash
# Replace with your actual model directory name (add --run-name only if one was configured)
python scripts/run_eval.py \
  --model openai_gpt-4o-mini \
  --questions 1
```

### What to check

- [ ] The script prints the evaluation header with judge model, paths, and filters.
- [ ] It finds and reports the model directory.
- [ ] A per-question line appears with approach, accuracy, and sources scores.
- [ ] Aggregate statistics are printed (mean, std, CI).
- [ ] No unhandled exceptions or tracebacks.

### Check output files

```bash
# Verify output files were created
ls -la evaluation_results/openai_gpt-4o-mini/

# Inspect the report
python -c "
import json
with open('evaluation_results/openai_gpt-4o-mini/report.json') as f:
    report = json.load(f)
print(f'Model: {report[\"model\"]}')
print(f'Questions evaluated: {len(report[\"questions\"])}')
print(f'Aggregate approach mean: {report[\"aggregate_approach\"][\"mean\"]}')
"

# Inspect the per-question JSON (in trial_1/ when multi-trial)
python -c "
import json
with open('evaluation_results/openai_gpt-4o-mini/trial_1/q1.json') as f:
    q = json.load(f)
print(f'Trial: {q[\"trial\"]}')
print(f'Approach score: {q[\"approach\"][\"score\"]}')
print(f'Accuracy score: {q[\"accuracy\"][\"score\"]}')
print(f'Sources score:  {q[\"sources\"][\"score\"]}')
print(f'Approach reasoning: {q[\"approach\"][\"reasoning\"][:100]}...')
"
```

### What to check

- [ ] `report.json` exists and is valid JSON.
- [ ] `summary.csv` exists with correct column headers.
- [ ] `q1.json` exists with `approach`, `accuracy`, `sources` scores and reasoning.
- [ ] Scores use expected ranges: `approach`/`sources` in [1.0, 5.0], `accuracy` in [0.0, 1.0].

---

## 7. Multi-Question Evaluation

**Goal:** Run the evaluation across all questions for one model.

### Steps

```bash
python scripts/run_eval.py \
  --model openai_gpt-4o-mini
```

### What to check

```bash
# Check summary CSV has a row per question
python -c "
import csv
with open('evaluation_results/openai_gpt-4o-mini/summary.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f'Questions in summary: {len(rows)}')
for row in rows:
    print(f'  Q{row[\"question_id\"]}: approach={row[\"approach_mean\"]}, accuracy={row[\"accuracy_mean\"]}, sources={row[\"sources_mean\"]}')
"
```

- [ ] Every question that has a trace file appears in `summary.csv`.
- [ ] Questions with missing traces are skipped with a warning (not a crash).
- [ ] Aggregate statistics in `report.json` reflect the mean across all evaluated questions.
- [ ] The `strategy` column matches what is configured in `strategy.categories` in the eval config YAML (`accuracy` for mapped categories, `attributes` for everything else by default).

---

## 8. Multi-Trial Evaluation

**Goal:** Verify trial discovery and aggregation when benchmark traces contain multiple trials.

### Prerequisites

Benchmark traces must be organized as:

```
benchmark_traces/{model}/
  trial_1/
    trace_q1_*.json
    trace_q2_*.json
  trial_2/
    trace_q1_*.json
    trace_q2_*.json
```

When `run_name` is configured in the benchmark config, an extra level is inserted: `benchmark_traces/{run_name}/{model}/trial_N/...`.

This structure is produced automatically by `run_benchmark.py` when `num_trials > 1` in the benchmark config.

### Steps

```bash
python scripts/run_eval.py \
  --model openai_gpt-4o-mini
```

### What to check

```bash
# Verify per-trial output files
ls evaluation_results/openai_gpt-4o-mini/trial_*/

# Check confidence intervals are non-degenerate
python -c "
import json
with open('evaluation_results/openai_gpt-4o-mini/report.json') as f:
    report = json.load(f)
print(f'Num trials: {report[\"num_trials\"]}')
for q in report['questions']:
    stats = q['approach_stats']
    ci_width = stats['ci_upper'] - stats['ci_lower']
    print(f'  Q{q[\"question_id\"]}: approach mean={stats[\"mean\"]:.3f}, CI width={ci_width:.3f}, n={stats[\"n\"]}')
"
```

- [ ] `num_trials` in the report matches the number of `trial_N/` directories.
- [ ] Per-trial `qN.json` files exist under each `trial_N/` subdirectory.
- [ ] When `n > 1`, confidence intervals have non-zero width (`ci_lower < ci_upper`).
- [ ] When `n == 1`, confidence intervals collapse to the single score (`ci_lower == ci_upper`).
- [ ] `std` is 0 for single-trial questions and >= 0 for multi-trial.

---

## 9. Cross-Model Comparison

**Goal:** Verify paired significance testing between two or more models.

### Prerequisites

Benchmark traces must exist for at least two models under the same trace directory.

### Steps

```bash
python scripts/run_eval.py \
  --compare
```

### What to check

```bash
# Inspect comparison report
python -c "
import json
with open('evaluation_results/comparison_report.json') as f:
    data = json.load(f)
print(f'Comparisons: {len(data[\"comparisons\"])}')
for c in data['comparisons']:
    sig = '*' if c['significant'] else ''
    print(f'  {c[\"model_a\"]} vs {c[\"model_b\"]} [{c[\"dimension\"]}]: p={c[\"p_value\"]:.4f}{sig} test={c[\"test_name\"]}')
"
```

- [ ] `comparison_report.json` is created under the output directory.
- [ ] There is one comparison entry per (model_pair, dimension) combination — 3 dimensions (approach, accuracy, sources) per pair.
- [ ] `test_name` is `wilcoxon` when there are >= 6 questions with non-zero score differences, `paired_t` otherwise, or `exact_tie` when all scores are identical.
- [ ] `direction` is one of `a>b`, `b>a`, or `equal`.
- [ ] `p_value` is in [0.0, 1.0].
- [ ] `significant` is `True` only when `p_value < 0.05` (the default alpha).
- [ ] The cross-model comparison table is printed to stdout.

---

## 10. Output Validation

**Goal:** Manually inspect the structure and content of all output files.

### 10a. report.json

```bash
python -c "
import json
with open('evaluation_results/openai_gpt-4o-mini/report.json') as f:
    r = json.load(f)
required_keys = ['model', 'run_name', 'num_trials', 'questions',
                 'aggregate_approach', 'aggregate_accuracy', 'aggregate_sources',
                 'aggregate_metrics', 'config_snapshot']
for k in required_keys:
    assert k in r, f'Missing key: {k}'
    print(f'  {k}: present')
print(f'Config snapshot: {r[\"config_snapshot\"]}')
"
```

- [ ] All required top-level keys are present.
- [ ] `config_snapshot` contains `judge_model`, `abs_tol`, `rel_tol`, `confidence_level`.

### 10b. summary.csv

```bash
head -5 evaluation_results/openai_gpt-4o-mini/summary.csv
```

- [ ] Header row: `question_id,category,difficulty,strategy,approach_mean,approach_ci,accuracy_mean,accuracy_ci,sources_mean,sources_ci,tool_calls,tokens,duration_s`.
- [ ] One data row per evaluated question.
- [ ] `approach_mean` and `sources_mean` are in [1.0, 5.0].
- [ ] `accuracy_mean` is in [0.0, 1.0].

### 10c. Per-trial qN.json

```bash
python -c "
import json
with open('evaluation_results/openai_gpt-4o-mini/trial_1/q1.json') as f:
    q = json.load(f)
for dim in ['approach', 'accuracy', 'sources']:
    s = q[dim]
    print(f'{dim}: score={s[\"score\"]}, judge_type={s[\"judge_type\"]}')
    print(f'  reasoning: {s[\"reasoning\"][:80]}...')
"
```

- [ ] Each dimension has `score`, `reasoning` (non-empty string), and `judge_type`.
- [ ] `approach.score` and `sources.score` are in [1.0, 5.0], `accuracy.score` is in [0.0, 1.0].
- [ ] `judge_type` for approach is `"approach"`, for sources is `"sources"`, for accuracy is either `"accuracy"` or `"attributes"` depending on category.

---

## 11. Score Sanity Checks

**Goal:** Verify that all scores are mathematically valid and within expected ranges.

### Steps

```bash
python -c "
import json
with open('evaluation_results/openai_gpt-4o-mini/report.json') as f:
    r = json.load(f)

errors = []
for q in r['questions']:
    qid = q['question_id']
    ranges = {
        'approach': (1.0, 5.0),
        'accuracy': (0.0, 1.0),
        'sources': (1.0, 5.0),
    }

    for dim in ['approach', 'accuracy', 'sources']:
        stats = q[f'{dim}_stats']
        score_mean = stats['mean']
        ci_lo = stats['ci_lower']
        ci_hi = stats['ci_upper']
        std = stats['std']
        lo, hi = ranges[dim]

        if not (lo <= score_mean <= hi):
            errors.append(f'Q{qid} {dim}: mean {score_mean} out of [{lo},{hi}]')
        if std < 0:
            errors.append(f'Q{qid} {dim}: negative std {std}')
        if ci_lo > score_mean:
            errors.append(f'Q{qid} {dim}: ci_lower {ci_lo} > mean {score_mean}')
        if ci_hi < score_mean:
            errors.append(f'Q{qid} {dim}: ci_upper {ci_hi} < mean {score_mean}')

    for t in q['trials']:
        for dim in ['approach', 'accuracy', 'sources']:
            s = t[dim]['score']
            lo, hi = ranges[dim]
            if not (lo <= s <= hi):
                errors.append(
                    f'Q{qid} trial {t[\"trial\"]} {dim}: score {s} out of [{lo},{hi}]'
                )

if errors:
    print(f'FAILED: {len(errors)} errors found')
    for e in errors:
        print(f'  - {e}')
else:
    print('All scores valid')
"
```

### What to check

- [ ] `approach` and `sources` scores are in [1.0, 5.0].
- [ ] `accuracy` scores (from `judge_accuracy`) are continuous in [0.0, 1.0].
- [ ] Attribute alignment scores (from `judge_attributes`) are continuous in [0.0, 1.0].
- [ ] For every question: `ci_lower <= mean <= ci_upper`.
- [ ] For every question: `std >= 0`.

---

## 12. Judge Model Override

**Goal:** Verify the judge LLM model can be swapped via CLI.

### Steps

```bash
python scripts/run_eval.py \
  --model openai_gpt-4o-mini \
  --questions 1 \
  --judge-model gpt-4o-mini
```

### What to check

- [ ] The header prints `Judge model: gpt-4o-mini` (not `gpt-4o`).
- [ ] Evaluation completes without errors.
- [ ] Scores are produced (they may differ from the default judge, which is expected).

---

## 13. Edge Cases

**Goal:** Verify the pipeline handles error conditions gracefully.

### 13a. Missing trace file

Run evaluation for a question number that has no trace file:

```bash
python scripts/run_eval.py \
  --model openai_gpt-4o-mini \
  --questions 99
```

- [ ] Prints a warning like `Warning: no trace for Q99 trial 1, skipping`.
- [ ] Does not crash with an unhandled exception.
- [ ] Report is still generated (with zero questions if none had traces).

### 13b. No model directories found

```bash
python scripts/run_eval.py \
  --run-name nonexistent_run \
  --model fake_model
```

- [ ] Prints `No model directories found under ...`.
- [ ] Exits with code 1 (no reports generated).
- [ ] No traceback.

### 13c. All questions (no filter)

```bash
python scripts/run_eval.py \
  --model openai_gpt-4o-mini
```

- [ ] Omitting `--questions` evaluates all questions that have traces.
- [ ] Output matches what you get when explicitly listing all question IDs.

### 13d. Config file not found

```bash
python scripts/run_eval.py --config nonexistent.yaml
```

- [ ] Falls back to default config (the script checks `config.exists()` before loading).
- [ ] No crash.

---

## 14. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'openai'` | Missing dependency | `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'scipy'` | Missing scipy | `pip install scipy` |
| `openai.AuthenticationError` or HTTP 401 | Invalid or missing API key | Check `OPENAI_API_KEY` in `.env` |
| `FileNotFoundError: Dataset CSV not found` | Wrong dataset path | Verify `data/eval_samples_with_answers.csv` exists, or pass `--dataset-path` |
| `No model directories found under ...` | Wrong run name or results path | Check `benchmark_traces/` directory structure, verify `--run-name` matches |
| `No trace file matching 'trace_q{N}_*.json'` | Missing trace for that question | Run the benchmark for that question first, or exclude it with `--questions` |
| Scores are all 0.0 or all 1.0 | Judge may not be evaluating correctly | Try a different `--judge-model`, inspect the `reasoning` field in `qN.json` |
| `comparison_report.json` not created | Forgot `--compare` flag or only one model | Pass `--compare` and ensure traces exist for 2+ models |
| `ValidationError` from Pydantic | Judge returned out-of-schema values | Check OpenAI API status; retry with `--judge-model gpt-4o` |
| Very wide confidence intervals | Too few trials | Run the benchmark with more trials (`num_trials` in benchmark config) |
| `KeyError: 'S/N'` when loading CSV | Wrong CSV format or encoding | Verify the CSV has a header row with `S/N` as the first column |
