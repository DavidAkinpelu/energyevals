import json
from pathlib import Path

from openai import OpenAI

from .config import EvalConfig
from .data_loader import (
    discover_trials,
    load_benchmark_result,
    load_eval_data,
    load_ground_truth,
)
from .judges import judge_accuracy, judge_approach, judge_attributes, judge_sources
from .models import (
    EvaluationReport,
    JudgeScore,
    MetricScore,
    ModelComparison,
    QuestionEval,
    TrialEval,
)
from .stats import compare_models_paired, compute_score_statistics
from .strategy import get_strategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_approach(raw_score: int) -> float:
    """Convert 1-5 approach score to 0-1 range."""
    return (raw_score - 1) / 4.0


def _normalize_source(raw_score: int) -> float:
    """Convert 1-5 source validity score to 0-1 range."""
    return (raw_score - 1) / 4.0


def _aggregate_metrics(metrics_list: list[MetricScore]) -> MetricScore:
    """Average operational metrics across trials."""
    n = len(metrics_list)
    if n == 0:
        return MetricScore()
    return MetricScore(
        tool_calls=round(sum(m.tool_calls for m in metrics_list) / n),
        total_tokens=round(sum(m.total_tokens for m in metrics_list) / n),
        duration_seconds=sum(m.duration_seconds for m in metrics_list) / n,
    )


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _discover_model_dirs(results_path: Path, run_name: str | None) -> list[Path]:
    """Auto-discover model subdirectories under the results path."""
    base = results_path / run_name if run_name else results_path
    if not base.is_dir():
        return []
    return sorted(
        d for d in base.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def _evaluate_trial(
    client: OpenAI,
    question_num: int,
    question_text: str,
    expected_answer: str,
    suggested_steps: str,
    category: str,
    trace_base: Path,
    trial: int | None,
    trial_index: int,
    config: EvalConfig,
) -> TrialEval:
    """Run all judges for a single trial of a single question."""
    entry = load_benchmark_result(trace_base, question_num, trial)
    agent_answer = entry.answer or ""
    agent_steps = entry.steps_trace
    model = config.judge.model

    raw_approach = judge_approach(
        client, question_text, suggested_steps, agent_steps, model=model,
    )
    raw_sources = judge_sources(
        client, question_text, suggested_steps, agent_answer, model=model,
    )

    strategy = get_strategy(category, config.category_strategies, config.default_strategy)
    raw_accuracy = None
    raw_attributes = None

    if strategy == "accuracy":
        raw_accuracy = judge_accuracy(
            client, question_text, expected_answer, agent_answer,
            abs_tol=config.abs_tol, rel_tol=config.rel_tol, model=model,
        )
        accuracy_score = raw_accuracy.accuracy_score
        accuracy_reasoning = raw_accuracy.reasoning
    else:
        raw_attributes = judge_attributes(
            client, question_text, expected_answer, agent_answer,
            abs_tol=config.abs_tol, rel_tol=config.rel_tol, model=model,
        )
        accuracy_score = raw_attributes.alignment_score
        accuracy_reasoning = raw_attributes.reasoning

    return TrialEval(
        trial=trial_index,
        approach=JudgeScore(
            score=_normalize_approach(raw_approach.approach_correctness),
            reasoning=raw_approach.reasoning,
            judge_type="approach",
        ),
        accuracy=JudgeScore(
            score=accuracy_score,
            reasoning=accuracy_reasoning,
            judge_type=strategy,
        ),
        sources=JudgeScore(
            score=_normalize_source(raw_sources.source_validity),
            reasoning=raw_sources.reasoning,
            judge_type="sources",
        ),
        raw_approach=raw_approach,
        raw_accuracy=raw_accuracy,
        raw_attributes=raw_attributes,
        raw_sources=raw_sources,
        metrics=entry.metrics,
    )


def _evaluate_model(
    client: OpenAI,
    model_dir: Path,
    ground_truths: dict[int, object],
    eval_data: list[dict],
    config: EvalConfig,
) -> EvaluationReport:
    """Evaluate all questions for a single model across all trials."""
    model_name = model_dir.name
    trials = discover_trials(model_dir)
    num_trials = len(trials)

    question_nums: list[int] = []
    for row in eval_data:
        qnum = int(row["S/N"])
        if config.questions and qnum not in config.questions:
            continue
        question_nums.append(qnum)

    question_evals: list[QuestionEval] = []

    for qnum in question_nums:
        gt = ground_truths.get(qnum)
        if gt is None:
            continue

        row = eval_data[qnum - 1]
        question_text = row["Question"]
        expected_answer = row.get("Answer", "")
        suggested_steps = row.get("Approach", "")
        category = row.get("Category", "")
        difficulty = row.get("Difficulty level", "")
        strategy = get_strategy(category, config.category_strategies, config.default_strategy)

        trial_evals: list[TrialEval] = []
        for trial in trials:
            trial_index = trial if trial is not None else 1
            try:
                te = _evaluate_trial(
                    client=client,
                    question_num=qnum,
                    question_text=question_text,
                    expected_answer=expected_answer,
                    suggested_steps=suggested_steps,
                    category=category,
                    trace_base=model_dir,
                    trial=trial,
                    trial_index=trial_index,
                    config=config,
                )
                trial_evals.append(te)
            except FileNotFoundError:
                print(f"    Warning: no trace for Q{qnum} trial {trial_index}, skipping")
                continue

        if not trial_evals:
            continue

        approach_scores = [te.approach.score for te in trial_evals]
        accuracy_scores = [te.accuracy.score for te in trial_evals]
        sources_scores = [te.sources.score for te in trial_evals]

        qeval = QuestionEval(
            question_id=qnum,
            category=category,
            difficulty=difficulty,
            accuracy_strategy=strategy,
            trials=trial_evals,
            approach_stats=compute_score_statistics(approach_scores, config.confidence_level),
            accuracy_stats=compute_score_statistics(accuracy_scores, config.confidence_level),
            sources_stats=compute_score_statistics(sources_scores, config.confidence_level),
            aggregated_metrics=_aggregate_metrics([te.metrics for te in trial_evals]),
        )
        question_evals.append(qeval)

    all_approach = [q.approach_stats.mean for q in question_evals]
    all_accuracy = [q.accuracy_stats.mean for q in question_evals]
    all_sources = [q.sources_stats.mean for q in question_evals]

    all_metrics = [q.aggregated_metrics for q in question_evals]

    return EvaluationReport(
        model=model_name,
        run_name=config.run_name or "",
        num_trials=num_trials,
        questions=question_evals,
        aggregate_approach=compute_score_statistics(all_approach, config.confidence_level),
        aggregate_accuracy=compute_score_statistics(all_accuracy, config.confidence_level),
        aggregate_sources=compute_score_statistics(all_sources, config.confidence_level),
        aggregate_metrics=_aggregate_metrics(all_metrics),
    )


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------

def _compare_models(
    reports: dict[str, EvaluationReport],
    alpha: float,
) -> list[ModelComparison]:
    """Run paired significance tests between all model pairs."""
    comparisons: list[ModelComparison] = []
    model_names = sorted(reports.keys())

    for i, name_a in enumerate(model_names):
        for name_b in model_names[i + 1:]:
            report_a = reports[name_a]
            report_b = reports[name_b]

            shared_ids = set(q.question_id for q in report_a.questions) & set(
                q.question_id for q in report_b.questions
            )
            if not shared_ids:
                continue

            scores_a_map = {q.question_id: q for q in report_a.questions}
            scores_b_map = {q.question_id: q for q in report_b.questions}

            for dimension in ("approach", "accuracy", "sources"):
                vals_a = [getattr(scores_a_map[qid], f"{dimension}_stats").mean for qid in sorted(shared_ids)]
                vals_b = [getattr(scores_b_map[qid], f"{dimension}_stats").mean for qid in sorted(shared_ids)]

                cmp = compare_models_paired(vals_a, vals_b, alpha=alpha)
                cmp.model_a = name_a
                cmp.model_b = name_b
                cmp.dimension = dimension
                comparisons.append(cmp)

    return comparisons


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _write_report(report: EvaluationReport, output_dir: Path) -> None:
    """Write per-model report.json, per-trial qN.json, and summary.csv."""
    model_out = output_dir / report.model
    model_out.mkdir(parents=True, exist_ok=True)

    for qeval in report.questions:
        for te in qeval.trials:
            if report.num_trials > 1 or (te.trial != 1):
                trial_dir = model_out / f"trial_{te.trial}"
            else:
                trial_dir = model_out
            _save_json(trial_dir / f"q{qeval.question_id}.json", te.model_dump())

    _save_json(model_out / "report.json", report.model_dump())

    _write_summary_csv(report, model_out / "summary.csv")


def _write_summary_csv(report: EvaluationReport, path: Path) -> None:
    """Write a flat CSV summary of per-question scores."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id", "category", "difficulty", "strategy",
            "approach_mean", "approach_ci",
            "accuracy_mean", "accuracy_ci",
            "sources_mean", "sources_ci",
            "tool_calls", "tokens", "duration_s",
        ])
        for q in report.questions:
            writer.writerow([
                q.question_id, q.category, q.difficulty, q.accuracy_strategy,
                f"{q.approach_stats.mean:.3f}",
                f"[{q.approach_stats.ci_lower:.3f}, {q.approach_stats.ci_upper:.3f}]",
                f"{q.accuracy_stats.mean:.3f}",
                f"[{q.accuracy_stats.ci_lower:.3f}, {q.accuracy_stats.ci_upper:.3f}]",
                f"{q.sources_stats.mean:.3f}",
                f"[{q.sources_stats.ci_lower:.3f}, {q.sources_stats.ci_upper:.3f}]",
                q.aggregated_metrics.tool_calls,
                q.aggregated_metrics.total_tokens,
                f"{q.aggregated_metrics.duration_seconds:.1f}",
            ])


def _print_report(report: EvaluationReport) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'=' * 70}")
    print(f"  Evaluation Report: {report.model}")
    print(f"  Run: {report.run_name}  |  Trials: {report.num_trials}")
    print(f"{'=' * 70}")

    for q in report.questions:
        ci_app = f"[{q.approach_stats.ci_lower:.2f}, {q.approach_stats.ci_upper:.2f}]"
        ci_acc = f"[{q.accuracy_stats.ci_lower:.2f}, {q.accuracy_stats.ci_upper:.2f}]"
        ci_src = f"[{q.sources_stats.ci_lower:.2f}, {q.sources_stats.ci_upper:.2f}]"
        print(
            f"  Q{q.question_id:>2} [{q.category}] "
            f"approach={q.approach_stats.mean:.2f} {ci_app}  "
            f"accuracy={q.accuracy_stats.mean:.2f} {ci_acc}  "
            f"sources={q.sources_stats.mean:.2f} {ci_src}"
        )

    print("\n  Aggregate:")
    for label, stat in [
        ("Approach", report.aggregate_approach),
        ("Accuracy", report.aggregate_accuracy),
        ("Sources", report.aggregate_sources),
    ]:
        ci = f"[{stat.ci_lower:.2f}, {stat.ci_upper:.2f}]"
        print(f"    {label}: {stat.mean:.3f} +/- {stat.std:.3f}  CI={ci}  (n={stat.n})")

    m = report.aggregate_metrics
    print(f"    Avg tool calls: {m.tool_calls}  |  Avg tokens: {m.total_tokens}  |  Avg time: {m.duration_seconds:.1f}s")


def _print_comparisons(comparisons: list[ModelComparison]) -> None:
    """Print cross-model comparison results."""
    if not comparisons:
        return
    print(f"\n{'=' * 70}")
    print("  Cross-Model Comparisons")
    print(f"{'=' * 70}")
    for c in comparisons:
        sig = "*" if c.significant else ""
        print(
            f"  {c.model_a} vs {c.model_b} [{c.dimension}]: "
            f"p={c.p_value:.4f}{sig}  effect={c.effect_size:.3f}  "
            f"direction={c.direction}  test={c.test_name}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_evaluation(config: EvalConfig) -> dict[str, EvaluationReport]:
    """Run the full evaluation pipeline.

    1. Load ground truth from the dataset CSV.
    2. Discover model directories under ``results_path/{run_name}/``.
    3. For each model, judge every trial of every question.
    4. Aggregate scores with confidence intervals.
    5. Optionally run cross-model significance tests.
    6. Write results to ``output_dir/{run_name}/``.

    Returns:
        Mapping of model name to its EvaluationReport.
    """
    client = OpenAI()

    eval_data = load_eval_data(config.dataset_path)
    ground_truths = load_ground_truth(config.dataset_path)

    model_dirs = _discover_model_dirs(config.results_path, config.run_name)

    if config.models:
        model_dirs = [d for d in model_dirs if d.name in config.models]

    if not model_dirs:
        print(f"  No model directories found under {config.results_path / (config.run_name or '')}")
        return {}

    print(f"  Found {len(model_dirs)} model(s): {[d.name for d in model_dirs]}")

    output_base = config.output_dir
    if config.run_name:
        output_base = output_base / config.run_name
    output_base.mkdir(parents=True, exist_ok=True)

    reports: dict[str, EvaluationReport] = {}

    for model_dir in model_dirs:
        print(f"\n  Evaluating model: {model_dir.name}")
        report = _evaluate_model(client, model_dir, ground_truths, eval_data, config)
        report.config_snapshot = {
            "judge_model": config.judge.model,
            "abs_tol": config.abs_tol,
            "rel_tol": config.rel_tol,
            "confidence_level": config.confidence_level,
        }
        reports[model_dir.name] = report

        _write_report(report, output_base)
        _print_report(report)

    comparisons: list[ModelComparison] = []
    if config.compare and len(reports) >= 2:
        comparisons = _compare_models(reports, config.significance_alpha)
        _save_json(output_base / "comparison_report.json", {"comparisons": [c.model_dump() for c in comparisons]})
        _print_comparisons(comparisons)

    # Save a copy of the eval config
    if hasattr(config, "__dict__"):
        config_snapshot = {
            "judge": {"model": config.judge.model, "provider": config.judge.provider},
            "results_path": str(config.results_path),
            "dataset_path": str(config.dataset_path),
            "run_name": config.run_name,
            "abs_tol": config.abs_tol,
            "rel_tol": config.rel_tol,
        }
        _save_json(output_base / "eval_config.yaml", config_snapshot)

    return reports
