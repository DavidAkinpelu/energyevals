import json
from datetime import datetime
from pathlib import Path

from .config import BenchmarkConfig
from .models import BenchmarkResult


def _serialize_result(r: BenchmarkResult) -> dict:
    return {
        "question_id": r.question.id,
        "category": r.question.category,
        "difficulty": r.question.difficulty,
        "question": r.question.question,
        "success": r.success,
        "answer": r.answer,
        "error": r.error,
        "metrics": r.metrics,
        "trace_id": r.trace_id,
    }


def save_results(
    all_results: dict[str, dict[int, list[BenchmarkResult]]],
    config: BenchmarkConfig,
    seed: int | None = None,
) -> Path:
    """Save benchmark results to JSON.

    Args:
        all_results: Dict mapping model display name -> trial number -> list of results.
        config: Benchmark configuration.

    Returns:
        Path to the saved results file.
    """
    config.results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if len(config.models) > 1:
        output_path = config.results_dir / f"benchmark_multi_{timestamp}.json"
    else:
        m = config.models[0]
        output_path = config.results_dir / f"benchmark_{m.provider}_{timestamp}.json"

    model_summaries = {}
    for model_name, trials in all_results.items():
        all_model = [r for trial_results in trials.values() for r in trial_results]
        model_summaries[model_name] = {
            "num_trials": len(trials),
            "passed": sum(1 for r in all_model if r.success),
            "failed": sum(1 for r in all_model if not r.success),
            "total_tokens": sum(r.metrics.get("total_tokens", 0) for r in all_model),
            "total_duration_seconds": sum(
                r.metrics.get("duration_seconds", 0) for r in all_model
            ),
        }

    results_by_model: dict[str, dict | list] = {}
    for model_name, trials in all_results.items():
        if config.num_trials > 1:
            results_by_model[model_name] = {
                f"trial_{trial_num}": [_serialize_result(r) for r in results]
                for trial_num, results in trials.items()
            }
        else:
            first_trial = next(iter(trials.values()), [])
            results_by_model[model_name] = [_serialize_result(r) for r in first_trial]

    first_trials = next(iter(all_results.values()), {})
    first_results = next(iter(first_trials.values()), [])

    data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": [
                {"provider": m.provider, "model": m.model} for m in config.models
            ],
            "questions_file": str(config.questions_file),
            "mcp_enabled": config.mcp_enabled,
            "max_iterations": config.max_iterations,
            "num_trials": config.num_trials,
            "shuffle": config.shuffle,
            "seed": seed,
        },
        "summary": {
            "total_questions": len(first_results),
            "models": model_summaries,
        },
        "results_by_model": results_by_model,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    output_path.chmod(0o600)

    return output_path
