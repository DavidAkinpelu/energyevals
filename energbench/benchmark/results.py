from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import BenchmarkConfig
    from .models import BenchmarkResult


def save_results(
    all_results: dict[str, list[BenchmarkResult]],
    config: BenchmarkConfig,
) -> Path:
    """Save benchmark results to JSON.

    Args:
        all_results: Dict mapping model display name to list of results.
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
    for model_name, results in all_results.items():
        model_summaries[model_name] = {
            "passed": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "total_tokens": sum(r.metrics.get("total_tokens", 0) for r in results),
            "total_duration_seconds": sum(
                r.metrics.get("duration_seconds", 0) for r in results
            ),
        }

    results_by_model = {}
    for model_name, results in all_results.items():
        results_by_model[model_name] = [
            {
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
            for r in results
        ]

    first_results = next(iter(all_results.values()), [])

    data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": [
                {"provider": m.provider, "model": m.model} for m in config.models
            ],
            "questions_file": str(config.questions_file),
            "mcp_enabled": config.mcp_enabled,
            "max_iterations": config.max_iterations,
        },
        "summary": {
            "total_questions": len(first_results),
            "models": model_summaries,
        },
        "results_by_model": results_by_model,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path
