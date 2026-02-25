from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .strategy import VALID_STRATEGIES


@dataclass
class JudgeConfig:
    """LLM judge settings."""
    provider: str = "openai"
    model: str = "gpt-5-mini"
    temperature: float = 0.0
    max_tokens: int = 4096
    reasoning_effort: str | None = None


@dataclass
class EvalConfig:
    """Evaluation pipeline configuration."""
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    results_path: Path = Path("./benchmark_traces")
    dataset_path: Path = Path("./data/eval_samples_with_answers.csv")
    output_dir: Path = Path("./evaluation_results")

    run_name: str | None = None
    models: list[str] | None = None
    questions: list[int] | None = None

    category_strategies: dict[str, str] = field(default_factory=dict)
    default_strategy: str = "attributes"

    abs_tol: float = 0.01
    rel_tol: float = 0.5

    confidence_level: float = 0.95
    significance_alpha: float = 0.05

    compare: bool = False

    log_level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR


def load_eval_config(path: Path | str | None = None, base_path: Path | None = None) -> EvalConfig:
    """Load evaluation config from a YAML file.

    Args:
        path: Path to YAML config file.  ``None`` returns defaults.
        base_path: Base directory for resolving relative paths (defaults to cwd).

    Returns:
        Parsed EvalConfig.
    """
    if base_path is None:
        base_path = Path.cwd()

    if path is None:
        return EvalConfig()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Eval config not found: {config_path}")

    with open(config_path) as f:
        data: dict = yaml.safe_load(f) or {}

    judge_data = data.get("judge", {})
    judge = JudgeConfig(
        provider=judge_data.get("provider", "openai"),
        model=judge_data.get("model", "gpt-5-mini"),
        temperature=judge_data.get("temperature", 0.0),
        max_tokens=judge_data.get("max_tokens", 4096),
        reasoning_effort=judge_data.get("reasoning_effort"),
    )

    strategy_data = data.get("strategy", {})
    category_strategies: dict[str, str] = strategy_data.get("categories", {})
    default_strategy: str = strategy_data.get("default", "attributes")

    invalid = {v for v in category_strategies.values() if v not in VALID_STRATEGIES}
    if invalid:
        raise ValueError(
            f"Invalid strategy value(s) {invalid} in config; must be one of {VALID_STRATEGIES}"
        )
    if default_strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Invalid default strategy {default_strategy!r}; must be one of {VALID_STRATEGIES}"
        )

    tolerances = data.get("tolerances", {})
    stats = data.get("statistics", {})

    models_raw = data.get("models")
    questions_raw = data.get("questions")
    if isinstance(questions_raw, list):
        questions_parsed = [int(q) for q in questions_raw]
    else:
        questions_parsed = None

    def _resolve(p: str | None, default: str) -> Path:
        raw = p if p is not None else default
        resolved = Path(raw)
        if not resolved.is_absolute():
            resolved = base_path / resolved
        return resolved

    return EvalConfig(
        judge=judge,
        results_path=_resolve(data.get("results_path"), "./benchmark_traces"),
        dataset_path=_resolve(data.get("dataset_path"), "./data/eval_samples_with_answers.csv"),
        output_dir=_resolve(data.get("output_dir"), "./evaluation_results"),
        run_name=data.get("run_name"),
        models=models_raw,
        questions=questions_parsed,
        category_strategies=category_strategies,
        default_strategy=default_strategy,
        abs_tol=tolerances.get("abs_tol", 0.01),
        rel_tol=tolerances.get("rel_tol", 0.5),
        confidence_level=stats.get("confidence_level", 0.95),
        significance_alpha=stats.get("significance_alpha", 0.05),
        compare=bool(data.get("compare", False)),
        log_level=data.get("log_level", "INFO"),
    )
