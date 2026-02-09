from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from energbench.agent.schema import ModelSpec
from energbench.benchmark.constants import DEFAULT_MAX_ITERATIONS
from energbench.core.errors import ConfigurationError
from energbench.core.types import ensure_path

if TYPE_CHECKING:
    pass

DEFAULT_CONFIG = {
    "models": [
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
        }
    ],
    "questions_file": "data/AI Evals New Questions.xlsx - Q&As.csv",
    "questions": None,
    "observability": {
        "enabled": True,
        "backend": "json",
        "output_dir": "./benchmark_traces",
        "run_name": None,
    },
    "mcp": {
        "enabled": True,
    },
    "agent": {
        "max_iterations": DEFAULT_MAX_ITERATIONS,
    },
    "output": {
        "results_dir": "./benchmark_results",
        "save_answers": True,
    },
}

PROVIDERS = {
    "openai": {
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o", "gpt-4o-mini"],
    },
    "anthropic": {
        "default_model": "claude-sonnet-4-20250514",
        "models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
    },
    "google": {
        "default_model": "gemini-2.0-flash",
        "models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    },
    "deepinfra": {
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-405B-Instruct",
        ],
    },
}


@dataclass
class ToolsConfig:
    """Tool selection configuration."""

    enabled: bool = True
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    models: list[ModelSpec]
    questions_file: Path
    questions: list[int] | None
    observability_enabled: bool
    observability_backend: str
    observability_output_dir: Path
    observability_run_name: str | None
    mcp_enabled: bool
    max_iterations: int
    results_dir: Path
    save_answers: bool
    tools_config: ToolsConfig = field(default_factory=ToolsConfig)
    config_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.questions_file = ensure_path(self.questions_file)
        self.observability_output_dir = ensure_path(self.observability_output_dir)
        self.results_dir = ensure_path(self.results_dir)

        errors = self.validate()
        if errors:
            raise ConfigurationError(
                "Invalid benchmark configuration:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if not self.models:
            errors.append("At least one model must be specified")

        for model in self.models:
            if not model.provider:
                errors.append(f"Model provider is required: {model}")
            if not model.model:
                errors.append(f"Model name is required: {model}")
            if model.provider not in PROVIDERS:
                errors.append(
                    f"Unknown provider '{model.provider}'. "
                    f"Available: {', '.join(PROVIDERS.keys())}"
                )

        if not self.questions_file.exists():
            errors.append(f"Questions file not found: {self.questions_file}")

        valid_backends = ["json", "langfuse"]
        if self.observability_backend not in valid_backends:
            errors.append(
                f"Invalid observability backend '{self.observability_backend}'. "
                f"Available: {', '.join(valid_backends)}"
            )

        if self.max_iterations < 1:
            errors.append(f"max_iterations must be at least 1, got {self.max_iterations}")

        if self.questions is not None:
            if not isinstance(self.questions, list):
                errors.append(f"questions must be a list, got {type(self.questions).__name__}")
            elif not all(isinstance(q, int) for q in self.questions):
                errors.append("All question IDs must be integers")
            elif not all(q > 0 for q in self.questions):
                errors.append("All question IDs must be positive")

        return errors

    @classmethod
    def from_dict(cls, data: dict, base_path: Path) -> BenchmarkConfig:
        """Create config from dictionary."""
        obs = data.get("observability", {})
        mcp = data.get("mcp", {})
        agent = data.get("agent", {})
        output = data.get("output", {})
        tools = data.get("tools", {})

        if "models" not in data:
            raise ConfigurationError(
                "Configuration must include a 'models' list. "
                "Example:\n  models:\n    - provider: openai\n      model: gpt-4o-mini"
            )

        models = [
            ModelSpec(
                provider=m["provider"],
                model=m["model"],
                is_reasoning_model=m.get("is_reasoning_model"),
            )
            for m in data["models"]
        ]

        questions = data.get("questions")
        if questions:
            questions = cls._parse_questions(questions)

        questions_file = base_path / str(data.get(
            "questions_file", DEFAULT_CONFIG["questions_file"]
        ))

        tools_config = ToolsConfig(
            enabled=tools.get("enabled", True),
            include=tools.get("include", []),
            exclude=tools.get("exclude", []),
        )

        return cls(
            models=models,
            questions_file=questions_file,
            questions=questions,
            observability_enabled=obs.get("enabled", True),
            observability_backend=obs.get("backend", "json"),
            observability_output_dir=Path(obs.get("output_dir", "./benchmark_traces")),
            observability_run_name=obs.get("run_name"),
            mcp_enabled=mcp.get("enabled", True),
            max_iterations=agent.get("max_iterations", DEFAULT_MAX_ITERATIONS),
            results_dir=Path(output.get("results_dir", "./benchmark_results")),
            save_answers=output.get("save_answers", True),
            tools_config=tools_config,
        )

    @staticmethod
    def _parse_questions(questions: str | list[int] | None) -> list[int] | None:
        """Parse question specification into list of IDs."""
        if questions is None:
            return None
        if isinstance(questions, list):
            return questions
        if isinstance(questions, str):
            result: list[int] = []
            for part in str(questions).split(","):
                part = part.strip()
                if "-" in part:
                    start, end = part.split("-")
                    result.extend(range(int(start), int(end) + 1))
                else:
                    result.append(int(part))
            return result
        return None


def load_config(config_path: Path | None, base_path: Path) -> BenchmarkConfig:
    """Load configuration from YAML file or use defaults.

    Args:
        config_path: Path to YAML config file, or None to use defaults.
        base_path: Base directory for resolving relative paths.

    Returns:
        Parsed BenchmarkConfig.

    Raises:
        ConfigurationError: If config_path is provided but does not exist.
    """
    if config_path is not None:
        if not config_path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            data = yaml.safe_load(f)
        print(f"Loaded config from: {config_path}")
        config = BenchmarkConfig.from_dict(data, base_path)
        config.config_path = config_path
        return config
    else:
        print("Using default configuration")
        return BenchmarkConfig.from_dict(DEFAULT_CONFIG, base_path)
