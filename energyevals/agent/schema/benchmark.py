from dataclasses import dataclass


@dataclass
class ModelSpec:
    """Specification for a single model to evaluate.

    Attributes:
        provider: The provider name (openai, anthropic, google, deepinfra).
        model: The model identifier.
        is_reasoning_model: Override for reasoning model detection (None = auto-detect).
    """

    provider: str
    model: str
    is_reasoning_model: bool | None = None
    effort: str | None = None  # Anthropic: "low"|"medium"|"high"|"max"; Google: "low"|"medium"|"high" (thinking_level); None = use provider default

    @property
    def display_name(self) -> str:
        """Return display name like 'openai/gpt-4o-mini'."""
        return f"{self.provider}/{self.model}"

    @property
    def params_summary(self) -> str:
        """Return a bracketed summary of non-default model params, e.g. '[effort=medium]'."""
        parts = []
        if self.effort is not None:
            parts.append(f"effort={self.effort}")
        if self.is_reasoning_model is True:
            parts.append("reasoning")
        elif self.is_reasoning_model is False:
            parts.append("no-reasoning")
        return f" [{', '.join(parts)}]" if parts else ""

    @property
    def safe_filename(self) -> str:
        """Return filesystem-safe name for output files."""
        return f"{self.provider}_{self.model.replace('/', '_').replace('.', '-')}"
