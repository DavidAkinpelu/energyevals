from dataclasses import dataclass
from typing import Optional


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
    is_reasoning_model: Optional[bool] = None

    @property
    def display_name(self) -> str:
        """Return display name like 'openai/gpt-4o-mini'."""
        return f"{self.provider}/{self.model}"

    @property
    def safe_filename(self) -> str:
        """Return filesystem-safe name for output files."""
        return f"{self.provider}_{self.model.replace('/', '_').replace('.', '-')}"
