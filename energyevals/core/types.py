from enum import StrEnum
from pathlib import Path

PathLike = str | Path


class ProviderName(StrEnum):
    """Known LLM provider identifiers.

    Using StrEnum so values compare equal to plain strings — existing code that
    checks ``model_spec.provider == "openai"`` continues to work without changes.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DEEPINFRA = "deepinfra"


def ensure_path(p: PathLike) -> Path:
    """Convert a path-like object to a Path.

    Args:
        p: String or Path object

    Returns:
        Path object
    """
    return Path(p) if isinstance(p, str) else p
