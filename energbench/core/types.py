from pathlib import Path

PathLike = str | Path


def ensure_path(p: PathLike) -> Path:
    """Convert a path-like object to a Path.

    Args:
        p: String or Path object

    Returns:
        Path object
    """
    return Path(p) if isinstance(p, str) else p
