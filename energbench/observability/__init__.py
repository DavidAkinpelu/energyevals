"""Observability module for energBench.

Supports multiple observability backends:
- Langfuse: Cloud-based observability platform
- JSON: Local JSON file logging
- Composite: Use multiple backends simultaneously

Usage:
    # Get a single observer
    observer = get_observer("langfuse")  # or "json" or "both"

    # Trace an agent run
    trace_id = observer.trace_agent_run(run, tags=["test"])

    # Custom configuration
    from energbench.observability import LangfuseObserver, JSONFileObserver
    observer = JSONFileObserver(output_dir="./my_traces", pretty_print=True)
"""

from typing import Optional

from .base import BaseObserver
from .langfuse_client import (
    LangfuseObserver,
    get_langfuse_client,
    observe_agent_run,
)
from .json_observer import JSONFileObserver
from .composite_observer import CompositeObserver


def get_observer(
    backend: str = "auto",
    output_dir: str = "./observability_logs",
    single_file: bool = False,
) -> BaseObserver:
    """Get an observer instance based on the specified backend.

    Args:
        backend: One of:
            - "langfuse": Use Langfuse cloud observability
            - "json": Use local JSON file logging
            - "both": Use both Langfuse and JSON (composite)
            - "auto": Use Langfuse if available, otherwise JSON
        output_dir: Directory for JSON file output (used with "json" or "both").
        single_file: If True, append all JSON traces to one JSONL file.

    Returns:
        Configured observer instance.

    Raises:
        ValueError: If backend is not recognized.

    Example:
        >>> observer = get_observer("json", output_dir="./traces")
        >>> observer.trace_agent_run(run, tags=["experiment"])
    """
    backend = backend.lower()

    if backend == "langfuse":
        observer = LangfuseObserver()
        if not observer.is_enabled:
            raise ValueError(
                "Langfuse observer not available. "
                "Check LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY env vars."
            )
        return observer

    elif backend == "json":
        return JSONFileObserver(
            output_dir=output_dir,
            single_file=single_file,
        )

    elif backend == "both":
        observers = []

        # Try Langfuse
        langfuse_obs = LangfuseObserver()
        if langfuse_obs.is_enabled:
            observers.append(langfuse_obs)

        # Always add JSON
        json_obs = JSONFileObserver(
            output_dir=output_dir,
            single_file=single_file,
        )
        observers.append(json_obs)

        return CompositeObserver(observers)

    elif backend == "auto":
        # Try Langfuse first, fall back to JSON
        langfuse_obs = LangfuseObserver()
        if langfuse_obs.is_enabled:
            return langfuse_obs
        return JSONFileObserver(
            output_dir=output_dir,
            single_file=single_file,
        )

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Choose from: langfuse, json, both, auto"
        )


__all__ = [
    # Base
    "BaseObserver",
    # Implementations
    "LangfuseObserver",
    "JSONFileObserver",
    "CompositeObserver",
    # Factory
    "get_observer",
    # Langfuse utilities
    "get_langfuse_client",
    "observe_agent_run",
]
