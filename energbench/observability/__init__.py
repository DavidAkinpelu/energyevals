from typing import Optional

from .base import BaseObserver
from .composite_observer import CompositeObserver
from .json_observer import JSONFileObserver
from .langfuse_client import (
    LangfuseObserver,
    ObserverContext,
    get_langfuse_client,
    observe_agent_run,
)


def get_observer(
    backend: str = "auto",
    output_dir: str = "./observability_logs",
    run_name: Optional[str] = None,
    single_file: bool = False,
) -> BaseObserver:
    """Get an observer instance based on the specified backend.

    Args:
        backend: One of:
            - "langfuse": Use Langfuse cloud observability
            - "json": Use local JSON file logging
            - "both": Use both Langfuse and JSON (composite)
            - "auto": Use Langfuse if available, otherwise JSON
        output_dir: Base directory for JSON file output (used with "json" or "both").
        run_name: Optional subdirectory for organizing runs (e.g., "no_tools", "with_tools").
                 When provided, traces are saved to: {output_dir}/{run_name}/{model}/
        single_file: If True, append all JSON traces to one JSONL file.

    Returns:
        Configured observer instance.

    Raises:
        ValueError: If backend is not recognized.

    Example:
        >>> observer = get_observer("json", output_dir="./traces", run_name="no_tools")
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
            run_name=run_name,
            single_file=single_file,
        )

    elif backend == "both":
        observers: list[BaseObserver] = []

        # Try Langfuse
        langfuse_obs = LangfuseObserver()
        if langfuse_obs.is_enabled:
            observers.append(langfuse_obs)

        # Always add JSON
        json_obs = JSONFileObserver(
            output_dir=output_dir,
            run_name=run_name,
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
            run_name=run_name,
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
    "ObserverContext",
]
