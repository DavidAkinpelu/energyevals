from typing import Any, Optional

from loguru import logger

from energbench.agent.schema import AgentRun

from .base import BaseObserver


class CompositeObserver(BaseObserver):
    """Observer that delegates to multiple backend observers.

    Use this when you want traces sent to multiple destinations,
    for example both Langfuse and local JSON files.

    Example:
        observer = CompositeObserver([
            LangfuseObserver(),
            JSONFileObserver(output_dir="./traces"),
        ])
        observer.trace_agent_run(run)  # Traces to both backends
    """

    def __init__(self, observers: list[BaseObserver]):
        """Initialize with multiple observers.

        Args:
            observers: List of observer instances to delegate to.
        """
        self.observers = [obs for obs in observers if obs.is_enabled]

        if not self.observers:
            logger.warning("CompositeObserver has no enabled observers")
        else:
            observer_names = [type(obs).__name__ for obs in self.observers]
            logger.info(f"CompositeObserver initialized with: {observer_names}")

    @property
    def is_enabled(self) -> bool:
        """Check if at least one observer is enabled."""
        return len(self.observers) > 0

    def trace_agent_run(
        self,
        run: AgentRun,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Trace to all configured observers.

        Args:
            run: The AgentRun to trace.
            metadata: Additional metadata.
            tags: Tags for the trace.
            user_id: User identifier.
            session_id: Session identifier.

        Returns:
            First successful trace ID, or None if all failed.
        """
        trace_ids = []
        errors = []

        for observer in self.observers:
            try:
                trace_id = observer.trace_agent_run(
                    run=run,
                    metadata=metadata,
                    tags=tags,
                    user_id=user_id,
                    session_id=session_id,
                )
                if trace_id:
                    trace_ids.append((type(observer).__name__, trace_id))
            except Exception as e:
                errors.append((type(observer).__name__, str(e)))
                logger.error(f"Observer {type(observer).__name__} failed: {e}")

        if errors:
            logger.warning(f"Some observers failed: {errors}")

        if trace_ids:
            logger.debug(f"Traced to observers: {trace_ids}")
            return trace_ids[0][1]  # Return first successful trace ID

        return None

    def trace_llm_call(
        self,
        trace_id: str,
        model: str,
        messages: list[dict],
        response: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        tool_calls: Optional[list[dict]] = None,
    ) -> None:
        """Delegate LLM call tracing to all observers."""
        for observer in self.observers:
            try:
                observer.trace_llm_call(
                    trace_id=trace_id,
                    model=model,
                    messages=messages,
                    response=response,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    tool_calls=tool_calls,
                )
            except Exception as e:
                logger.error(f"Observer {type(observer).__name__} trace_llm_call failed: {e}")

    def trace_tool_execution(
        self,
        trace_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        """Delegate tool execution tracing to all observers."""
        for observer in self.observers:
            try:
                observer.trace_tool_execution(
                    trace_id=trace_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                    latency_ms=latency_ms,
                    error=error,
                )
            except Exception as e:
                logger.error(f"Observer {type(observer).__name__} trace_tool_execution failed: {e}")

    def flush(self) -> None:
        """Flush all observers."""
        for observer in self.observers:
            try:
                observer.flush()
            except Exception as e:
                logger.error(f"Failed to flush {type(observer).__name__}: {e}")

    def shutdown(self) -> None:
        """Shutdown all observers."""
        for observer in self.observers:
            try:
                observer.shutdown()
            except Exception as e:
                logger.error(f"Failed to shutdown {type(observer).__name__}: {e}")

    def add_observer(self, observer: BaseObserver) -> None:
        """Add an observer to the composite.

        Args:
            observer: Observer to add.
        """
        if observer.is_enabled:
            self.observers.append(observer)
            logger.info(f"Added observer: {type(observer).__name__}")

    def remove_observer(self, observer_type: type) -> bool:
        """Remove observers of a specific type.

        Args:
            observer_type: Type of observer to remove.

        Returns:
            True if any observers were removed.
        """
        original_count = len(self.observers)
        self.observers = [
            obs for obs in self.observers
            if not isinstance(obs, observer_type)
        ]
        removed = original_count - len(self.observers)
        if removed:
            logger.info(f"Removed {removed} observer(s) of type {observer_type.__name__}")
        return removed > 0
