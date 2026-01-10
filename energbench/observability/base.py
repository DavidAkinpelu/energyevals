"""Base observer interface for observability backends."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from energbench.agent.schema import AgentRun


class BaseObserver(ABC):
    """Abstract base class for observability backends.

    All observers must implement this interface to ensure consistent
    behavior across different backends (Langfuse, JSON, etc.).
    """

    @abstractmethod
    def trace_agent_run(
        self,
        run: AgentRun,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Trace a complete agent run.

        Args:
            run: The AgentRun to trace (includes all steps, metrics, errors).
            metadata: Additional metadata to attach to the trace.
            tags: Tags for filtering/categorizing traces.
            user_id: User identifier for the trace.
            session_id: Session identifier for grouping related traces.

        Returns:
            Trace ID if successful, None otherwise.
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush any pending traces to the backend."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the observer and release resources."""
        pass

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if the observer is enabled and functional."""
        pass
