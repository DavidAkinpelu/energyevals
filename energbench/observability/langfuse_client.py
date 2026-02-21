import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from loguru import logger

from energbench.agent.schema import AgentRun, AgentStep, StepType

from .base import BaseObserver

try:
    from langfuse import Langfuse as _Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    _Langfuse = None
    LANGFUSE_AVAILABLE = False


def get_langfuse_client(
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
) -> Any:
    """Get a Langfuse client instance.

    Args:
        public_key: Langfuse public key. Defaults to LANGFUSE_PUBLIC_KEY env var.
        secret_key: Langfuse secret key. Defaults to LANGFUSE_SECRET_KEY env var.
        host: Langfuse host URL. Defaults to LANGFUSE_HOST env var.

    Returns:
        Langfuse client or None if not available/configured.
    """
    if not LANGFUSE_AVAILABLE:
        logger.warning("Langfuse not installed. Run: pip install langfuse")
        return None

    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning("Langfuse credentials not configured. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
        return None

    try:
        return _Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        return None


class LangfuseObserver(BaseObserver):
    """Observer for tracing agent runs to Langfuse.

    This class provides methods to trace complete agent runs including
    all LLM calls, tool executions, and the full reasoning chain.

    Captures:
    - Complete traces of agent execution
    - All steps (action, observation, answer, error, thought)
    - Full tool inputs and outputs (not truncated)
    - Failed tool calls with error details
    - Token usage and latency metrics
    """

    def __init__(
        self,
        client: Any = None,
        enabled: bool = True,
    ):
        """Initialize the observer.

        Args:
            client: Langfuse client. If None, will try to create one.
            enabled: Whether observability is enabled.
        """
        self._enabled = enabled and LANGFUSE_AVAILABLE
        self.client = client or (get_langfuse_client() if self._enabled else None)

        if self._enabled and not self.client:
            logger.warning("Langfuse observer disabled - client not available")
            self._enabled = False
        elif self._enabled:
            logger.info("LangfuseObserver initialized")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def trace_agent_run(
        self,
        run: AgentRun,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> str | None:
        """Trace a complete agent run to Langfuse.

        Args:
            run: The AgentRun to trace.
            metadata: Additional metadata to attach.
            tags: Tags for filtering in Langfuse.
            user_id: User identifier for the trace.
            session_id: Session identifier for grouping traces.

        Returns:
            Trace ID if successful, None otherwise.
        """
        if not self._enabled or not self.client:
            return None

        try:
            trace = self.client.trace(
                name="agent_run",
                input={"query": run.query},
                output={"answer": run.final_answer} if run.final_answer else None,
                metadata={
                    "success": run.success,
                    "iterations": run.iterations,
                    "tool_calls_count": run.tool_calls_count,
                    "total_tokens": run.total_tokens,
                    "total_reasoning_tokens": run.total_reasoning_tokens,
                    "duration_seconds": run.duration_seconds,
                    "error": run.error,
                    **(metadata or {}),
                },
                tags=tags or [],
                user_id=user_id,
                session_id=session_id,
            )

            for i, step in enumerate(run.steps):
                self._trace_step(trace, step, i)

            self.client.flush()

            logger.debug(f"Traced agent run to Langfuse: {trace.id}")
            return str(trace.id)

        except Exception as e:
            logger.error(f"Failed to trace agent run: {e}")
            return None

    def _trace_step(self, trace: Any, step: AgentStep, index: int) -> None:
        step_name = f"{index:02d}_{step.step_type.value}"

        if step.step_type == StepType.ACTION:
            trace.span(
                name=step_name,
                input={
                    "tool": step.tool_name,
                    "arguments": step.tool_input,
                },
                metadata={
                    "step_type": step.step_type.value,
                    "latency_ms": step.latency_ms,
                },
            )

        elif step.step_type == StepType.OBSERVATION:
            trace.span(
                name=step_name,
                input={"tool": step.tool_name},
                output=step.tool_output,
                metadata={
                    "step_type": step.step_type.value,
                    "latency_ms": step.latency_ms,
                    "content_length": len(step.tool_output) if step.tool_output else 0,
                },
            )

        elif step.step_type == StepType.ANSWER:
            trace.span(
                name=step_name,
                output=step.content,
                metadata={
                    "step_type": step.step_type.value,
                    "tokens_used": step.tokens_used,
                    "latency_ms": step.latency_ms,
                },
            )

        elif step.step_type == StepType.ERROR:
            trace.span(
                name=step_name,
                output=step.content,
                metadata={
                    "step_type": step.step_type.value,
                    "error": True,
                },
                level="ERROR",
            )

        else:
            trace.span(
                name=step_name,
                output=step.content,
                metadata={
                    "step_type": step.step_type.value,
                    "tokens_used": step.tokens_used,
                    "latency_ms": step.latency_ms,
                },
            )

    def trace_llm_call(
        self,
        trace_id: str,
        model: str,
        messages: list[dict[str, Any]],
        response: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Trace an individual LLM call.

        Args:
            trace_id: Parent trace ID.
            model: Model name.
            messages: Full message history (not truncated).
            response: Full response content.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            latency_ms: Call latency in milliseconds.
            tool_calls: Any tool calls made.
        """
        if not self._enabled or not self.client:
            return

        try:
            self.client.generation(
                trace_id=trace_id,
                name="llm_call",
                model=model,
                input=messages,
                output=response,
                usage={
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                },
                metadata={
                    "latency_ms": latency_ms,
                    "tool_calls": tool_calls,
                },
            )
        except Exception as e:
            logger.error(f"Failed to trace LLM call: {e}")

    def trace_tool_execution(
        self,
        trace_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        latency_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Trace a tool execution.

        Args:
            trace_id: Parent trace ID.
            tool_name: Name of the tool.
            arguments: Full tool arguments.
            result: Full tool result (not truncated).
            latency_ms: Execution latency.
            error: Error message if failed.
        """
        if not self._enabled or not self.client:
            return

        try:
            self.client.span(
                trace_id=trace_id,
                name=f"tool_{tool_name}",
                input=arguments,
                output=result,
                metadata={
                    "latency_ms": latency_ms,
                    "error": error,
                    "result_length": len(result),
                },
                level="ERROR" if error else "DEFAULT",
            )
        except Exception as e:
            logger.error(f"Failed to trace tool execution: {e}")

    def flush(self) -> None:
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse: {e}")

    def shutdown(self) -> None:
        if self.client:
            try:
                self.client.shutdown()
            except Exception as e:
                logger.error(f"Failed to shutdown Langfuse: {e}")


class ObserverContext:
    """Context holder for observe_agent_run with pre-configured trace settings."""

    def __init__(
        self,
        observer: BaseObserver,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ):
        self.observer = observer
        self.metadata = metadata
        self.tags = tags
        self.user_id = user_id
        self.session_id = session_id

    def trace(self, run: AgentRun) -> str | None:
        """Trace an agent run with pre-configured settings.

        Args:
            run: The AgentRun to trace.

        Returns:
            Trace ID if successful, None otherwise.
        """
        return self.observer.trace_agent_run(
            run=run,
            metadata=self.metadata,
            tags=self.tags,
            user_id=self.user_id,
            session_id=self.session_id,
        )


@contextmanager
def observe_agent_run(
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    observer: BaseObserver | None = None,
) -> Iterator[ObserverContext]:
    """Context manager for observing an agent run.

    Usage:
        with observe_agent_run(tags=["test"]) as ctx:
            run = await agent.run("my query")
            ctx.trace(run)  # Uses pre-configured metadata, tags, etc.

    Args:
        metadata: Additional metadata.
        tags: Tags for the trace.
        user_id: User identifier.
        session_id: Session identifier.
        observer: Observer to use. Defaults to LangfuseObserver().

    Yields:
        ObserverContext with pre-configured trace settings.
    """
    obs = observer or LangfuseObserver()
    ctx = ObserverContext(
        observer=obs,
        metadata=metadata,
        tags=tags,
        user_id=user_id,
        session_id=session_id,
    )

    try:
        yield ctx
    finally:
        obs.flush()
