import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from energbench.agent.constants import CSV_THRESHOLD, MAX_ITERATIONS


class StepType(Enum):
    """Types of steps in the agent's execution."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    ANSWER = "answer"
    ERROR = "error"


@dataclass
class AgentStep:
    """Represents a single step in the agent's execution.

    Attributes:
        step_type: The type of step (thought, action, observation, answer, error).
        content: The content or description of this step.
        tool_name: Name of the tool called (for action/observation steps).
        tool_input: Input arguments passed to the tool.
        tool_output: Output returned from the tool.
        tokens_used: Number of tokens used in this step.
        latency_ms: Time taken for this step in milliseconds.
        timestamp: Unix timestamp when this step occurred.
    """

    step_type: StepType
    content: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: str | None = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentRun:
    """Represents a complete agent execution run.

    Attributes:
        query: The original user query.
        steps: List of all steps taken during execution.
        final_answer: The final answer produced by the agent.
        total_input_tokens: Total input tokens used across all steps.
        total_cached_tokens: Total cached input tokens used across all steps.
        total_output_tokens: Total output tokens used across all steps.
        total_reasoning_tokens: Total reasoning tokens used across all steps.
        total_latency_ms: Total latency across all steps.
        tool_calls_count: Number of tool calls made.
        iterations: Number of iterations completed.
        success: Whether the run completed successfully.
        error: Error message if the run failed.
        start_time: Unix timestamp when the run started.
        end_time: Unix timestamp when the run ended.
    """

    query: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None
    total_input_tokens: int = 0
    total_cached_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_latency_ms: float = 0.0
    tool_calls_count: int = 0
    iterations: int = 0
    success: bool = False
    error: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used in this run."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def duration_seconds(self) -> float:
        """Total duration of the run in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


@dataclass
class AgentConfig:
    """Configuration for a ReAct agent.

    Attributes:
        max_iterations: Maximum number of iterations before stopping.
        csv_threshold: Row count threshold for saving results to CSV.
        csv_output_dir: Directory to save CSV files.
        system_prompt: Custom system prompt (None uses default).
    """

    max_iterations: int = MAX_ITERATIONS
    csv_threshold: int = CSV_THRESHOLD
    csv_output_dir: str = "./agent_outputs"
    system_prompt: str | None = None
