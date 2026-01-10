"""Schema definitions for the energBench agent framework.

This module contains all data classes and type definitions used across
the agent, providers, and tools.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# =============================================================================
# Content Types (for multi-modal support)
# =============================================================================


@dataclass
class TextContent:
    """Text content in a message."""

    type: str = "text"
    text: str = ""


@dataclass
class ImageContent:
    """Image content in a message (base64 encoded)."""

    type: str = "image"
    image_base64: str = ""
    media_type: str = "image/jpeg"  # image/jpeg, image/png, image/gif, image/webp


# Union type for content
ContentPart = TextContent | ImageContent | dict[str, Any]


# =============================================================================
# Message Types
# =============================================================================


@dataclass
class Message:
    """Represents a message in a conversation.

    Attributes:
        role: The role of the message sender ("system", "user", "assistant", "tool").
        content: The text content of the message (for simple text messages).
        content_parts: List of content parts for multi-modal messages.
        tool_calls: List of tool calls made by the assistant.
        tool_call_id: ID of the tool call this message is responding to.
        name: Name of the tool for tool messages.
    """

    role: str
    content: str = ""
    content_parts: Optional[list[ContentPart]] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    @property
    def has_images(self) -> bool:
        """Check if message contains images."""
        if not self.content_parts:
            return False
        return any(
            (isinstance(p, ImageContent) or (isinstance(p, dict) and p.get("type") == "image"))
            for p in self.content_parts
        )

    @property
    def text_content(self) -> str:
        """Get concatenated text content."""
        if self.content:
            return self.content
        if not self.content_parts:
            return ""
        texts = []
        for part in self.content_parts:
            if isinstance(part, TextContent):
                texts.append(part.text)
            elif isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "\n".join(texts)


# =============================================================================
# Tool Types
# =============================================================================


@dataclass
class ToolCall:
    """Represents a tool call from the model.

    Attributes:
        id: Unique identifier for this tool call.
        name: Name of the tool being called.
        arguments: Arguments passed to the tool.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolDefinition:
    """Definition of a tool for the LLM.

    Attributes:
        name: Unique name of the tool.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema defining the tool's parameters.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


# Type alias for tool executor function
ToolExecutor = Callable[[str, dict[str, Any]], str]


# =============================================================================
# Provider Types
# =============================================================================


@dataclass
class ProviderResponse:
    """Response from an LLM provider.

    Attributes:
        content: The text content of the response.
        tool_calls: List of tool calls requested by the model.
        input_tokens: Number of input tokens used.
        cached_tokens: Number of cached input tokens used (if reported by provider).
        output_tokens: Number of output tokens generated.
        latency_ms: Response latency in milliseconds.
        model: The model that generated the response.
        finish_reason: Reason the model stopped generating.
        raw_response: The raw response object from the provider.
    """

    content: str
    tool_calls: Optional[list[ToolCall]] = None
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None


# =============================================================================
# Agent Types
# =============================================================================


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
    tool_name: Optional[str] = None
    tool_input: Optional[dict[str, Any]] = None
    tool_output: Optional[str] = None
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
    final_answer: Optional[str] = None
    total_input_tokens: int = 0
    total_cached_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    tool_calls_count: int = 0
    iterations: int = 0
    success: bool = False
    error: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

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


# =============================================================================
# Configuration Types
# =============================================================================


@dataclass
class AgentConfig:
    """Configuration for a ReAct agent.

    Attributes:
        max_iterations: Maximum number of iterations before stopping.
        csv_threshold: Row count threshold for saving results to CSV.
        csv_output_dir: Directory to save CSV files.
        system_prompt: Custom system prompt (None uses default).
    """

    max_iterations: int = 10
    csv_threshold: int = 20
    csv_output_dir: str = "./agent_outputs"
    system_prompt: Optional[str] = None


@dataclass
class ToolResult:
    """Result from a tool execution.

    Attributes:
        success: Whether the tool executed successfully.
        data: The result data (can be dict, list, or string).
        error: Error message if execution failed.
        row_count: Number of rows if result is tabular data.
        csv_path: Path to CSV file if data was saved.
        metadata: Additional metadata about the result.
    """

    success: bool
    data: Any
    error: Optional[str] = None
    row_count: int = 0
    csv_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_context_string(self, csv_threshold: int = 20) -> str:
        """Convert result to a string suitable for LLM context.

        If the result has more rows than csv_threshold and was saved to CSV,
        returns a reference to the CSV file instead of the full data.

        Args:
            csv_threshold: Row count threshold for using CSV reference.

        Returns:
            String representation of the result for LLM context.
        """
        import json

        if not self.success:
            return json.dumps({"error": self.error})

        if self.csv_path and self.row_count > csv_threshold:
            return json.dumps({
                "status": "success",
                "row_count": self.row_count,
                "csv_file": self.csv_path,
                "message": f"Results saved to {self.csv_path}. Use Python to read and analyze the CSV file.",
                "preview": self.data[:5] if isinstance(self.data, list) else None,
            }, indent=2, default=str)

        return json.dumps(self.data, indent=2, default=str)
