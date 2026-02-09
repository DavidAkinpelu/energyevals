from dataclasses import dataclass
from typing import Any, Optional

from .tools import ToolCall


@dataclass
class ProviderResponse:
    """Response from an LLM provider.

    Attributes:
        content: The text content of the response.
        tool_calls: List of tool calls requested by the model.
        input_tokens: Number of input tokens used.
        cached_tokens: Number of cached input tokens used (if reported by provider).
        output_tokens: Number of output tokens generated.
        reasoning_tokens: Number of reasoning tokens used (for reasoning models).
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
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None
