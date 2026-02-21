from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from energbench.agent.constants import MAX_TOKENS
from energbench.agent.schema import (
    ImageContent,
    Message,
    ProviderResponse,
    TextContent,
    ToolCall,
    ToolDefinition,
)

__all__ = [
    "BaseProvider",
    "ImageContent",
    "Message",
    "ProviderResponse",
    "TextContent",
    "ToolCall",
    "ToolDefinition",
]


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations must inherit from this class and implement
    the required abstract methods.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the provider.

        Args:
            model: The model identifier to use.
            api_key: API key for authentication. If None, will try to load from environment.
            base_url: Optional base URL for the API.
            **kwargs: Additional provider-specific configuration.
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.config = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = MAX_TOKENS,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a completion from the model.

        Args:
            messages: List of messages in the conversation.
            tools: Optional list of tools available to the model.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Returns:
            ProviderResponse containing the model's response.
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = MAX_TOKENS,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion from the model.

        Args:
            messages: List of messages in the conversation.
            tools: Optional list of tools available to the model.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Yields:
            String chunks as they are generated.
        """
        yield ""
        raise NotImplementedError

    @abstractmethod
    def format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Format tools for this provider's API format.

        Args:
            tools: List of tool definitions in standard format.

        Returns:
            Tools formatted for the provider's API.
        """
        pass

    def format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Format messages for this provider's API format.

        Default implementation that can be overridden by specific providers.

        Args:
            messages: List of messages in standard format.

        Returns:
            Messages formatted for the provider's API.
        """
        formatted = []
        for msg in messages:
            formatted_msg: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.tool_calls:
                formatted_msg["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                formatted_msg["tool_call_id"] = msg.tool_call_id
            if msg.name:
                formatted_msg["name"] = msg.name
            formatted.append(formatted_msg)
        return formatted

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
