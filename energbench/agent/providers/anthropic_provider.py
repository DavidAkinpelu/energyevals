import os
import time
from collections.abc import AsyncIterator
from typing import Any, Optional

from anthropic import AsyncAnthropic

from energbench.agent.constants import MAX_TOKENS

from .base_provider import (
    BaseProvider,
    ImageContent,
    Message,
    ProviderResponse,
    TextContent,
    ToolCall,
    ToolDefinition,
)


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider implementation.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the Anthropic provider.

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-20250514").
            api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
            base_url: Optional base URL for API.
            **kwargs: Additional configuration.
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        super().__init__(model, api_key, base_url, **kwargs)

        self.client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def complete(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = MAX_TOKENS,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a completion using Anthropic's API."""
        start_time = time.time()

        system_msg, formatted_messages = self._separate_system_message(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
        }

        if temperature > 0:
            request_kwargs["temperature"] = temperature

        if system_msg:
            request_kwargs["system"] = system_msg

        if tools:
            request_kwargs["tools"] = self.format_tools(tools)

        response = await self.client.messages.create(**request_kwargs)

        latency_ms = (time.time() - start_time) * 1000

        content = ""
        tool_calls = None

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        cached_tokens = getattr(response.usage, "cache_read_input_tokens", 0) or 0

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            model=response.model,
            finish_reason=response.stop_reason,
            raw_response=response,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = MAX_TOKENS,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion from Anthropic."""
        system_msg, formatted_messages = self._separate_system_message(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
        }

        if temperature > 0:
            request_kwargs["temperature"] = temperature

        if system_msg:
            request_kwargs["system"] = system_msg

        if tools:
            request_kwargs["tools"] = self.format_tools(tools)

        async with self.client.messages.stream(**request_kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Format tools for Anthropic's tool use format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Format messages for Anthropic's API.

        Handles multi-modal content (text + images).
        """
        formatted = []
        for msg in messages:
            if msg.role == "system":
                continue

            role = msg.role
            if role == "tool":
                role = "user"

            formatted_msg: dict[str, Any] = {"role": role}

            if msg.role == "tool" and msg.tool_call_id:
                tool_content = self._format_tool_result_content(msg)
                formatted_msg["content"] = [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": tool_content,
                    }
                ]
            elif msg.tool_calls:
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["arguments"],
                        }
                    )
                formatted_msg["content"] = content_blocks
            elif msg.content_parts and msg.has_images:
                formatted_msg["content"] = self._format_multimodal_content(msg)
            else:
                formatted_msg["content"] = msg.content

            formatted.append(formatted_msg)

        return formatted

    def _format_multimodal_content(self, msg: Message) -> list[dict[str, Any]]:
        """Format multi-modal content (text + images) for Anthropic."""
        content_blocks: list[dict[str, Any]] = []

        for part in msg.content_parts or []:
            if isinstance(part, TextContent):
                content_blocks.append({
                    "type": "text",
                    "text": part.text,
                })
            elif isinstance(part, ImageContent):
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.media_type,
                        "data": part.image_base64,
                    },
                })
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    content_blocks.append({
                        "type": "text",
                        "text": part.get("text", ""),
                    })
                elif part.get("type") == "image":
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.get("media_type", "image/jpeg"),
                            "data": part.get("image_base64", ""),
                        },
                    })

        return content_blocks

    def _format_tool_result_content(self, msg: Message) -> list[dict[str, Any]]:
        """Format tool result content, handling images from RAG results."""
        content_blocks: list[dict[str, Any]] = []

        if msg.content:
            content_blocks.append({
                "type": "text",
                "text": msg.content,
            })

        if msg.content_parts:
            for part in msg.content_parts:
                if isinstance(part, ImageContent):
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.media_type,
                            "data": part.image_base64,
                        },
                    })
                elif isinstance(part, dict) and part.get("type") == "image":
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.get("media_type", "image/jpeg"),
                            "data": part.get("image_base64", ""),
                        },
                    })

        if not content_blocks:
            return [{"type": "text", "text": msg.content or ""}]

        return content_blocks

    def _separate_system_message(
        self, messages: list[Message]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Separate system message from conversation messages.

        Anthropic requires system message to be passed separately.
        """
        system_msg = ""
        conversation = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                conversation.append(msg)

        formatted = self.format_messages(conversation)
        return system_msg, formatted
