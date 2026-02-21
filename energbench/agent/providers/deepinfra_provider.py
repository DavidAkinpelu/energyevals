import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from energbench.agent.constants import MAX_TOKENS
from energbench.agent.schema.messages import ImageContent, TextContent

from .base_provider import (
    BaseProvider,
    Message,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
)


class DeepInfraProvider(BaseProvider):
    """DeepInfra API provider implementation.

    Uses OpenAI-compatible API to access various open-source models like
    Llama, Mixtral, and others hosted on DeepInfra.
    """

    DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.3-70B-Instruct",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the DeepInfra provider.

        Args:
            model: Full model path (e.g., "meta-llama/Llama-3.3-70B-Instruct",
                   "Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-V3").
            api_key: DeepInfra API key. Defaults to DEEPINFRA_API_KEY env var.
            base_url: Optional base URL override.
            **kwargs: Additional configuration.
        """
        api_key = api_key or os.getenv("DEEPINFRA_API_KEY")
        base_url = base_url or self.DEEPINFRA_BASE_URL

        super().__init__(model, api_key, base_url, **kwargs)

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    @property
    def provider_name(self) -> str:
        return "deepinfra"

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = MAX_TOKENS,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a completion using DeepInfra's API."""
        start_time = time.time()

        formatted_messages = self.format_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            request_kwargs["tools"] = self.format_tools(tools)
            request_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        request_kwargs.update({k: v for k, v in kwargs.items() if k not in request_kwargs})

        response = await self.client.chat.completions.create(**request_kwargs)

        latency_ms = (time.time() - start_time) * 1000

        tool_calls = None
        message = response.choices[0].message
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]

        cached_tokens = 0
        if response.usage and getattr(response.usage, "prompt_tokens_details", None):
            cached_tokens = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0) or 0

        return ProviderResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            cached_tokens=cached_tokens,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            latency_ms=latency_ms,
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
            raw_response=response,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = MAX_TOKENS,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion from DeepInfra."""
        formatted_messages = self.format_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if tools:
            request_kwargs["tools"] = self.format_tools(tools)
            request_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        response = await self.client.chat.completions.create(**request_kwargs)

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Format tools for OpenAI-compatible function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Format messages for DeepInfra's API (OpenAI-compatible)."""
        formatted = []
        for msg in messages:
            formatted_msg: dict[str, Any] = {
                "role": msg.role,
            }
            if msg.content_parts and msg.has_images:
                formatted_msg["content"] = self._format_multimodal_content(msg)
            else:
                formatted_msg["content"] = msg.content

            if msg.tool_calls:
                formatted_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.dumps(tc["arguments"])
                                if isinstance(tc["arguments"], dict)
                                else tc["arguments"]
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]

            if msg.role == "tool" and msg.tool_call_id:
                formatted_msg["tool_call_id"] = msg.tool_call_id

            formatted.append(formatted_msg)

        return formatted

    def _format_multimodal_content(self, msg: Message) -> list[dict[str, Any]]:
        """Format multi-modal content (text + images) for OpenAI-compatible API."""
        content_parts: list[dict[str, Any]] = []
        for part in msg.content_parts or []:
            if isinstance(part, TextContent):
                content_parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImageContent):
                if part.image_url:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": part.image_url},
                    })
                elif part.image_base64:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{part.media_type};base64,{part.image_base64}"
                        },
                    })
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    content_parts.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") == "image":
                    url = part.get("image_url") or f"data:{part.get('media_type', 'image/jpeg')};base64,{part.get('image_base64', '')}"
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": url},
                    })
        return content_parts
