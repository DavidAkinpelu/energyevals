"""OpenAI provider implementation."""

import json
import os
import time
from typing import Any, AsyncIterator, Literal, Optional

from openai import AsyncOpenAI

from .base_provider import (
    BaseProvider,
    ImageContent,
    Message,
    ProviderResponse,
    TextContent,
    ToolCall,
    ToolDefinition,
)

# Models that use reasoning_effort instead of temperature (o-series and gpt-5 series)
# These models do NOT support: temperature, top_p, presence_penalty, frequency_penalty,
# logprobs, top_logprobs, logit_bias, max_tokens
REASONING_MODEL_PREFIXES = (
    "o1",
    "o3",
    "o4",
    "gpt-5",
)


def is_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning model (o1/o3/o4/gpt-5 series).

    Reasoning models use reasoning_effort instead of temperature and
    max_completion_tokens instead of max_tokens.
    """
    model_lower = model.lower()
    return any(model_lower.startswith(prefix) for prefix in REASONING_MODEL_PREFIXES)


class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation.

    Supports GPT-4, GPT-4o, GPT-4o-mini, and reasoning models (o1, o3, o4, gpt-5 series).

    For reasoning models (o1, o3, o4, gpt-5):
    - Uses `reasoning_effort` instead of `temperature`
    - Uses `max_completion_tokens` instead of `max_tokens`
    - System messages are converted to developer messages
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        **kwargs: Any,
    ):
        """Initialize the OpenAI provider.

        Args:
            model: Model identifier (e.g., "gpt-4o", "o3-mini", "gpt-5").
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            base_url: Optional base URL for API.
            reasoning_effort: For reasoning models, controls thinking depth
                ("low", "medium", "high"). Default is "medium".
            **kwargs: Additional configuration.
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(model, api_key, base_url, **kwargs)

        self.reasoning_effort = reasoning_effort
        self.is_reasoning_model = is_reasoning_model(model)

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    async def complete(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a completion using OpenAI's API.

        Args:
            messages: List of conversation messages.
            tools: Optional list of tool definitions.
            temperature: Temperature for non-reasoning models (0.0-2.0).
            max_tokens: Maximum tokens for non-reasoning models.
            reasoning_effort: For reasoning models (o1/o3/o4/gpt-5), controls
                thinking depth ("low", "medium", "high"). Overrides instance default.
            **kwargs: Additional API parameters.

        Returns:
            ProviderResponse with completion results.
        """
        start_time = time.time()

        system_text = None
        input_items: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                content_text = ""
                if msg.content:
                    first = msg.content[0]
                    if isinstance(first, TextContent):
                        content_text = first.text
                    else:
                        content_text = str(first)
                if system_text:
                    system_text = f"{system_text}\n\n{content_text}"
                else:
                    system_text = content_text
                continue

            if msg.role == "tool":
                if not msg.tool_call_id:
                    continue
                tool_output = msg.text_content or msg.content or ""
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id,
                        "output": tool_output,
                    }
                )
                continue

            role = msg.role if msg.role in {"user", "assistant", "developer"} else "assistant"
            content_items: list[dict[str, Any]] = []

            parts = msg.content_parts if msg.content_parts else [msg.content]
            for item in parts:
                if isinstance(item, TextContent):
                    content_items.append({"type": "input_text", "text": item.text})
                elif isinstance(item, ImageContent):
                    content_items.append({"type": "input_image", "image_url": item.image_url})
                elif isinstance(item, str):
                    if item:
                        content_items.append({"type": "input_text", "text": item})
                else:
                    content_items.append({"type": "input_text", "text": str(item)})

            if content_items:
                input_items.append({"role": role, "content": content_items})

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
                    name = tool_call.get("name") if isinstance(tool_call, dict) else None
                    arguments = tool_call.get("arguments") if isinstance(tool_call, dict) else {}
                    if call_id and name:
                        input_items.append(
                            {
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                                "arguments": json.dumps(arguments),
                            }
                        )

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }

        # Handle reasoning models differently
        if self.is_reasoning_model:
            # Reasoning models use reasoning_effort instead of temperature
            effort = reasoning_effort or self.reasoning_effort
            request_kwargs["reasoning"] = {"effort": effort}
            # Use max_completion_tokens instead of max_tokens
            request_kwargs["max_output_tokens"] = max_tokens
        else:
            # Standard models use temperature and max_tokens
            request_kwargs["temperature"] = temperature
            request_kwargs["max_output_tokens"] = max_tokens

        if system_text:
            request_kwargs["instructions"] = system_text

        if tools:
            request_kwargs["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in tools
            ]
            request_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Merge any additional kwargs (except ones we handle specially)
        excluded_keys = {"tool_choice", "reasoning_effort"}
        request_kwargs.update({k: v for k, v in kwargs.items() if k not in request_kwargs and k not in excluded_keys})

        response = await self.client.responses.create(**request_kwargs)

        latency_ms = (time.time() - start_time) * 1000

        # Extract tool calls if present
        tool_calls = None
        content = ""
        if hasattr(response, "output_text") and response.output_text:
            content = response.output_text
        else:
            for item in response.output:
                if getattr(item, "type", None) == "message":
                    for block in item.content:
                        if getattr(block, "type", None) in {"output_text", "text"}:
                            content += block.text
                elif getattr(item, "type", None) in {"tool_call", "function_call"}:
                    tool_calls = tool_calls or []
                    name = getattr(item, "name", None) or getattr(item, "tool_name", None)
                    arguments = getattr(item, "arguments", None)
                    if arguments is None and getattr(item, "function", None):
                        name = name or item.function.name
                        arguments = getattr(item.function, "arguments", None)
                    call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                    try:
                        parsed_args = json.loads(arguments) if isinstance(arguments, str) else arguments or {}
                    except json.JSONDecodeError:
                        parsed_args = {}
                    tool_calls.append(ToolCall(id=call_id or "", name=name or "", arguments=parsed_args))

        cached_tokens = 0
        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = getattr(response.usage, "input_tokens", 0) or 0
            output_tokens = getattr(response.usage, "output_tokens", 0) or 0
            details = getattr(response.usage, "input_tokens_details", None)
            if details is not None:
                cached_tokens = getattr(details, "cached_tokens", 0) or 0

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model=response.model,
            finish_reason=getattr(response, "status", None),
            raw_response=response,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion from OpenAI.

        Note: Reasoning models (o1/o3/o4/gpt-5) have limited streaming support.
        The full response may be returned at once after reasoning completes.
        """
        formatted_messages = self.format_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": True,
        }

        # Handle reasoning models differently
        if self.is_reasoning_model:
            effort = reasoning_effort or self.reasoning_effort
            request_kwargs["reasoning_effort"] = effort
            request_kwargs["max_completion_tokens"] = max_tokens
        else:
            request_kwargs["temperature"] = temperature
            request_kwargs["max_tokens"] = max_tokens

        if tools:
            request_kwargs["tools"] = self.format_tools(tools)
            request_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        response = await self.client.chat.completions.create(**request_kwargs)

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Format tools for OpenAI's function calling format."""
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
        """Format messages for OpenAI's API.

        For reasoning models (o1/o3/o4/gpt-5), system messages are converted
        to developer messages as required by the API.

        Handles multi-modal content (text + images).
        """
        formatted = []
        for msg in messages:
            role = msg.role

            # For reasoning models, convert system messages to developer messages
            if self.is_reasoning_model and role == "system":
                role = "developer"

            formatted_msg: dict[str, Any] = {"role": role}

            # Handle multi-modal content
            if msg.content_parts and msg.has_images:
                content_parts = []
                for part in msg.content_parts:
                    if isinstance(part, TextContent):
                        content_parts.append({
                            "type": "text",
                            "text": part.text,
                        })
                    elif isinstance(part, ImageContent):
                        # OpenAI uses image_url with base64 data URL
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{part.media_type};base64,{part.image_base64}",
                            },
                        })
                    elif isinstance(part, dict):
                        if part.get("type") == "text":
                            content_parts.append({
                                "type": "text",
                                "text": part.get("text", ""),
                            })
                        elif part.get("type") == "image":
                            media_type = part.get("media_type", "image/jpeg")
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{part.get('image_base64', '')}",
                                },
                            })
                formatted_msg["content"] = content_parts
            else:
                # Simple text content
                formatted_msg["content"] = msg.content

            # Handle tool calls in assistant messages
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

            # Handle tool response messages
            if msg.role == "tool" and msg.tool_call_id:
                formatted_msg["tool_call_id"] = msg.tool_call_id

            formatted.append(formatted_msg)

        return formatted
