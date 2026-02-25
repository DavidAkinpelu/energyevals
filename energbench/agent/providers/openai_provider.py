import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from openai import AsyncOpenAI

from energbench.agent.exceptions import ProviderError

from .base_provider import (
    BaseProvider,
    ImageContent,
    Message,
    ProviderResponse,
    TextContent,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

REASONING_MODEL_PREFIXES = (
    "o1",
    "o3",
    "o4",
    "gpt-5",
)


def is_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning model (o1/o3/o4/gpt-5 series).

    Reasoning models use reasoning_effort instead of temperature.
    """
    model_lower = model.lower()
    return any(model_lower.startswith(prefix) for prefix in REASONING_MODEL_PREFIXES)


class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation using responses.create endpoint.

    Supports all OpenAI models through the unified responses.create API:
    - GPT-4, GPT-4o, GPT-4o-mini
    - Reasoning models (o1, o3, o4, gpt-5 series)

    For reasoning models (o1, o3, o4, gpt-5):
    - Uses `reasoning.effort` instead of `temperature`
    - System messages are passed as instructions
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: Literal["low", "medium", "high"] = "low",
        is_reasoning_model_override: bool | None = None,
        **kwargs: Any,
    ):
        """Initialize the OpenAI provider.

        Args:
            model: Model identifier (e.g., "gpt-4o", "o3-mini", "gpt-5").
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            base_url: Optional base URL for API.
            reasoning_effort: For reasoning models, controls thinking depth
                ("low", "medium", "high"). Default is "medium".
            is_reasoning_model_override: Explicitly set whether this model is a
                reasoning model. If None, auto-detects based on model name prefix.
                Use this to override the hardcoded list for new/custom models.
            **kwargs: Additional configuration.
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(model, api_key, base_url, **kwargs)

        self.reasoning_effort = reasoning_effort
        auto_detected = is_reasoning_model(model)
        if is_reasoning_model_override is not None:
            if is_reasoning_model_override is False and auto_detected:
                logger.warning(
                    f"Model '{model}' is a known reasoning model (matches prefix: "
                    f"{', '.join(REASONING_MODEL_PREFIXES)}), but is_reasoning_model "
                    f"override is set to False. This may cause API errors. "
                    f"Proceeding with override."
                )
            self.is_reasoning_model = is_reasoning_model_override
        else:
            self.is_reasoning_model = auto_detected

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        del self.api_key

    @property
    def provider_name(self) -> str:
        return "openai"

    def _build_input_items(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal Message list to OpenAI Responses API `input` format.

        Returns:
            (system_text, input_items) where system_text is extracted from
            any system-role messages and input_items is the formatted list.
        """
        system_text: str | None = None
        input_items: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                content_text = msg.content or ""
                if not content_text and msg.content_parts:
                    parts = []
                    for part in msg.content_parts:
                        if isinstance(part, TextContent):
                            parts.append(part.text)
                        elif isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                    content_text = "\n".join(parts)
                system_text = f"{system_text}\n\n{content_text}" if system_text else content_text
                continue

            if msg.role == "tool":
                if not msg.tool_call_id:
                    continue
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": msg.text_content or msg.content or "",
                })
                continue

            role = msg.role if msg.role in {"user", "assistant", "developer"} else "assistant"
            text_type = "output_text" if role == "assistant" else "input_text"
            content_items: list[dict[str, Any]] = []

            msg_parts: list[TextContent | ImageContent] = (
                msg.content_parts if msg.content_parts else [TextContent(text=msg.content or "")]
            )
            for item in msg_parts:
                if isinstance(item, TextContent):
                    content_items.append({"type": text_type, "text": item.text})
                elif isinstance(item, ImageContent):
                    if item.image_url:
                        content_items.append({"type": "input_image", "image_url": item.image_url})
                    elif item.image_base64:
                        content_items.append({
                            "type": "input_image",
                            "image_url": f"data:{item.media_type};base64,{item.image_base64}",
                        })
                else:
                    content_items.append({"type": text_type, "text": str(item)})

            if content_items:
                input_items.append({"role": role, "content": content_items})

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    call_id = tc.get("id") if isinstance(tc, dict) else None
                    name = tc.get("name") if isinstance(tc, dict) else None
                    arguments = tc.get("arguments") if isinstance(tc, dict) else {}
                    if call_id and name:
                        input_items.append({
                            "type": "function_call",
                            "call_id": call_id,
                            "name": name,
                            "arguments": json.dumps(arguments),
                        })

        return system_text, input_items

    def _build_request_kwargs(
        self,
        input_items: list[dict[str, Any]],
        system_text: str | None,
        tools: list[ToolDefinition] | None,
        temperature: float,
        max_tokens: int | None,
        reasoning_effort: str | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the request_kwargs dict for responses.create / responses.stream."""
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }

        if self.is_reasoning_model:
            request_kwargs["reasoning"] = {"effort": reasoning_effort or self.reasoning_effort}
        else:
            request_kwargs["temperature"] = temperature
        request_kwargs["max_output_tokens"] = max_tokens

        if system_text:
            request_kwargs["instructions"] = system_text

        if tools:
            request_kwargs["tools"] = self.format_tools(tools)
            request_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        excluded = {"tool_choice", "reasoning_effort"}
        request_kwargs.update({k: v for k, v in kwargs.items() if k not in request_kwargs and k not in excluded})

        return request_kwargs

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
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

        system_text, input_items = self._build_input_items(messages)
        request_kwargs = self._build_request_kwargs(
            input_items, system_text, tools, temperature, max_tokens, reasoning_effort, **kwargs
        )
        response = await self.client.responses.create(**request_kwargs)

        latency_ms = (time.time() - start_time) * 1000

        try:
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
            reasoning_tokens = 0
            if response.usage:
                input_tokens = getattr(response.usage, "input_tokens", 0) or 0
                output_tokens = getattr(response.usage, "output_tokens", 0) or 0
                details = getattr(response.usage, "input_tokens_details", None)
                if details is not None:
                    cached_tokens = getattr(details, "cached_tokens", 0) or 0
                output_details = getattr(response.usage, "output_tokens_details", None)
                if output_details is not None:
                    reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) or 0

            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,
                input_tokens=input_tokens,
                cached_tokens=cached_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                latency_ms=latency_ms,
                model=response.model,
                finish_reason=getattr(response, "status", None),
                raw_response=response,
            )
        except (KeyError, AttributeError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"Malformed API response: {exc}. Raw: {response!r}",
                provider=self.provider_name,
            ) from exc

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion from OpenAI using responses.create.

        Note: Reasoning models (o1/o3/o4/gpt-5) have limited streaming support.
        The full response may be returned at once after reasoning completes.
        """
        system_text, input_items = self._build_input_items(messages)
        request_kwargs = self._build_request_kwargs(
            input_items, system_text, tools, temperature, max_tokens, reasoning_effort, **kwargs
        )
        async with self.client.responses.stream(**request_kwargs) as stream:
            async for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta":
                    delta: str = getattr(event, "delta", "")
                    if delta:
                        yield delta

    def format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Format messages for OpenAI's responses.create API."""
        _, items = self._build_input_items(messages)
        return items

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Format tools for OpenAI's responses.create API."""
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ]
