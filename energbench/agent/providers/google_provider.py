import base64
import os
import time
from collections.abc import AsyncIterator
from typing import Any, Optional

from energbench.agent.schema.messages import ImageContent, TextContent

from .base_provider import (
    BaseProvider,
    Message,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
)


class GoogleProvider(BaseProvider):
    """Google Gemini API provider implementation.

    Uses the google-genai SDK (replaces the deprecated google-generativeai package).
    Supports Gemini models via a centralized Client object.
    """

    _TYPE_MAP: dict[str, str] = {
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the Google provider.

        Args:
            model: Model identifier (e.g., "gemini-2.0-flash", "gemini-2.5-pro").
            api_key: Google API key. Defaults to GOOGLE_API_KEY env var.
            base_url: Optional base URL for API (not typically used).
            **kwargs: Additional configuration.
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        super().__init__(model, api_key, base_url, **kwargs)

        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=self.api_key)
        self._types = types

    @property
    def provider_name(self) -> str:
        return "google"

    async def complete(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a completion using Google's Gemini API."""
        start_time = time.time()

        system_msg, formatted_messages = self._separate_system_message(messages)

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
        }
        if system_msg:
            config_kwargs["system_instruction"] = system_msg
        if tools:
            config_kwargs["tools"] = self._format_tools(tools)

        config = self._types.GenerateContentConfig(**config_kwargs)

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=formatted_messages,
            config=config,
        )

        latency_ms = (time.time() - start_time) * 1000

        content = ""
        tool_calls: Optional[list[ToolCall]] = None

        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        content += part.text
                    elif part.function_call:
                        if tool_calls is None:
                            tool_calls = []
                        fc = part.function_call
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{len(tool_calls)}",
                                name=str(fc.name),
                                arguments=dict(fc.args) if fc.args else {},
                            )
                        )

        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0
            reasoning_tokens = (
                getattr(response.usage_metadata, "thoughts_token_count", 0) or 0
            )

        finish_reason = "stop"
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.finish_reason:
                fr_str = str(candidate.finish_reason)
                finish_reason = fr_str.rsplit(".", 1)[-1].lower()

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            cached_tokens=0,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_ms=latency_ms,
            model=self.model,
            finish_reason=finish_reason,
            raw_response=response,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion from Google Gemini."""
        system_msg, formatted_messages = self._separate_system_message(messages)

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
        }
        if system_msg:
            config_kwargs["system_instruction"] = system_msg
        if tools:
            config_kwargs["tools"] = self._format_tools(tools)

        config = self._types.GenerateContentConfig(**config_kwargs)

        response_stream = await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=formatted_messages,
            config=config,
        )
        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    def format_tools(self, tools: list[ToolDefinition]) -> list[Any]:
        """Format tools for Google Gemini's function calling format."""
        return self._format_tools(tools)

    def _format_tools(self, tools: list[ToolDefinition]) -> list[Any]:
        """Internal method to format tools for Google Gemini's function calling format."""
        function_declarations = []

        for tool in tools:
            parameters = self._convert_schema_to_gemini(tool.parameters)

            function_declarations.append(
                self._types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=parameters,
                )
            )

        return [self._types.Tool(function_declarations=function_declarations)]

    def _convert_schema_to_gemini(self, schema: dict[str, Any]) -> Any:
        """Convert JSON Schema to Gemini Schema format."""
        if not schema:
            return None

        schema_type = schema.get("type", "object")
        gemini_type = self._TYPE_MAP.get(schema_type, "OBJECT")

        properties = {}
        if "properties" in schema:
            for name, prop in schema["properties"].items():
                properties[name] = self._convert_property_to_gemini(prop)

        return self._types.Schema(
            type=gemini_type,  # type: ignore[arg-type]
            properties=properties if properties else None,
            required=schema.get("required", []),
            description=schema.get("description"),
        )

    def _convert_property_to_gemini(self, prop: dict[str, Any]) -> Any:
        """Convert a single property schema to Gemini format."""
        prop_type = prop.get("type", "string")
        gemini_type = self._TYPE_MAP.get(prop_type, "STRING")

        schema_kwargs: dict[str, Any] = {
            "type": gemini_type,
            "description": prop.get("description"),
        }

        if "enum" in prop:
            schema_kwargs["enum"] = prop["enum"]

        if prop_type == "array" and "items" in prop:
            schema_kwargs["items"] = self._convert_property_to_gemini(prop["items"])

        if prop_type == "object" and "properties" in prop:
            schema_kwargs["properties"] = {
                name: self._convert_property_to_gemini(p)
                for name, p in prop["properties"].items()
            }

        return self._types.Schema(**schema_kwargs)

    def _format_multimodal_content(self, msg: Message) -> list[Any]:
        """Format multi-modal content (text + images) for Gemini."""
        types = self._types
        parts: list[Any] = []
        for part in msg.content_parts or []:
            if isinstance(part, TextContent):
                parts.append(types.Part.from_text(text=part.text))
            elif isinstance(part, ImageContent):
                if part.image_base64:
                    parts.append(
                        types.Part.from_bytes(
                            data=base64.b64decode(part.image_base64),
                            mime_type=part.media_type,
                        )
                    )
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(types.Part.from_text(text=part.get("text", "")))
                elif part.get("type") == "image":
                    parts.append(
                        types.Part.from_bytes(
                            data=base64.b64decode(part.get("image_base64", "")),
                            mime_type=part.get("media_type", "image/jpeg"),
                        )
                    )
        return parts

    def format_messages(self, messages: list[Message]) -> list[Any]:
        """Format messages for Google Gemini's API.

        Gemini uses a different format with 'user' and 'model' roles.
        """
        types = self._types
        formatted: list[Any] = []

        for msg in messages:
            if msg.role == "system":
                continue

            role = "user" if msg.role in ("user", "tool") else "model"

            if msg.role == "tool" and msg.tool_call_id:
                formatted.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=msg.name or "unknown",
                                response={"result": msg.content},
                            )
                        ],
                    )
                )
            elif msg.tool_calls:
                parts: list[Any] = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                for tc in msg.tool_calls:
                    parts.append(
                        types.Part.from_function_call(
                            name=tc["name"],
                            args=tc["arguments"],
                        )
                    )
                formatted.append(types.Content(role="model", parts=parts))
            elif msg.content_parts and msg.has_images:
                parts = self._format_multimodal_content(msg)
                formatted.append(types.Content(role=role, parts=parts))
            else:
                formatted.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg.content or "")],
                    )
                )

        return formatted

    def _separate_system_message(
        self, messages: list[Message]
    ) -> tuple[str, list[Any]]:
        """Separate system message from conversation messages.

        Gemini uses system_instruction in GenerateContentConfig for system prompts.
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
