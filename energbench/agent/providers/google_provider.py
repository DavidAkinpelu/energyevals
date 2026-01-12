"""Google Gemini provider implementation."""

import os
import time
from typing import Any, AsyncIterator, Optional

from .base_provider import (
    BaseProvider,
    Message,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
)


class GoogleProvider(BaseProvider):
    """Google Gemini API provider implementation.

    Supports Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0, and other Gemini models.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the Google provider.

        Args:
            model: Model identifier (e.g., "gemini-2.0-flash", "gemini-1.5-pro").
            api_key: Google API key. Defaults to GOOGLE_API_KEY env var.
            base_url: Optional base URL for API (not typically used).
            **kwargs: Additional configuration.
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        super().__init__(model, api_key, base_url, **kwargs)

        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self.genai = genai
        self.client = genai.GenerativeModel(self.model)

    @property
    def provider_name(self) -> str:
        return "google"

    async def complete(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a completion using Google's Gemini API."""
        start_time = time.time()

        # Separate system message and format conversation
        system_msg, formatted_messages = self._separate_system_message(messages)

        # Create model with system instruction if present
        if system_msg:
            model = self.genai.GenerativeModel(
                self.model,
                system_instruction=system_msg,
            )
        else:
            model = self.client

        # Build generation config
        generation_config = self.genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Format tools if provided
        gemini_tools = None
        if tools:
            gemini_tools = self._format_tools(tools)

        # Generate response
        response = await model.generate_content_async(
            formatted_messages,
            generation_config=generation_config,
            tools=gemini_tools,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract content and tool calls
        content = ""
        tool_calls = None

        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    content += part.text
                elif hasattr(part, "function_call") and part.function_call:
                    if tool_calls is None:
                        tool_calls = []
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{len(tool_calls)}",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )

        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        # Determine finish reason
        finish_reason = "stop"
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason"):
                finish_reason = str(candidate.finish_reason.name).lower()

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            cached_tokens=0,
            output_tokens=output_tokens,
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
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion from Google Gemini."""
        system_msg, formatted_messages = self._separate_system_message(messages)

        if system_msg:
            model = self.genai.GenerativeModel(
                self.model,
                system_instruction=system_msg,
            )
        else:
            model = self.client

        generation_config = self.genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        gemini_tools = None
        if tools:
            gemini_tools = self._format_tools(tools)

        response = await model.generate_content_async(
            formatted_messages,
            generation_config=generation_config,
            tools=gemini_tools,
            stream=True,
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text

    def _format_tools(self, tools: list[ToolDefinition]) -> list[Any]:
        """Format tools for Google Gemini's function calling format."""
        function_declarations = []

        for tool in tools:
            # Convert JSON Schema to Gemini format
            parameters = self._convert_schema_to_gemini(tool.parameters)

            function_declarations.append(
                self.genai.protos.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=parameters,
                )
            )

        return [self.genai.protos.Tool(function_declarations=function_declarations)]

    def _convert_schema_to_gemini(self, schema: dict) -> Any:
        """Convert JSON Schema to Gemini Schema format."""
        if not schema:
            return None

        type_mapping = {
            "string": self.genai.protos.Type.STRING,
            "number": self.genai.protos.Type.NUMBER,
            "integer": self.genai.protos.Type.INTEGER,
            "boolean": self.genai.protos.Type.BOOLEAN,
            "array": self.genai.protos.Type.ARRAY,
            "object": self.genai.protos.Type.OBJECT,
        }

        schema_type = schema.get("type", "object")
        gemini_type = type_mapping.get(schema_type, self.genai.protos.Type.OBJECT)

        properties = {}
        if "properties" in schema:
            for name, prop in schema["properties"].items():
                properties[name] = self._convert_property_to_gemini(prop)

        return self.genai.protos.Schema(
            type=gemini_type,
            properties=properties if properties else None,
            required=schema.get("required", []),
            description=schema.get("description"),
        )

    def _convert_property_to_gemini(self, prop: dict) -> Any:
        """Convert a single property schema to Gemini format."""
        type_mapping = {
            "string": self.genai.protos.Type.STRING,
            "number": self.genai.protos.Type.NUMBER,
            "integer": self.genai.protos.Type.INTEGER,
            "boolean": self.genai.protos.Type.BOOLEAN,
            "array": self.genai.protos.Type.ARRAY,
            "object": self.genai.protos.Type.OBJECT,
        }

        prop_type = prop.get("type", "string")
        gemini_type = type_mapping.get(prop_type, self.genai.protos.Type.STRING)

        schema_kwargs = {
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

        return self.genai.protos.Schema(**schema_kwargs)

    def format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Format messages for Google Gemini's API.

        Gemini uses a different format with 'user' and 'model' roles.
        """
        formatted = []

        for msg in messages:
            if msg.role == "system":
                # System messages handled separately via system_instruction
                continue

            # Map roles to Gemini format
            role = "user" if msg.role in ("user", "tool") else "model"

            if msg.role == "tool" and msg.tool_call_id:
                # Tool result format for Gemini
                formatted.append({
                    "role": "user",
                    "parts": [
                        self.genai.protos.Part(
                            function_response=self.genai.protos.FunctionResponse(
                                name=msg.name or "unknown",
                                response={"result": msg.content},
                            )
                        )
                    ],
                })
            elif msg.tool_calls:
                # Assistant message with tool calls
                parts = []
                if msg.content:
                    parts.append({"text": msg.content})
                for tc in msg.tool_calls:
                    parts.append(
                        self.genai.protos.Part(
                            function_call=self.genai.protos.FunctionCall(
                                name=tc["name"],
                                args=tc["arguments"],
                            )
                        )
                    )
                formatted.append({"role": "model", "parts": parts})
            else:
                formatted.append({
                    "role": role,
                    "parts": [{"text": msg.content or ""}],
                })

        return formatted

    def _separate_system_message(
        self, messages: list[Message]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Separate system message from conversation messages.

        Gemini uses system_instruction for system prompts.
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
