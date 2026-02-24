import base64
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from energbench.agent.providers import Message
from energbench.agent.providers.google_provider import GoogleProvider


class TestGoogleProviderUnit:
    """Mocked unit tests for GoogleProvider.complete()."""

    @pytest.mark.asyncio
    async def test_complete_returns_text_response(self, monkeypatch):
        """complete() maps a mocked response to a ProviderResponse correctly.

        google-genai may not be installed in the test environment, so the
        google.genai import inside GoogleProvider.__init__ is patched via
        sys.modules for the duration of provider construction.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "test")

        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            provider = GoogleProvider(model="gemini-2.0-flash")

        mock_part = MagicMock()
        mock_part.text = "Hi"
        mock_part.function_call = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason = "STOP"

        mock_resp = MagicMock()
        mock_resp.candidates = [mock_candidate]
        mock_resp.usage_metadata.prompt_token_count = 10
        mock_resp.usage_metadata.candidates_token_count = 5
        mock_resp.usage_metadata.thoughts_token_count = 0

        provider.client = MagicMock()
        provider.client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        result = await provider.complete([Message(role="user", content="Hi")])

        assert result.content == "Hi"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    @pytest.mark.asyncio
    async def test_complete_captures_thought_signature(self, monkeypatch):
        """complete() should capture thought_signature for Gemini tool calls."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test")

        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            provider = GoogleProvider(model="gemini-2.0-flash")

        mock_fc = MagicMock()
        mock_fc.name = "test_tool"
        mock_fc.args = {"x": 1}

        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_fc
        mock_part.thought_signature = b"sig"

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason = "STOP"

        mock_resp = MagicMock()
        mock_resp.candidates = [mock_candidate]
        mock_resp.usage_metadata.prompt_token_count = 10
        mock_resp.usage_metadata.candidates_token_count = 5
        mock_resp.usage_metadata.thoughts_token_count = 0

        provider.client = MagicMock()
        provider.client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        result = await provider.complete([Message(role="user", content="Hi")])

        assert result.tool_calls
        assert result.tool_calls[0].thought_signature == base64.b64encode(b"sig").decode("ascii")

    def test_convert_property_to_gemini_drops_non_string_enum(self, monkeypatch):
        """_convert_property_to_gemini should omit non-string enums for Gemini."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test")

        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            provider = GoogleProvider(model="gemini-2.0-flash")

        provider._types.Schema = MagicMock(side_effect=lambda **kwargs: kwargs)

        numeric_enum = {"type": "integer", "enum": [7]}
        result_numeric = provider._convert_property_to_gemini(numeric_enum)
        assert "enum" not in result_numeric

        string_enum = {"type": "string", "enum": ["a", "b"]}
        result_string = provider._convert_property_to_gemini(string_enum)
        assert result_string["enum"] == ["a", "b"]

    def test_format_messages_includes_thought_signature(self, monkeypatch):
        """format_messages should include thought_signature on functionCall parts."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test")

        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            provider = GoogleProvider(model="gemini-2.0-flash")

        class DummyPart:
            def __init__(self):
                self.thought_signature = None
                self.text = None
                self.function_call = None

        created_parts = []

        def make_part(*, name, args):
            _ = (name, args)
            part = DummyPart()
            created_parts.append(part)
            return part

        provider._types.Part.from_function_call = MagicMock(side_effect=make_part)
        provider._types.Content = MagicMock()

        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[{
                    "name": "test_tool",
                    "arguments": {"x": 1},
                    "thought_signature": "c2ln",
                }],
            )
        ]

        provider.format_messages(messages)

        # format_messages decodes the base64 str to bytes for Google's API
        assert created_parts[0].thought_signature == b"sig"


@pytest.mark.integration
@pytest.mark.requires_api_key
@pytest.mark.google
class TestGoogleProviderIntegration:
    """Integration tests with real Google API."""

    @pytest.mark.asyncio
    async def test_basic_completion(self):
        """Test basic completion with real API."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

        provider = GoogleProvider(model="gemini-2.0-flash")
        messages = [Message(role="user", content="Say 'test passed'.")]

        response = await provider.complete(messages)

        assert response.content
        assert response.output_tokens > 0

    @pytest.mark.asyncio
    async def test_tool_calling(self, sample_tool):
        """Test tool calling with real API."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

        provider = GoogleProvider(model="gemini-2.0-flash")
        messages = [
            Message(role="user", content="Get the energy price for ERCOT market."),
        ]

        response = await provider.complete(messages, tools=[sample_tool])

        assert response.tool_calls is not None or response.content
        if response.tool_calls:
            assert len(response.tool_calls) > 0
            assert response.tool_calls[0].name
            assert response.tool_calls[0].id

    @pytest.mark.asyncio
    async def test_streaming(self):
        """Test streaming with real API."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

        provider = GoogleProvider(model="gemini-2.0-flash")
        messages = [Message(role="user", content="Count to 3.")]

        chunks = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0

    @pytest.mark.asyncio
    async def test_system_message(self):
        """Test system message handling."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

        provider = GoogleProvider(model="gemini-2.0-flash")
        messages = [
            Message(role="system", content="You are an energy expert."),
            Message(role="user", content="What is energy?"),
        ]

        response = await provider.complete(messages)

        assert response.content
        assert response.input_tokens > 0
        assert response.output_tokens > 0

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

        provider = GoogleProvider(model="gemini-2.0-flash")
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello!"),
            Message(role="assistant", content="Hi! How can I help you?"),
            Message(role="user", content="What's 2+2?"),
        ]

        response = await provider.complete(messages)

        assert response.content
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_response_structure(self):
        """Test full response structure validation."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

        provider = GoogleProvider(model="gemini-2.0-flash")
        messages = [Message(role="user", content="Say hi")]

        response = await provider.complete(messages)

        assert isinstance(response.content, str)
        assert len(response.content) > 0
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.latency_ms > 0
        assert response.model == "gemini-2.0-flash"
        assert response.finish_reason is not None

    @pytest.mark.asyncio
    async def test_tool_call_roundtrip(self, sample_tool):
        """Test full tool call roundtrip."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

        provider = GoogleProvider(model="gemini-2.0-flash")

        # Step 1: User asks question that needs a tool
        messages = [
            Message(role="user", content="Get the energy price for ERCOT market."),
        ]
        response = await provider.complete(messages, tools=[sample_tool])

        # Should get a tool call
        if response.tool_calls:
            assert len(response.tool_calls) > 0
            tool_call = response.tool_calls[0]
            assert tool_call.name == "get_energy_price"
            assert "market" in tool_call.arguments

            # Step 2: Add assistant message with tool call and tool result
            tool_call_payload = {
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
            if tool_call.thought_signature:
                tool_call_payload["thought_signature"] = tool_call.thought_signature
            messages.append(
                Message(role="assistant", content="", tool_calls=[
                    tool_call_payload
                ])
            )
            messages.append(
                Message(
                    role="tool",
                    content='{"price": 45.5, "unit": "$/MWh"}',
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                )
            )

            # Step 3: Get final answer
            final_response = await provider.complete(messages, tools=[sample_tool])
            assert final_response.content
            assert len(final_response.content) > 0

    @pytest.mark.asyncio
    async def test_multimodal_image(self, sample_image_message):
        """Test multimodal image input."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

        provider = GoogleProvider(model="gemini-2.0-flash")

        response = await provider.complete([sample_image_message])

        assert response.content
        assert len(response.content) > 0
