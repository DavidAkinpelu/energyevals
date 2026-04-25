import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from energyevals.agent.providers import Message
from energyevals.agent.providers.anthropic_provider import AnthropicProvider


class TestAnthropicProviderUnit:
    """Mocked unit tests for AnthropicProvider.complete()."""

    @pytest.mark.asyncio
    async def test_complete_returns_text_response(self, monkeypatch):
        """complete() maps a mocked response to a ProviderResponse correctly."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Hi"

        mock_resp = MagicMock()
        mock_resp.content = [mock_block]
        mock_resp.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_resp.usage.cache_read_input_tokens = 0
        mock_resp.model = "claude-sonnet-4-20250514"
        mock_resp.stop_reason = "end_turn"

        provider.client = MagicMock()
        provider.client.messages.create = AsyncMock(return_value=mock_resp)

        result = await provider.complete([Message(role="user", content="Hi")])

        assert result.content == "Hi"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    @pytest.mark.asyncio
    async def test_default_effort_sends_low(self, monkeypatch):
        """complete() always sends output_config={"effort": "low"} by default."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "ok"

        mock_resp = MagicMock()
        mock_resp.content = [mock_block]
        mock_resp.usage = MagicMock(input_tokens=5, output_tokens=2)
        mock_resp.usage.cache_read_input_tokens = 0
        mock_resp.model = "claude-sonnet-4-20250514"
        mock_resp.stop_reason = "end_turn"

        provider.client = MagicMock()
        provider.client.messages.create = AsyncMock(return_value=mock_resp)

        await provider.complete([Message(role="user", content="hi")])

        call_kwargs = provider.client.messages.create.call_args.kwargs
        assert call_kwargs.get("output_config") == {"effort": "low"}

    @pytest.mark.asyncio
    async def test_custom_effort_passed_to_api(self, monkeypatch):
        """complete() passes the specified effort value in output_config."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        provider = AnthropicProvider(model="claude-opus-4-20250514", effort="high")

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "ok"

        mock_resp = MagicMock()
        mock_resp.content = [mock_block]
        mock_resp.usage = MagicMock(input_tokens=5, output_tokens=2)
        mock_resp.usage.cache_read_input_tokens = 0
        mock_resp.model = "claude-opus-4-20250514"
        mock_resp.stop_reason = "end_turn"

        provider.client = MagicMock()
        provider.client.messages.create = AsyncMock(return_value=mock_resp)

        await provider.complete([Message(role="user", content="hi")])

        call_kwargs = provider.client.messages.create.call_args.kwargs
        assert call_kwargs.get("output_config") == {"effort": "high"}


@pytest.mark.integration
@pytest.mark.requires_api_key
@pytest.mark.anthropic
class TestAnthropicProviderIntegration:
    """Integration tests with real Anthropic API."""

    @pytest.mark.asyncio
    async def test_basic_completion(self):
        """Test basic completion with real API."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        messages = [Message(role="user", content="Say 'test passed'.")]

        response = await provider.complete(messages)

        assert response.content
        assert response.output_tokens > 0
        assert response.input_tokens > 0

    @pytest.mark.asyncio
    async def test_tool_calling(self, sample_tool):
        """Test tool calling with real API."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
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
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
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
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
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
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
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
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        messages = [Message(role="user", content="Say hi")]

        response = await provider.complete(messages)

        assert isinstance(response.content, str)
        assert len(response.content) > 0
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.latency_ms > 0
        assert response.model
        assert response.finish_reason is not None

    @pytest.mark.asyncio
    async def test_tool_call_roundtrip(self, sample_tool):
        """Test full tool call roundtrip."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        provider = AnthropicProvider(model="claude-sonnet-4-20250514")

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
            messages.append(
                Message(role="assistant", content="", tool_calls=[
                    {"id": tool_call.id, "name": tool_call.name, "arguments": tool_call.arguments}
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
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        provider = AnthropicProvider(model="claude-sonnet-4-20250514")

        response = await provider.complete([sample_image_message])

        assert response.content
        assert len(response.content) > 0
