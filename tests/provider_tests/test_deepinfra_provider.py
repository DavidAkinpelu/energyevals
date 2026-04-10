import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from energyevals.agent.providers import Message
from energyevals.agent.providers.deepinfra_provider import DeepInfraProvider


class TestDeepInfraProviderUnit:
    """Mocked unit tests for DeepInfraProvider.complete()."""

    @pytest.mark.asyncio
    async def test_complete_returns_text_response(self, monkeypatch):
        """complete() maps a mocked response to a ProviderResponse correctly."""
        monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
        provider = DeepInfraProvider(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")

        mock_message = MagicMock()
        mock_message.content = "Hi"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        mock_resp.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            prompt_tokens_details=None,
        )

        provider.client = MagicMock()
        provider.client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await provider.complete([Message(role="user", content="Hi")])

        assert result.content == "Hi"
        assert result.input_tokens == 10
        assert result.output_tokens == 5


@pytest.mark.integration
@pytest.mark.requires_api_key
@pytest.mark.deepinfra
class TestDeepInfraProviderIntegration:
    """Integration tests with real DeepInfra API."""

    @pytest.mark.asyncio
    async def test_basic_completion(self):
        """Test basic completion with real API."""
        if not os.getenv("DEEPINFRA_API_KEY"):
            pytest.skip("DEEPINFRA_API_KEY not set")

        provider = DeepInfraProvider(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
        messages = [Message(role="user", content="Say 'test passed'.")]

        response = await provider.complete(messages)

        assert response.content

    @pytest.mark.asyncio
    async def test_tool_calling(self, sample_tool):
        """Test tool calling with real API."""
        if not os.getenv("DEEPINFRA_API_KEY"):
            pytest.skip("DEEPINFRA_API_KEY not set")

        provider = DeepInfraProvider(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
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
        if not os.getenv("DEEPINFRA_API_KEY"):
            pytest.skip("DEEPINFRA_API_KEY not set")

        provider = DeepInfraProvider(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
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
        if not os.getenv("DEEPINFRA_API_KEY"):
            pytest.skip("DEEPINFRA_API_KEY not set")

        provider = DeepInfraProvider(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
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
        if not os.getenv("DEEPINFRA_API_KEY"):
            pytest.skip("DEEPINFRA_API_KEY not set")

        provider = DeepInfraProvider(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
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
        if not os.getenv("DEEPINFRA_API_KEY"):
            pytest.skip("DEEPINFRA_API_KEY not set")

        provider = DeepInfraProvider(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
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
        if not os.getenv("DEEPINFRA_API_KEY"):
            pytest.skip("DEEPINFRA_API_KEY not set")

        provider = DeepInfraProvider(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")

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
        """Test multimodal image input with vision model."""
        if not os.getenv("DEEPINFRA_API_KEY"):
            pytest.skip("DEEPINFRA_API_KEY not set")

        # Use a vision-capable model for multimodal test
        provider = DeepInfraProvider(model="meta-llama/Llama-3.2-11B-Vision-Instruct")

        response = await provider.complete([sample_image_message])

        assert response.content
        assert len(response.content) > 0
