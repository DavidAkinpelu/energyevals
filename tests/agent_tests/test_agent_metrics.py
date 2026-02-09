import os
from unittest.mock import AsyncMock, Mock

import pytest

from energbench.agent.providers import ProviderResponse, ToolCall, get_provider
from energbench.agent.react_agent import ReActAgent
from energbench.agent.schema import StepType
from energbench.tools.base_tool import BaseTool, ToolDefinition, ToolRegistry


class SimpleTestTool(BaseTool):
    """Simple test tool for metrics testing."""

    def __init__(self):
        super().__init__(name="test_tool", description="A test tool")
        self.register_method("get_answer", self.get_answer)

    def get_tools(self):
        return [
            ToolDefinition(
                name="get_answer",
                description="Get a test answer",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            )
        ]

    def get_answer(self, query: str) -> str:
        """Return test answer."""
        return f"Answer for: {query}"


@pytest.fixture
def mock_provider():
    """Mock provider that returns predictable responses."""
    mock = Mock()

    tool_response = ProviderResponse(
        content=None,
        tool_calls=[
            ToolCall(
                id="call_1",
                name="get_answer",
                arguments={"query": "test"},
            )
        ],
        model="mock-model",
        input_tokens=50,
        output_tokens=20,
        latency_ms=100,
    )

    final_response = ProviderResponse(
        content="Final answer based on tool result.",
        tool_calls=None,
        model="mock-model",
        input_tokens=75,
        output_tokens=15,
        latency_ms=120,
    )

    mock.complete = AsyncMock(side_effect=[tool_response, final_response])
    mock.model = "mock-model"
    mock.provider_name = "mock"

    return mock


@pytest.fixture
def test_tool():
    """Test tool fixture."""
    return SimpleTestTool()


class TestAgentMetricsCollection:
    """Test metrics collection in agent runs."""

    @pytest.mark.asyncio
    async def test_basic_metrics_tracking(self, mock_provider, test_tool):
        """Test that agent collects all metrics correctly."""
        agent = ReActAgent(provider=mock_provider, tools=[test_tool])
        run = await agent.run("Test query")

        assert run.total_input_tokens == 125
        assert run.total_output_tokens == 35
        assert run.total_tokens == 160
        assert run.total_latency_ms == 220
        assert run.iterations == 2
        assert run.tool_calls_count > 0
        assert run.success is True

    @pytest.mark.asyncio
    async def test_step_recording(self, mock_provider, test_tool):
        """Test that agent records steps correctly."""
        agent = ReActAgent(provider=mock_provider, tools=[test_tool])
        run = await agent.run("Test query")

        assert len(run.steps) == 3

        step1 = run.steps[0]
        assert step1.step_type == StepType.ACTION
        assert step1.tool_name == "get_answer"
        assert step1.latency_ms == 100

        step2 = run.steps[1]
        assert step2.step_type == StepType.OBSERVATION
        assert step2.tool_name == "get_answer"

        step3 = run.steps[2]
        assert step3.step_type == StepType.ANSWER
        assert step3.content is not None

    @pytest.mark.asyncio
    async def test_token_accumulation(self, mock_provider, test_tool):
        """Test that tokens accumulate correctly across steps."""
        agent = ReActAgent(provider=mock_provider, tools=[test_tool])
        run = await agent.run("Test query")

        assert run.total_input_tokens == 125
        assert run.total_output_tokens == 35
        assert run.total_tokens == 160

    @pytest.mark.asyncio
    async def test_latency_accumulation(self, mock_provider, test_tool):
        """Test that latency accumulates correctly across steps."""
        agent = ReActAgent(provider=mock_provider, tools=[test_tool])
        run = await agent.run("Test query")

        assert run.total_latency_ms == 220

    @pytest.mark.asyncio
    async def test_iteration_count(self, mock_provider, test_tool):
        """Test that iteration count is correct."""
        agent = ReActAgent(provider=mock_provider, tools=[test_tool])
        run = await agent.run("Test query")

        assert run.iterations == 2
        assert len(run.steps) == 3


class TestAgentMetricsEdgeCases:
    """Test edge cases in metrics collection."""

    @pytest.mark.asyncio
    async def test_no_tool_calls(self):
        """Test metrics when agent doesn't use tools."""
        mock = Mock()
        mock.complete = AsyncMock(
            return_value=ProviderResponse(
                content="Direct answer",
                tool_calls=None,
                model="mock",
                input_tokens=30,
                output_tokens=10,
                latency_ms=50,
            )
        )
        mock.model = "mock"
        mock.provider_name = "mock"

        agent = ReActAgent(provider=mock, tools=[])
        run = await agent.run("Simple query")

        assert run.iterations == 1
        assert run.total_input_tokens == 30
        assert run.total_output_tokens == 10
        assert run.final_answer is not None
        assert len(run.steps) == 1

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self, test_tool):
        """Test metrics when max iterations is reached."""
        mock = Mock()
        mock.complete = AsyncMock(
            return_value=ProviderResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call", name="get_answer", arguments={"query": "test"})
                ],
                model="mock",
                input_tokens=50,
                output_tokens=20,
                latency_ms=100,
            )
        )
        mock.model = "mock"
        mock.provider_name = "mock"

        agent = ReActAgent(provider=mock, tools=[test_tool], max_iterations=3)
        run = await agent.run("Query")

        assert run.iterations == 3
        assert run.success is False
        assert len(run.steps) == 6
        assert "max iterations" in run.error.lower()

    @pytest.mark.asyncio
    async def test_zero_tokens_handled(self):
        """Test that zero token counts are handled correctly."""
        mock = Mock()
        mock.complete = AsyncMock(
            return_value=ProviderResponse(
                content="Answer",
                tool_calls=None,
                model="mock",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
            )
        )
        mock.model = "mock"
        mock.provider_name = "mock"

        agent = ReActAgent(provider=mock, tools=[])
        run = await agent.run("Query")

        assert run.total_input_tokens == 0
        assert run.total_output_tokens == 0
        assert run.total_tokens == 0


@pytest.mark.integration
@pytest.mark.requires_api_key
@pytest.mark.openai
class TestAgentMetricsIntegration:
    """Integration tests for metrics with real LLM providers."""

    @pytest.mark.asyncio
    async def test_real_provider_metrics(self):
        """Test metrics collection with real OpenAI provider."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        provider = get_provider("openai", model="gpt-4o-mini")
        agent = ReActAgent(provider=provider, tools=[])

        run = await agent.run("Say 'test' and nothing else.")

        assert run.total_input_tokens > 0
        assert run.total_output_tokens > 0
        assert run.total_latency_ms > 0
        assert run.iterations > 0
        assert run.success is True
        assert len(run.steps) == run.iterations

    @pytest.mark.asyncio
    async def test_real_tool_usage_metrics(self):
        """Test metrics with real tool usage."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        provider = get_provider("openai", model="gpt-4o-mini")
        registry = ToolRegistry()
        registry.register(SimpleTestTool())
        tools = registry.get_all_tools()
        executor = registry.get_executor()
        agent = ReActAgent(
            provider=provider, tools=tools, tool_executor=executor, max_iterations=5
        )

        run = await agent.run("Use the get_answer tool to answer: what is energy?")

        assert run.iterations >= 2
        assert run.total_input_tokens > 0
        assert run.total_output_tokens > 0

        has_tool_calls = any(step.tool_name for step in run.steps)
        assert has_tool_calls or run.iterations == 1
