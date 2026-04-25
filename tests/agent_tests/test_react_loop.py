"""Tests for ReActAgent loop behavior (max_iterations, final answer, non-recoverable errors, timeouts)."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from energyevals.agent.react_agent import ReActAgent
from energyevals.agent.schema import ToolCall, ToolDefinition
from energyevals.agent.schema.responses import ProviderResponse


def _make_tool_response(**kwargs) -> ProviderResponse:
    """Build a ProviderResponse that requests a tool call."""
    defaults = dict(
        content="Let me use a tool.",
        tool_calls=[ToolCall(id="tc1", name="my_tool", arguments={})],
        input_tokens=10,
        cached_tokens=0,
        output_tokens=5,
        reasoning_tokens=0,
        latency_ms=50.0,
        model="test-model",
    )
    defaults.update(kwargs)
    return ProviderResponse(**defaults)


def _make_answer_response(answer: str = "FINAL ANSWER: 42") -> ProviderResponse:
    """Build a ProviderResponse that contains a final answer (no tool calls)."""
    return ProviderResponse(
        content=answer,
        tool_calls=None,
        input_tokens=10,
        cached_tokens=0,
        output_tokens=20,
        reasoning_tokens=0,
        latency_ms=50.0,
        model="test-model",
    )


@pytest.fixture
def mock_provider():
    mock = MagicMock()
    mock.model = "test-model"
    mock.provider_name = "mock"
    # Default: provide a tool response (no final answer)
    mock.complete = AsyncMock(return_value=_make_tool_response())
    return mock


@pytest.fixture
def tool_defs():
    return [
        ToolDefinition(
            name="my_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )
    ]


class TestMaxIterations:
    """Agent must stop after max_iterations when no final answer is produced."""

    @pytest.mark.asyncio
    async def test_stops_at_max_iterations(self, mock_provider, tool_defs):
        # Tool executor returns a benign recoverable result each time
        async def executor(tool_name: str, args: dict) -> str:
            return json.dumps({"success": True, "data": "ok", "metadata": {}})

        agent = ReActAgent(
            provider=mock_provider,
            tools=tool_defs,
            tool_executor=executor,
            max_iterations=3,
        )

        run = await agent.run("What is the answer?")

        assert run.iterations == 3
        assert run.success is False
        assert "Max iterations" in (run.error or "")
        assert mock_provider.complete.await_count == 3


class TestFinalAnswer:
    """Agent must stop and mark success when the model returns a final answer."""

    @pytest.mark.asyncio
    async def test_stops_on_final_answer(self, mock_provider, tool_defs):
        # First call: tool response; second call: final answer
        mock_provider.complete = AsyncMock(side_effect=[
            _make_tool_response(),
            _make_answer_response("FINAL ANSWER: 42"),
        ])

        async def executor(tool_name: str, args: dict) -> str:
            return json.dumps({"success": True, "data": "result", "metadata": {}})

        agent = ReActAgent(
            provider=mock_provider,
            tools=tool_defs,
            tool_executor=executor,
            max_iterations=10,
        )

        run = await agent.run("Compute something.")

        assert run.success is True
        assert run.iterations == 2
        assert mock_provider.complete.await_count == 2
        assert "42" in (run.final_answer or "")


class TestNonRecoverableTool:
    """Agent must stop immediately when a tool returns recoverable=False."""

    @pytest.mark.asyncio
    async def test_nonrecoverable_tool_stops_loop(self, mock_provider, tool_defs):
        call_count = 0

        async def executor(tool_name: str, args: dict) -> str:
            nonlocal call_count
            call_count += 1
            return json.dumps({
                "success": False,
                "data": None,
                "error": "Fatal tool error",
                "metadata": {"recoverable": False},
            })

        agent = ReActAgent(
            provider=mock_provider,
            tools=tool_defs,
            tool_executor=executor,
            max_iterations=10,
        )

        run = await agent.run("Do something.")

        # Should stop after 1 iteration (1 LLM call + 1 tool call)
        assert run.success is False
        assert "Fatal tool error" in (run.error or "")
        assert run.iterations < 10
        assert call_count == 1


class TestToolTimeout:
    """Agent must handle tool timeouts without crashing the loop."""

    @pytest.mark.asyncio
    async def test_tool_timeout_is_recorded(self, mock_provider, tool_defs):
        # Tool sleeps longer than timeout — triggers TimeoutError inside _execute_tool
        async def slow_executor(tool_name: str, args: dict) -> str:
            await asyncio.sleep(10)  # way longer than timeout
            return "{}"

        # Model always requests the tool (no final answer) so we set max_iterations=1
        agent = ReActAgent(
            provider=mock_provider,
            tools=tool_defs,
            tool_executor=slow_executor,
            max_iterations=1,
            tool_timeout=0.05,  # 50ms — very short for testing
        )

        run = await agent.run("Use my_tool.")

        # The timeout is caught and recorded as a timed-out observation, not a crash
        step_types = [s.step_type.value for s in run.steps]
        assert "observation" in step_types
        obs_steps = [s for s in run.steps if s.step_type.value == "observation"]
        assert any("timed out" in (s.content or "").lower() for s in obs_steps)
        # Run should complete (not raise) even when tool times out
        assert run.iterations == 1
