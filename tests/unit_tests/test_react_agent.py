import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from energbench.agent.react_agent import ReActAgent
from energbench.agent.schema import ProviderResponse, StepType, ToolCall, ToolDefinition


def _make_provider(responses):
    p = MagicMock()
    p.provider_name = "mock"
    p.complete = AsyncMock(side_effect=responses)
    return p


def _text(text="done"):
    return ProviderResponse(content=text, tool_calls=None, input_tokens=5, output_tokens=3)


def _tool_call(name, args, call_id="tc1"):
    return ProviderResponse(
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
        input_tokens=5,
        output_tokens=3,
    )


def _agent(provider, tools=None, executor=None, max_iterations=5, max_retries=0):
    return ReActAgent(
        provider=provider,
        tools=tools or [],
        tool_executor=executor or (lambda name, args: "ok"),
        max_iterations=max_iterations,
        max_retries=max_retries,
        retry_base_delay=0.0,
    )


class TestReActAgentRun:
    """Unit tests for ReActAgent.run() execution loop."""

    @pytest.mark.asyncio
    async def test_run_returns_final_answer(self):
        """Agent returns a final answer when provider gives text immediately."""
        provider = _make_provider([_text("42")])
        agent = _agent(provider)

        run = await agent.run("What is 6 * 7?")

        assert run.success is True
        assert run.final_answer == "42"
        assert run.iterations == 1
        assert run.tool_calls_count == 0

    @pytest.mark.asyncio
    async def test_run_with_tool_call_then_answer(self):
        """Agent handles one tool call then returns final answer."""
        tool = ToolDefinition(name="my_tool", description="A test tool")
        provider = _make_provider([_tool_call("my_tool", {"x": 1}), _text("done")])
        agent = _agent(provider, tools=[tool], executor=lambda name, args: '{"value": 100}')

        run = await agent.run("Use my_tool")

        assert run.success is True
        assert run.tool_calls_count == 1
        assert run.iterations == 2
        step_types = [s.step_type for s in run.steps]
        assert StepType.ACTION in step_types
        assert StepType.OBSERVATION in step_types

    @pytest.mark.asyncio
    async def test_run_max_iterations_reached(self):
        """Agent fails with an error when max iterations is reached."""
        tool = ToolDefinition(name="my_tool", description="A test tool")
        provider = _make_provider([_tool_call("my_tool", {}), _tool_call("my_tool", {})])
        agent = _agent(provider, tools=[tool], max_iterations=2)

        run = await agent.run("Loop forever")

        assert run.success is False
        assert "Max iterations" in run.error
        assert run.iterations == 2

    @pytest.mark.asyncio
    async def test_run_unknown_tool_name(self):
        """Agent handles a call to an unregistered tool without raising."""
        provider = _make_provider([_tool_call("ghost_tool", {}), _text("ok")])
        agent = _agent(provider)  # no tools registered

        run = await agent.run("Use ghost_tool")

        assert run.success is True
        obs_steps = [s for s in run.steps if s.step_type == StepType.OBSERVATION]
        assert obs_steps
        assert "error" in obs_steps[0].content

    @pytest.mark.asyncio
    async def test_run_tool_executor_raises(self):
        """Agent records error observation when tool executor raises."""
        tool = ToolDefinition(name="my_tool", description="A test tool")
        bad_executor = MagicMock(side_effect=RuntimeError("boom"))
        provider = _make_provider([_tool_call("my_tool", {}), _text("recovered")])
        agent = _agent(provider, tools=[tool], executor=bad_executor)

        run = await agent.run("Use my_tool")

        assert run.success is True
        obs_steps = [s for s in run.steps if s.step_type == StepType.OBSERVATION]
        assert obs_steps
        obs_content = json.loads(obs_steps[0].content)
        assert "error" in obs_content

    @pytest.mark.asyncio
    async def test_run_provider_raises_exhausts_retries(self):
        """Agent fails gracefully when provider always raises."""
        provider = _make_provider(Exception("rate limited"))
        agent = _agent(provider, max_retries=0)

        run = await agent.run("What is the price?")

        assert run.success is False
        assert run.error
