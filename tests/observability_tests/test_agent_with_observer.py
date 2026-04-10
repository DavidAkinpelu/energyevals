import os
from pathlib import Path

import pytest

from energyevals.agent.providers import get_provider
from energyevals.agent.react_agent import ReActAgent
from energyevals.observability import JSONFileObserver
from energyevals.tools import create_default_registry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tool_registry():
    """Create a default tool registry."""
    return create_default_registry()


@pytest.fixture
def agent(tool_registry):
    """Create a ReActAgent with an OpenAI provider and default tools."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    provider = get_provider("openai", model="gpt-4o-mini")
    tools = tool_registry.get_all_tools()

    async def tool_executor(tool_name: str, arguments: dict) -> str:
        result = await tool_registry.execute(tool_name, **arguments)
        return result.to_json()

    return ReActAgent(
        provider=provider,
        tools=tools,
        tool_executor=tool_executor,
        max_iterations=5,
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.requires_api_key
class TestAgentWithObserver:
    """Integration tests: real agent runs traced by observers."""

    @pytest.mark.asyncio
    async def test_agent_with_json_observer(self, agent, tmp_path: Path):
        """Agent run should be fully captured by JSONFileObserver."""
        observer = JSONFileObserver(
            output_dir=str(tmp_path),
            single_file=False,
            pretty_print=True,
        )

        query = "What is 15 multiplied by 7? Use the calculator tool."
        run = await agent.run(query)

        trace_id = observer.trace_agent_run(
            run=run,
            metadata={"test": "agent_integration"},
            tags=["test", "agent", "calculator"],
            user_id="integration_test",
        )

        assert trace_id is not None, "Should create trace"

        trace_data = observer.load_trace(trace_id)
        assert trace_data is not None, "Should load trace"

        # All steps captured
        assert len(trace_data["steps"]) == len(run.steps)

        # Metrics match the run
        metrics = trace_data["metrics"]
        assert metrics["tool_calls_count"] == run.tool_calls_count
        assert metrics["iterations"] == run.iterations

        # Query and answer captured
        assert trace_data["query"] == query
        assert trace_data["final_answer"] is not None

        # Trace file exists on disk
        trace_file = observer.get_trace_file(trace_id)
        assert trace_file is not None
        assert trace_file.exists()

    @pytest.mark.asyncio
    async def test_agent_with_named_json_observer(self, agent, tmp_path: Path):
        """Agent run with a named JSON observer should write into the run subdirectory."""
        observer = JSONFileObserver(output_dir=str(tmp_path), run_name="integration_run")

        query = "Calculate the square root of 144 using the calculator."
        run = await agent.run(query)

        trace_id = observer.trace_agent_run(
            run=run,
            metadata={"test": "named_json_observer"},
            tags=["test", "json"],
        )

        assert trace_id is not None, "Should create trace"

        json_files = list((tmp_path / "integration_run").glob("trace_*.json"))
        assert len(json_files) >= 1, "Should have at least one JSON trace file"

        observer.flush()
        observer.shutdown()
