import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from energbench.agent.schema import AgentRun, AgentStep, StepType
from energbench.observability import (
    CompositeObserver,
    JSONFileObserver,
    LangfuseObserver,
    get_observer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_run() -> AgentRun:
    """Create a realistic mock agent run with various step types including errors."""
    start_time = time.time()

    steps = [
        AgentStep(
            step_type=StepType.THOUGHT,
            content="I need to query the database to find energy prices.",
            timestamp=start_time,
            tokens_used=50,
            latency_ms=100,
        ),
        AgentStep(
            step_type=StepType.ACTION,
            content="Calling database tool",
            tool_name="query_database",
            tool_input={"query": "SELECT * FROM prices WHERE date > '2024-01-01'"},
            timestamp=start_time + 0.1,
            latency_ms=50,
        ),
        AgentStep(
            step_type=StepType.OBSERVATION,
            content="Tool returned results",
            tool_name="query_database",
            tool_output=json.dumps({
                "success": True,
                "rows": [
                    {"date": "2024-01-02", "price": 45.5, "region": "ERCOT"},
                    {"date": "2024-01-03", "price": 48.2, "region": "ERCOT"},
                ],
                "count": 2,
            }),
            timestamp=start_time + 0.5,
            latency_ms=400,
        ),
        AgentStep(
            step_type=StepType.ACTION,
            content="Calling API tool",
            tool_name="fetch_realtime_prices",
            tool_input={"region": "INVALID_REGION"},
            timestamp=start_time + 0.6,
            latency_ms=50,
        ),
        AgentStep(
            step_type=StepType.OBSERVATION,
            content="Tool returned error",
            tool_name="fetch_realtime_prices",
            tool_output=json.dumps({
                "success": False,
                "error": "Invalid region: INVALID_REGION. Valid regions are: ERCOT, CAISO, PJM",
                "error_type": "ValidationError",
            }),
            timestamp=start_time + 0.8,
            latency_ms=200,
        ),
        AgentStep(
            step_type=StepType.ERROR,
            content="The fetch_realtime_prices tool failed with validation error. "
                    "Proceeding with available data.",
            timestamp=start_time + 0.9,
            latency_ms=10,
        ),
        AgentStep(
            step_type=StepType.ANSWER,
            content="Based on the database query, ERCOT energy prices in early January 2024 "
                    "ranged from $45.50 to $48.20 per MWh.",
            timestamp=start_time + 1.0,
            tokens_used=80,
            latency_ms=150,
        ),
    ]

    return AgentRun(
        query="What are the current energy prices in ERCOT?",
        final_answer="Based on the database query, ERCOT energy prices in early January 2024 "
                     "ranged from $45.50 to $48.20 per MWh.",
        success=True,
        error=None,
        steps=steps,
        iterations=2,
        tool_calls_count=2,
        total_input_tokens=500,
        total_output_tokens=200,
        total_cached_tokens=100,
        total_latency_ms=960,
        start_time=start_time,
        end_time=start_time + 1.0,
    )


# ---------------------------------------------------------------------------
# JSONFileObserver tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestJSONObserver:
    """Unit tests for JSONFileObserver."""

    def test_init_creates_directory(self, tmp_path: Path):
        """Observer should create output directory on init."""
        output_dir = tmp_path / "traces"
        observer = JSONFileObserver(output_dir=str(output_dir))

        assert observer.is_enabled
        assert output_dir.exists()

    def test_init_with_run_name(self, tmp_path: Path):
        """Observer should create run_name subdirectory."""
        observer = JSONFileObserver(output_dir=str(tmp_path), run_name="no_tools")

        assert observer.output_dir == tmp_path / "no_tools"
        assert observer.output_dir.exists()

    def test_trace_agent_run_returns_trace_id(self, tmp_path: Path, mock_run: AgentRun):
        """trace_agent_run should return a non-None trace ID."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        trace_id = observer.trace_agent_run(run=mock_run)

        assert trace_id is not None

    def test_trace_captures_all_steps(self, tmp_path: Path, mock_run: AgentRun):
        """All steps should be captured in the trace."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        trace_id = observer.trace_agent_run(run=mock_run)

        trace_data = observer.load_trace(trace_id)
        assert trace_data is not None
        assert len(trace_data["steps"]) == 7

    def test_trace_step_types_correct(self, tmp_path: Path, mock_run: AgentRun):
        """Step types should be serialized in the correct order."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        trace_id = observer.trace_agent_run(run=mock_run)

        trace_data = observer.load_trace(trace_id)
        step_types = [s["step_type"] for s in trace_data["steps"]]
        expected = ["thought", "action", "observation", "action", "observation", "error", "answer"]
        assert step_types == expected

    def test_trace_captures_failed_tool_call(self, tmp_path: Path, mock_run: AgentRun):
        """Failed tool calls should be marked with is_error and error_details."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        trace_id = observer.trace_agent_run(run=mock_run)

        trace_data = observer.load_trace(trace_id)
        failed_obs = trace_data["steps"][4]
        assert failed_obs["is_error"] is True
        assert failed_obs["error_details"] is not None
        assert failed_obs["error_details"]["error_type"] == "ValidationError"

    def test_trace_captures_error_step(self, tmp_path: Path, mock_run: AgentRun):
        """Error steps should be captured with content."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        trace_id = observer.trace_agent_run(run=mock_run)

        trace_data = observer.load_trace(trace_id)
        error_step = trace_data["steps"][5]
        assert error_step["step_type"] == "error"
        assert "validation error" in error_step["content"].lower()

    def test_trace_preserves_full_tool_output(self, tmp_path: Path, mock_run: AgentRun):
        """Tool output should be preserved in full (not truncated)."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        trace_id = observer.trace_agent_run(run=mock_run)

        trace_data = observer.load_trace(trace_id)
        successful_obs = trace_data["steps"][2]
        tool_output = json.loads(successful_obs["tool_output"])
        assert tool_output["count"] == 2
        assert len(tool_output["rows"]) == 2

    def test_trace_captures_step_summary(self, tmp_path: Path, mock_run: AgentRun):
        """Step summary should include failed tool calls and errors."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        trace_id = observer.trace_agent_run(run=mock_run)

        trace_data = observer.load_trace(trace_id)
        summary = trace_data["step_summary"]
        assert len(summary["failed_tool_calls"]) == 1
        assert len(summary["errors"]) == 1
        assert summary["total_steps"] == 7

    def test_trace_captures_metrics(self, tmp_path: Path, mock_run: AgentRun):
        """Metrics should be accurately captured."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        trace_id = observer.trace_agent_run(run=mock_run)

        trace_data = observer.load_trace(trace_id)
        metrics = trace_data["metrics"]
        assert metrics["total_input_tokens"] == 500
        assert metrics["total_output_tokens"] == 200
        assert metrics["total_cached_tokens"] == 100
        assert metrics["tool_calls_count"] == 2
        assert metrics["iterations"] == 2

    def test_trace_captures_metadata_and_tags(self, tmp_path: Path, mock_run: AgentRun):
        """Metadata, tags, user_id, session_id should be captured."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        trace_id = observer.trace_agent_run(
            run=mock_run,
            metadata={"test": True, "experiment": "obs_test"},
            tags=["test", "mock"],
            user_id="test_user",
            session_id="test_session",
        )

        trace_data = observer.load_trace(trace_id)
        assert trace_data["metadata"]["test"] is True
        assert trace_data["tags"] == ["test", "mock"]
        assert trace_data["user_id"] == "test_user"
        assert trace_data["session_id"] == "test_session"

    def test_disabled_observer_returns_none(self, tmp_path: Path, mock_run: AgentRun):
        """Disabled observer should return None without writing."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        observer._enabled = False

        result = observer.trace_agent_run(run=mock_run)
        assert result is None

    def test_shutdown_disables_observer(self, tmp_path: Path):
        """shutdown() should disable the observer."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        assert observer.is_enabled

        observer.shutdown()
        assert not observer.is_enabled

    # -- single_file (JSONL) mode --

    def test_single_file_mode_appends(self, tmp_path: Path, mock_run: AgentRun):
        """Single-file mode should append traces to one JSONL file."""
        observer = JSONFileObserver(output_dir=str(tmp_path), single_file=True)

        id1 = observer.trace_agent_run(run=mock_run, tags=["first"])
        id2 = observer.trace_agent_run(run=mock_run, tags=["second"])

        assert id1 is not None
        assert id2 is not None
        assert id1 != id2

        jsonl_path = tmp_path / "agent_traces.jsonl"
        assert jsonl_path.exists()

        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_single_file_list_traces(self, tmp_path: Path, mock_run: AgentRun):
        """list_traces should return all trace IDs in JSONL mode."""
        observer = JSONFileObserver(output_dir=str(tmp_path), single_file=True)

        id1 = observer.trace_agent_run(run=mock_run)
        id2 = observer.trace_agent_run(run=mock_run)

        traces = observer.list_traces()
        assert len(traces) == 2
        assert id1 in traces
        assert id2 in traces

    def test_single_file_load_trace(self, tmp_path: Path, mock_run: AgentRun):
        """load_trace should find the correct trace in JSONL mode."""
        observer = JSONFileObserver(output_dir=str(tmp_path), single_file=True)

        id1 = observer.trace_agent_run(run=mock_run, tags=["first"])
        id2 = observer.trace_agent_run(run=mock_run, tags=["second"])

        trace1 = observer.load_trace(id1)
        assert trace1 is not None
        assert trace1["tags"] == ["first"]

        trace2 = observer.load_trace(id2)
        assert trace2 is not None
        assert trace2["tags"] == ["second"]

    def test_single_file_get_trace_file(self, tmp_path: Path):
        """get_trace_file in single-file mode should return the JSONL path."""
        observer = JSONFileObserver(output_dir=str(tmp_path), single_file=True)
        result = observer.get_trace_file("any_id")
        assert result == tmp_path / "agent_traces.jsonl"

    # -- list_traces / load_trace / get_trace_file for individual files --

    def test_list_traces_individual_files(self, tmp_path: Path, mock_run: AgentRun):
        """list_traces should find all traces in individual file mode."""
        observer = JSONFileObserver(output_dir=str(tmp_path))

        id1 = observer.trace_agent_run(run=mock_run)
        id2 = observer.trace_agent_run(run=mock_run)

        traces = observer.list_traces()
        assert len(traces) == 2
        assert id1 in traces
        assert id2 in traces

    def test_get_trace_file_individual(self, tmp_path: Path, mock_run: AgentRun):
        """get_trace_file should return the correct path."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        trace_id = observer.trace_agent_run(run=mock_run)

        result = observer.get_trace_file(trace_id)
        assert result is not None
        assert result.exists()

    def test_get_trace_file_nonexistent(self, tmp_path: Path):
        """get_trace_file should return None for a nonexistent trace."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        result = observer.get_trace_file("nonexistent_id")
        assert result is None

    def test_load_trace_nonexistent(self, tmp_path: Path):
        """load_trace should return None for a nonexistent trace."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        result = observer.load_trace("nonexistent_id")
        assert result is None

    # -- Organized subdirectory traces --

    def test_organized_trace_write_and_load(self, tmp_path: Path, mock_run: AgentRun):
        """Traces with provider/model/question_id metadata should be written to subdirs."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        metadata = {"provider": "openai", "model": "gpt-4o", "question_id": 1}

        trace_id = observer.trace_agent_run(run=mock_run, metadata=metadata)
        assert trace_id is not None

        # Verify subdirectory structure
        subdir = tmp_path / "openai_gpt-4o"
        assert subdir.exists()
        files = list(subdir.glob("trace_q1_*.json"))
        assert len(files) == 1

        # Verify load_trace can find it
        trace_data = observer.load_trace(trace_id)
        assert trace_data is not None
        assert trace_data["trace_id"] == trace_id

    def test_organized_trace_list_traces(self, tmp_path: Path, mock_run: AgentRun):
        """list_traces should find traces in model subdirectories."""
        observer = JSONFileObserver(output_dir=str(tmp_path))

        # Write one flat trace and one organized trace
        flat_id = observer.trace_agent_run(run=mock_run)
        organized_id = observer.trace_agent_run(
            run=mock_run,
            metadata={"provider": "openai", "model": "gpt-4o", "question_id": 1},
        )

        traces = observer.list_traces()
        assert len(traces) == 2
        assert flat_id in traces
        assert organized_id in traces

    def test_organized_trace_get_trace_file(self, tmp_path: Path, mock_run: AgentRun):
        """get_trace_file should find organized traces in subdirectories."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        metadata = {"provider": "anthropic", "model": "claude-3", "question_id": 5}

        trace_id = observer.trace_agent_run(run=mock_run, metadata=metadata)
        result = observer.get_trace_file(trace_id)

        assert result is not None
        assert result.exists()
        assert "anthropic_claude-3" in str(result)

    # -- _is_tool_error edge cases --

    def test_is_tool_error_with_error_key(self, tmp_path: Path):
        """Should detect error when 'error' key has a truthy value."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        assert observer._is_tool_error(json.dumps({"error": "something failed"})) is True

    def test_is_tool_error_with_success_false(self, tmp_path: Path):
        """Should detect error when 'success' is explicitly False."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        assert observer._is_tool_error(json.dumps({"success": False})) is True

    def test_is_tool_error_no_false_positive(self, tmp_path: Path):
        """Should NOT flag output that incidentally mentions 'error' in values."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        # This used to be a false positive with the old string-matching heuristic
        output = json.dumps({"message": "No errors found", "status": "ok"})
        assert observer._is_tool_error(output) is False

    def test_is_tool_error_success_true(self, tmp_path: Path):
        """Should NOT flag successful tool output."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        assert observer._is_tool_error(json.dumps({"success": True, "data": [1, 2]})) is False

    def test_is_tool_error_invalid_json(self, tmp_path: Path):
        """Should return False for non-JSON output."""
        observer = JSONFileObserver(output_dir=str(tmp_path))
        assert observer._is_tool_error("this is not json") is False


# ---------------------------------------------------------------------------
# LangfuseObserver tests (mocked)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLangfuseObserver:
    """Unit tests for LangfuseObserver using mocked Langfuse client."""

    def test_disabled_when_langfuse_unavailable(self):
        """Observer should be disabled when Langfuse package is not available."""
        with patch("energbench.observability.langfuse_client.LANGFUSE_AVAILABLE", False):
            observer = LangfuseObserver()
            assert not observer.is_enabled

    def test_disabled_when_no_credentials(self, monkeypatch):
        """Observer should be disabled when credentials are missing."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        observer = LangfuseObserver()
        assert not observer.is_enabled

    @patch("energbench.observability.langfuse_client.LANGFUSE_AVAILABLE", True)
    def test_enabled_with_mock_client(self):
        """Observer should be enabled when a client is provided."""
        mock_client = MagicMock()
        observer = LangfuseObserver(client=mock_client)
        assert observer.is_enabled

    @patch("energbench.observability.langfuse_client.LANGFUSE_AVAILABLE", True)
    def test_trace_agent_run_with_mock_client(self, mock_run: AgentRun):
        """trace_agent_run should create a trace and spans via the client."""
        mock_client = MagicMock()
        mock_trace = MagicMock()
        mock_trace.id = "langfuse-trace-123"
        mock_client.trace.return_value = mock_trace

        observer = LangfuseObserver(client=mock_client)
        trace_id = observer.trace_agent_run(
            run=mock_run,
            metadata={"test": True},
            tags=["test"],
            user_id="test_user",
        )

        assert trace_id == "langfuse-trace-123"
        mock_client.trace.assert_called_once()
        # Should create one span per step (7 steps)
        assert mock_trace.span.call_count == len(mock_run.steps)
        mock_client.flush.assert_called_once()

    def test_trace_agent_run_disabled_returns_none(self, mock_run: AgentRun):
        """Disabled observer should return None without calling client."""
        mock_client = MagicMock()
        observer = LangfuseObserver(client=mock_client, enabled=False)

        result = observer.trace_agent_run(run=mock_run)
        assert result is None
        mock_client.trace.assert_not_called()

    @patch("energbench.observability.langfuse_client.LANGFUSE_AVAILABLE", True)
    def test_trace_llm_call_with_mock_client(self):
        """trace_llm_call should call client.generation."""
        mock_client = MagicMock()
        observer = LangfuseObserver(client=mock_client)

        observer.trace_llm_call(
            trace_id="trace-1",
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            response="Hi there!",
            input_tokens=10,
            output_tokens=5,
        )

        mock_client.generation.assert_called_once()

    @patch("energbench.observability.langfuse_client.LANGFUSE_AVAILABLE", True)
    def test_trace_tool_execution_with_mock_client(self):
        """trace_tool_execution should call client.span."""
        mock_client = MagicMock()
        observer = LangfuseObserver(client=mock_client)

        observer.trace_tool_execution(
            trace_id="trace-1",
            tool_name="search",
            arguments={"query": "energy prices"},
            result='{"results": []}',
        )

        mock_client.span.assert_called_once()
        call_kwargs = mock_client.span.call_args[1]
        assert call_kwargs["name"] == "tool_search"
        assert call_kwargs["level"] == "DEFAULT"

    @patch("energbench.observability.langfuse_client.LANGFUSE_AVAILABLE", True)
    def test_trace_tool_execution_with_error(self):
        """trace_tool_execution should set ERROR level when error is provided."""
        mock_client = MagicMock()
        observer = LangfuseObserver(client=mock_client)

        observer.trace_tool_execution(
            trace_id="trace-1",
            tool_name="search",
            arguments={"query": "test"},
            result='{"error": "failed"}',
            error="Request failed",
        )

        call_kwargs = mock_client.span.call_args[1]
        assert call_kwargs["level"] == "ERROR"

    def test_flush_calls_client(self):
        """flush() should call client.flush()."""
        mock_client = MagicMock()
        # Manually set up the observer without going through __init__'s LANGFUSE_AVAILABLE check
        observer = LangfuseObserver.__new__(LangfuseObserver)
        observer._enabled = True
        observer.client = mock_client
        observer.flush()
        mock_client.flush.assert_called_once()

    def test_shutdown_calls_client(self):
        """shutdown() should call client.shutdown()."""
        mock_client = MagicMock()
        observer = LangfuseObserver.__new__(LangfuseObserver)
        observer._enabled = True
        observer.client = mock_client
        observer.shutdown()
        mock_client.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# CompositeObserver tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCompositeObserver:
    """Unit tests for CompositeObserver."""

    def test_enabled_with_at_least_one_observer(self, tmp_path: Path):
        """Should be enabled when at least one observer is enabled."""
        json_obs = JSONFileObserver(output_dir=str(tmp_path))
        composite = CompositeObserver([json_obs])
        assert composite.is_enabled

    def test_disabled_with_no_enabled_observers(self):
        """Should be disabled when all observers are disabled."""
        composite = CompositeObserver([])
        assert not composite.is_enabled

    def test_filters_disabled_observers(self, tmp_path: Path):
        """Should filter out disabled observers at init."""
        enabled_obs = JSONFileObserver(output_dir=str(tmp_path))
        disabled_obs = JSONFileObserver(output_dir=str(tmp_path))
        disabled_obs._enabled = False

        composite = CompositeObserver([enabled_obs, disabled_obs])
        assert len(composite.observers) == 1

    def test_trace_agent_run_delegates(self, tmp_path: Path, mock_run: AgentRun):
        """trace_agent_run should delegate to all observers."""
        obs1 = JSONFileObserver(output_dir=str(tmp_path / "obs1"))
        obs2 = JSONFileObserver(output_dir=str(tmp_path / "obs2"))
        composite = CompositeObserver([obs1, obs2])

        trace_id = composite.trace_agent_run(run=mock_run, tags=["composite"])
        assert trace_id is not None

        # Both observers should have written a trace
        assert len(list((tmp_path / "obs1").glob("trace_*.json"))) == 1
        assert len(list((tmp_path / "obs2").glob("trace_*.json"))) == 1

    def test_trace_agent_run_returns_first_id(self, tmp_path: Path, mock_run: AgentRun):
        """Should return the first successful trace ID."""
        obs = JSONFileObserver(output_dir=str(tmp_path))
        composite = CompositeObserver([obs])

        trace_id = composite.trace_agent_run(run=mock_run)
        assert trace_id is not None

    def test_add_observer(self, tmp_path: Path):
        """add_observer should add an enabled observer."""
        composite = CompositeObserver([])
        assert len(composite.observers) == 0

        obs = JSONFileObserver(output_dir=str(tmp_path))
        composite.add_observer(obs)
        assert len(composite.observers) == 1

    def test_add_disabled_observer_ignored(self, tmp_path: Path):
        """add_observer should ignore disabled observers."""
        composite = CompositeObserver([])
        obs = JSONFileObserver(output_dir=str(tmp_path))
        obs._enabled = False

        composite.add_observer(obs)
        assert len(composite.observers) == 0

    def test_remove_observer(self, tmp_path: Path):
        """remove_observer should remove observers by type."""
        obs = JSONFileObserver(output_dir=str(tmp_path))
        composite = CompositeObserver([obs])
        assert len(composite.observers) == 1

        removed = composite.remove_observer(JSONFileObserver)
        assert removed is True
        assert len(composite.observers) == 0

    def test_remove_observer_nonexistent_type(self, tmp_path: Path):
        """remove_observer should return False if type not found."""
        obs = JSONFileObserver(output_dir=str(tmp_path))
        composite = CompositeObserver([obs])

        removed = composite.remove_observer(LangfuseObserver)
        assert removed is False
        assert len(composite.observers) == 1

    def test_flush_delegates(self, tmp_path: Path):
        """flush() should call flush on all observers."""
        obs = MagicMock(spec=JSONFileObserver)
        obs.is_enabled = True
        composite = CompositeObserver([obs])

        composite.flush()
        obs.flush.assert_called_once()

    def test_shutdown_delegates(self, tmp_path: Path):
        """shutdown() should call shutdown on all observers."""
        obs = MagicMock(spec=JSONFileObserver)
        obs.is_enabled = True
        composite = CompositeObserver([obs])

        composite.shutdown()
        obs.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# get_observer factory tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestObserverFactory:
    """Unit tests for the get_observer factory function."""

    def test_json_backend(self, tmp_path: Path):
        """get_observer('json') should return JSONFileObserver."""
        obs = get_observer("json", output_dir=str(tmp_path))
        assert isinstance(obs, JSONFileObserver)
        assert obs.is_enabled

    def test_both_backend(self, tmp_path: Path):
        """get_observer('both') should return CompositeObserver."""
        obs = get_observer("both", output_dir=str(tmp_path))
        assert isinstance(obs, CompositeObserver)
        assert obs.is_enabled

    def test_auto_backend(self, tmp_path: Path):
        """get_observer('auto') should return an enabled observer."""
        obs = get_observer("auto", output_dir=str(tmp_path))
        assert obs.is_enabled

    def test_invalid_backend_raises(self):
        """get_observer with invalid backend should raise ValueError."""
        with pytest.raises(ValueError, match="invalid"):
            get_observer("invalid")

    def test_case_insensitive(self, tmp_path: Path):
        """Backend name should be case-insensitive."""
        obs = get_observer("JSON", output_dir=str(tmp_path))
        assert isinstance(obs, JSONFileObserver)

    def test_json_with_run_name(self, tmp_path: Path):
        """run_name should be passed through to JSONFileObserver."""
        obs = get_observer("json", output_dir=str(tmp_path), run_name="experiment_1")
        assert isinstance(obs, JSONFileObserver)
        assert obs.run_name == "experiment_1"
        assert obs.output_dir == tmp_path / "experiment_1"

    def test_json_with_single_file(self, tmp_path: Path):
        """single_file should be passed through to JSONFileObserver."""
        obs = get_observer("json", output_dir=str(tmp_path), single_file=True)
        assert isinstance(obs, JSONFileObserver)
        assert obs.single_file is True
