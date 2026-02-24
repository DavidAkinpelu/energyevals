from energbench.evaluation.data_loader import _parse_latency_breakdown


class TestLatencyBreakdown:
    """Tests for _parse_latency_breakdown (Issue #5)."""

    def test_single_action_counted_once(self):
        steps = [
            {"step_type": "action", "latency_ms": 100.0, "timestamp_unix": 1000.0},
            {"step_type": "observation", "latency_ms": 50.0, "tool_name": "search"},
        ]
        result = _parse_latency_breakdown(steps)
        assert result.llm_thinking_ms == 100.0
        assert result.tool_execution_ms == 50.0

    def test_multi_tool_call_not_double_counted(self):
        """Multiple action steps from the same LLM call (same timestamp)
        should only count the LLM latency once."""
        steps = [
            {"step_type": "action", "latency_ms": 200.0, "timestamp_unix": 1000.0, "tool_name": "tool_a"},
            {"step_type": "observation", "latency_ms": 30.0, "tool_name": "tool_a"},
            {"step_type": "action", "latency_ms": 200.0, "timestamp_unix": 1000.0, "tool_name": "tool_b"},
            {"step_type": "observation", "latency_ms": 40.0, "tool_name": "tool_b"},
            {"step_type": "action", "latency_ms": 200.0, "timestamp_unix": 1000.0, "tool_name": "tool_c"},
            {"step_type": "observation", "latency_ms": 20.0, "tool_name": "tool_c"},
        ]
        result = _parse_latency_breakdown(steps)
        assert result.llm_thinking_ms == 200.0
        assert result.tool_execution_ms == 90.0

    def test_separate_llm_calls_both_counted(self):
        """Action steps from different LLM calls (different timestamps)
        should each be counted."""
        steps = [
            {"step_type": "action", "latency_ms": 150.0, "timestamp_unix": 1000.0},
            {"step_type": "observation", "latency_ms": 25.0, "tool_name": "search"},
            {"step_type": "action", "latency_ms": 120.0, "timestamp_unix": 2000.0},
            {"step_type": "observation", "latency_ms": 35.0, "tool_name": "search"},
        ]
        result = _parse_latency_breakdown(steps)
        assert result.llm_thinking_ms == 270.0
        assert result.tool_execution_ms == 60.0

    def test_final_answer_included_in_llm_time(self):
        """The final answer step must contribute to llm_thinking_ms."""
        steps = [
            {"step_type": "action", "latency_ms": 100.0, "timestamp_unix": 1000.0},
            {"step_type": "observation", "latency_ms": 50.0, "tool_name": "search"},
            {"step_type": "answer", "latency_ms": 80.0},
        ]
        result = _parse_latency_breakdown(steps)
        assert result.llm_thinking_ms == 180.0
        assert result.tool_execution_ms == 50.0
        assert result.wall_clock_ms == 230.0

    def test_per_tool_ms_aggregation(self):
        steps = [
            {"step_type": "observation", "latency_ms": 10.0, "tool_name": "search"},
            {"step_type": "observation", "latency_ms": 20.0, "tool_name": "search"},
            {"step_type": "observation", "latency_ms": 30.0, "tool_name": "database"},
        ]
        result = _parse_latency_breakdown(steps)
        assert result.per_tool_ms["search"] == 30.0
        assert result.per_tool_ms["database"] == 30.0

    def test_empty_steps(self):
        result = _parse_latency_breakdown([])
        assert result.llm_thinking_ms == 0.0
        assert result.tool_execution_ms == 0.0
        assert result.wall_clock_ms == 0.0
