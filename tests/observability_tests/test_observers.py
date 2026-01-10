"""Test observability backends.

Tests both JSON file observer and Langfuse observer with real agent runs.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from energbench.observability import (
    get_observer,
    JSONFileObserver,
    LangfuseObserver,
    CompositeObserver,
)
from energbench.agent.schema import AgentRun, AgentStep, StepType


def create_mock_run() -> AgentRun:
    """Create a mock agent run with various step types including errors."""
    import time

    start_time = time.time()

    steps = [
        # Thought step
        AgentStep(
            step_type=StepType.THOUGHT,
            content="I need to query the database to find energy prices.",
            timestamp=start_time,
            tokens_used=50,
            latency_ms=100,
        ),
        # Action step - tool call
        AgentStep(
            step_type=StepType.ACTION,
            content="Calling database tool",
            tool_name="query_database",
            tool_input={"query": "SELECT * FROM prices WHERE date > '2024-01-01'"},
            timestamp=start_time + 0.1,
            latency_ms=50,
        ),
        # Observation step - successful tool result
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
        # Another action - this one will fail
        AgentStep(
            step_type=StepType.ACTION,
            content="Calling API tool",
            tool_name="fetch_realtime_prices",
            tool_input={"region": "INVALID_REGION"},
            timestamp=start_time + 0.6,
            latency_ms=50,
        ),
        # Observation step - FAILED tool result
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
        # Error step
        AgentStep(
            step_type=StepType.ERROR,
            content="The fetch_realtime_prices tool failed with validation error. Proceeding with available data.",
            timestamp=start_time + 0.9,
            latency_ms=10,
        ),
        # Final answer
        AgentStep(
            step_type=StepType.ANSWER,
            content="Based on the database query, ERCOT energy prices in early January 2024 ranged from $45.50 to $48.20 per MWh. Note: Real-time price fetch failed due to invalid region.",
            timestamp=start_time + 1.0,
            tokens_used=80,
            latency_ms=150,
        ),
    ]

    return AgentRun(
        query="What are the current energy prices in ERCOT?",
        final_answer="Based on the database query, ERCOT energy prices in early January 2024 ranged from $45.50 to $48.20 per MWh.",
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


def test_json_observer():
    """Test JSON file observer captures all data including failures."""
    print("\n" + "="*60)
    print("  JSON OBSERVER TEST")
    print("="*60)

    import tempfile
    import shutil

    # Create temp directory for traces
    temp_dir = tempfile.mkdtemp(prefix="energbench_trace_test_")

    try:
        # Initialize observer
        observer = JSONFileObserver(
            output_dir=temp_dir,
            single_file=False,
            pretty_print=True,
        )

        assert observer.is_enabled, "Observer should be enabled"
        print(f"  [OK] Observer initialized at {temp_dir}")

        # Create mock run with failures
        run = create_mock_run()

        # Trace the run
        trace_id = observer.trace_agent_run(
            run=run,
            metadata={"test": True, "experiment": "observer_test"},
            tags=["test", "mock"],
            user_id="test_user",
            session_id="test_session",
        )

        assert trace_id is not None, "Should return trace ID"
        print(f"  [OK] Trace created: {trace_id}")

        # Load and verify trace
        trace_data = observer.load_trace(trace_id)
        assert trace_data is not None, "Should load trace data"

        # Verify all steps captured
        assert len(trace_data["steps"]) == 7, f"Should have 7 steps, got {len(trace_data['steps'])}"
        print(f"  [OK] All {len(trace_data['steps'])} steps captured")

        # Verify step types
        step_types = [s["step_type"] for s in trace_data["steps"]]
        expected_types = ["thought", "action", "observation", "action", "observation", "error", "answer"]
        assert step_types == expected_types, f"Step types mismatch: {step_types}"
        print("  [OK] Step types correct")

        # Verify failed tool call captured
        failed_observation = trace_data["steps"][4]
        assert failed_observation["is_error"] == True, "Should mark as error"
        assert failed_observation["error_details"] is not None, "Should have error details"
        print("  [OK] Failed tool call captured with error details")

        # Verify error step captured
        error_step = trace_data["steps"][5]
        assert error_step["step_type"] == "error", "Should be error step"
        assert "validation error" in error_step["content"].lower(), "Should contain error content"
        print("  [OK] Error step captured")

        # Verify step summary includes failures
        summary = trace_data["step_summary"]
        assert len(summary["failed_tool_calls"]) == 1, "Should have 1 failed tool call"
        assert len(summary["errors"]) == 1, "Should have 1 error"
        print("  [OK] Step summary includes failures")

        # Verify full tool output (not truncated)
        successful_obs = trace_data["steps"][2]
        tool_output = json.loads(successful_obs["tool_output"])
        assert tool_output["count"] == 2, "Should have full tool output"
        assert len(tool_output["rows"]) == 2, "Should have all rows"
        print("  [OK] Full tool output preserved (not truncated)")

        # Verify metrics
        metrics = trace_data["metrics"]
        assert metrics["total_input_tokens"] == 500, "Should have correct input tokens"
        assert metrics["total_cached_tokens"] == 100, "Should have cached tokens"
        assert metrics["tool_calls_count"] == 2, "Should have correct tool call count"
        print("  [OK] Metrics captured correctly")

        # Verify metadata
        assert trace_data["metadata"]["test"] == True, "Should have metadata"
        assert trace_data["tags"] == ["test", "mock"], "Should have tags"
        assert trace_data["user_id"] == "test_user", "Should have user_id"
        print("  [OK] Metadata, tags, user_id captured")

        print("\n  [PASS] JSON Observer Test")
        return True

    except Exception as e:
        print(f"\n  [FAIL] JSON Observer Test: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_langfuse_observer():
    """Test Langfuse observer (if available)."""
    print("\n" + "="*60)
    print("  LANGFUSE OBSERVER TEST")
    print("="*60)

    try:
        observer = LangfuseObserver()

        if not observer.is_enabled:
            print("  [SKIP] Langfuse not configured (missing credentials)")
            return True

        print("  [OK] Langfuse observer initialized")

        # Create mock run
        run = create_mock_run()

        # Trace the run
        trace_id = observer.trace_agent_run(
            run=run,
            metadata={"test": True},
            tags=["test", "observability_test"],
            user_id="test_user",
            session_id="test_session",
        )

        if trace_id:
            print(f"  [OK] Trace created in Langfuse: {trace_id}")
        else:
            print("  [WARN] Trace ID not returned (may still be uploaded)")

        # Flush to ensure data is sent
        observer.flush()
        print("  [OK] Trace flushed to Langfuse")

        observer.shutdown()
        print("  [OK] Observer shutdown")

        print("\n  [PASS] Langfuse Observer Test")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Langfuse Observer Test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_composite_observer():
    """Test composite observer with both backends."""
    print("\n" + "="*60)
    print("  COMPOSITE OBSERVER TEST")
    print("="*60)

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="energbench_composite_test_")

    try:
        # Create composite observer
        observers = [
            JSONFileObserver(output_dir=temp_dir),
        ]

        # Try adding Langfuse if available
        langfuse_obs = LangfuseObserver()
        if langfuse_obs.is_enabled:
            observers.append(langfuse_obs)
            print("  [OK] Added Langfuse to composite")

        composite = CompositeObserver(observers)
        assert composite.is_enabled, "Composite should be enabled"
        print(f"  [OK] Composite observer with {len(composite.observers)} backends")

        # Create mock run
        run = create_mock_run()

        # Trace
        trace_id = composite.trace_agent_run(
            run=run,
            metadata={"composite_test": True},
            tags=["composite"],
        )

        assert trace_id is not None, "Should return trace ID"
        print(f"  [OK] Trace created: {trace_id}")

        # Verify JSON file was created
        json_files = list(Path(temp_dir).glob("trace_*.json"))
        assert len(json_files) == 1, f"Should have 1 JSON file, got {len(json_files)}"
        print("  [OK] JSON trace file created")

        composite.flush()
        composite.shutdown()

        print("\n  [PASS] Composite Observer Test")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Composite Observer Test: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_get_observer_factory():
    """Test the get_observer factory function."""
    print("\n" + "="*60)
    print("  GET_OBSERVER FACTORY TEST")
    print("="*60)

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="energbench_factory_test_")

    try:
        # Test JSON backend
        json_obs = get_observer("json", output_dir=temp_dir)
        assert isinstance(json_obs, JSONFileObserver), "Should return JSONFileObserver"
        print("  [OK] get_observer('json') returns JSONFileObserver")

        # Test auto backend (should return something)
        auto_obs = get_observer("auto", output_dir=temp_dir)
        assert auto_obs.is_enabled, "Auto observer should be enabled"
        print(f"  [OK] get_observer('auto') returns {type(auto_obs).__name__}")

        # Test both backend
        both_obs = get_observer("both", output_dir=temp_dir)
        assert isinstance(both_obs, CompositeObserver), "Should return CompositeObserver"
        print(f"  [OK] get_observer('both') returns CompositeObserver with {len(both_obs.observers)} backends")

        # Test invalid backend
        try:
            get_observer("invalid")
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "invalid" in str(e).lower()
            print("  [OK] get_observer('invalid') raises ValueError")

        print("\n  [PASS] Factory Test")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Factory Test: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all observer tests."""
    print("\n" + "="*60)
    print("  OBSERVABILITY MODULE TESTS")
    print("="*60)

    results = []

    # Run tests
    results.append(("JSON Observer", test_json_observer()))
    results.append(("Langfuse Observer", test_langfuse_observer()))
    results.append(("Composite Observer", test_composite_observer()))
    results.append(("Factory Function", test_get_observer_factory()))

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
