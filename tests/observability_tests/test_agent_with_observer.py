"""Test observability with real agent runs.

This test runs a real agent query and verifies all steps are captured
in the observability trace, including tool calls and any failures.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import pytest

from energbench.agent.providers import get_provider
from energbench.agent.react_agent import ReActAgent
from energbench.tools import create_default_registry
from energbench.observability import get_observer, JSONFileObserver


@pytest.mark.asyncio
async def test_agent_with_json_observer():
    """Test agent run with JSON observer captures all data."""
    print("\n" + "="*60)
    print("  AGENT WITH JSON OBSERVER TEST")
    print("="*60)

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="energbench_agent_obs_test_")

    try:
        # Initialize observer
        observer = JSONFileObserver(
            output_dir=temp_dir,
            single_file=False,
            pretty_print=True,
        )
        print(f"  [OK] Observer initialized at {temp_dir}")

        # Initialize provider and tools
        provider = get_provider("openai", model="gpt-4o-mini")
        registry = create_default_registry()
        tools = registry.get_all_tools()

        # Build tool executor
        async def tool_executor(tool_name: str, arguments: dict) -> str:
            result = await registry.execute(tool_name, **arguments)
            return result.to_json()

        # Create agent
        agent = ReActAgent(
            provider=provider,
            tools=tools,
            tool_executor=tool_executor,
            max_iterations=5,
        )
        print(f"  [OK] Agent initialized with {len(tools)} tools")

        # Run a query that uses tools
        query = "What is 15 multiplied by 7? Use the calculator tool."

        print(f"\n  Running query: {query}")
        run = await agent.run(query)

        print(f"\n  Agent result:")
        print(f"    Success: {run.success}")
        print(f"    Iterations: {run.iterations}")
        print(f"    Tool calls: {run.tool_calls_count}")
        print(f"    Steps: {len(run.steps)}")
        if run.final_answer:
            print(f"    Answer: {run.final_answer[:100]}...")

        # Trace the run
        trace_id = observer.trace_agent_run(
            run=run,
            metadata={"test": "agent_integration"},
            tags=["test", "agent", "calculator"],
            user_id="integration_test",
        )

        assert trace_id is not None, "Should create trace"
        print(f"\n  [OK] Trace created: {trace_id}")

        # Load and verify trace
        trace_data = observer.load_trace(trace_id)
        assert trace_data is not None, "Should load trace"

        # Verify all steps captured
        num_steps = len(trace_data["steps"])
        assert num_steps == len(run.steps), f"Should have all {len(run.steps)} steps, got {num_steps}"
        print(f"  [OK] All {num_steps} steps captured")

        # Verify metrics
        metrics = trace_data["metrics"]
        assert metrics["tool_calls_count"] == run.tool_calls_count, "Tool call count should match"
        assert metrics["iterations"] == run.iterations, "Iterations should match"
        print(f"  [OK] Metrics match: {metrics['tool_calls_count']} tool calls, {metrics['iterations']} iterations")

        # Verify query and answer captured
        assert trace_data["query"] == query, "Query should be captured"
        assert trace_data["final_answer"] is not None, "Final answer should be captured"
        print(f"  [OK] Query and answer captured")

        # Print trace file location
        trace_file = observer.get_trace_file(trace_id)
        print(f"\n  Trace file: {trace_file}")

        # Show step summary
        print(f"\n  Step Summary:")
        for step_type, count in trace_data["step_summary"]["step_types"].items():
            print(f"    {step_type}: {count}")

        print("\n  [PASS] Agent with JSON Observer Test")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Agent with JSON Observer Test: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_agent_with_both_backends():
    """Test agent run with both Langfuse and JSON backends."""
    print("\n" + "="*60)
    print("  AGENT WITH BOTH BACKENDS TEST")
    print("="*60)

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="energbench_both_test_")

    try:
        # Get composite observer
        observer = get_observer("both", output_dir=temp_dir)
        print(f"  [OK] Observer created: {type(observer).__name__}")
        print(f"      Backends: {[type(o).__name__ for o in observer.observers]}")

        # Initialize provider and tools
        provider = get_provider("openai", model="gpt-4o-mini")
        registry = create_default_registry()
        tools = registry.get_all_tools()

        async def tool_executor(tool_name: str, arguments: dict) -> str:
            result = await registry.execute(tool_name, **arguments)
            return result.to_json()

        agent = ReActAgent(
            provider=provider,
            tools=tools,
            tool_executor=tool_executor,
            max_iterations=5,
        )
        print(f"  [OK] Agent initialized with {len(tools)} tools")

        # Run query
        query = "Calculate the square root of 144 using the calculator."
        print(f"\n  Running query: {query}")

        run = await agent.run(query)
        answer = run.final_answer or ""
        print(f"  Result: {answer[:100]}..." if len(answer) > 100 else f"  Result: {answer}")

        # Trace
        trace_id = observer.trace_agent_run(
            run=run,
            metadata={"test": "both_backends"},
            tags=["test", "composite"],
        )

        assert trace_id is not None, "Should create trace"
        print(f"\n  [OK] Trace created: {trace_id}")

        # Verify JSON file created
        json_files = list(Path(temp_dir).glob("trace_*.json"))
        assert len(json_files) >= 1, "Should have JSON file"
        print(f"  [OK] JSON trace file created: {json_files[0].name}")

        # Flush and cleanup
        observer.flush()
        observer.shutdown()

        print("\n  [PASS] Agent with Both Backends Test")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Agent with Both Backends Test: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def main():
    """Run integration tests."""
    print("\n" + "="*60)
    print("  AGENT + OBSERVABILITY INTEGRATION TESTS")
    print("="*60)

    results = []

    results.append(("Agent with JSON Observer", await test_agent_with_json_observer()))
    results.append(("Agent with Both Backends", await test_agent_with_both_backends()))

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
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
