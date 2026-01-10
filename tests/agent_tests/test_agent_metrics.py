#!/usr/bin/env python3
"""Test suite to verify agent metrics collection with different providers.

Tests agent execution with real MCP tools and standard tools,
verifying that all metrics (tokens, latency, steps, tool calls) are collected.

Usage:
    python -m tests.agent_tests.test_agent_metrics
    python -m tests.agent_tests.test_agent_metrics --provider openai
    python -m tests.agent_tests.test_agent_metrics --provider anthropic --query "custom query"
"""

import asyncio
import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from energbench.agent.providers import get_provider
from energbench.agent.react_agent import ReActAgent
from energbench.agent.schema import AgentRun, StepType
from energbench.mcp.client import MCPClient, MCPServerConfig
from energbench.tools import create_default_registry


# Realistic energy market prompts for testing
ENERGY_PROMPTS = {
    "database_query": "Show me the available databases in the energy data warehouse.",
    "rag_search": "Search for documents about ERCOT ancillary services market.",
    "market_analysis": "What tables are available in the ercot_db database?",
}


@dataclass
class MetricsTestResult:
    """Result of an agent metrics test."""

    provider: str
    model: str
    test_name: str
    passed: bool
    message: str
    metrics: Optional[dict[str, Any]] = None


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def print_subheader(text: str) -> None:
    """Print a subheader."""
    print(f"\n  --- {text} ---")


def print_result(result: MetricsTestResult) -> None:
    """Print a test result."""
    status = "[PASS]" if result.passed else "[FAIL]"
    print(f"\n  {status} {result.test_name}")
    if result.metrics:
        print(f"       Metrics:")
        for key, value in result.metrics.items():
            print(f"         {key}: {value}")
    if not result.passed:
        print(f"       Error: {result.message}")


def validate_metrics(run: AgentRun) -> tuple[bool, list[str]]:
    """Validate that all metrics are properly collected."""
    issues = []

    if run.total_input_tokens <= 0:
        issues.append(f"total_input_tokens should be > 0, got {run.total_input_tokens}")

    if run.total_output_tokens <= 0:
        issues.append(f"total_output_tokens should be > 0, got {run.total_output_tokens}")

    if run.total_latency_ms <= 0:
        issues.append(f"total_latency_ms should be > 0, got {run.total_latency_ms}")

    if run.iterations <= 0:
        issues.append(f"iterations should be > 0, got {run.iterations}")

    expected_total = run.total_input_tokens + run.total_output_tokens
    if run.total_tokens != expected_total:
        issues.append(f"total_tokens mismatch: {run.total_tokens} != {expected_total}")

    if len(run.steps) == 0:
        issues.append("No steps recorded")

    for i, step in enumerate(run.steps):
        if step.latency_ms < 0:
            issues.append(f"Step {i} has negative latency: {step.latency_ms}")

        if step.step_type in [StepType.ACTION, StepType.OBSERVATION]:
            if not step.tool_name:
                issues.append(f"Step {i} ({step.step_type.value}) missing tool_name")

    if run.success and not run.final_answer:
        issues.append("Run succeeded but no final_answer set")

    return len(issues) == 0, issues


async def setup_mcp_client() -> MCPClient:
    """Set up MCP client with remote servers."""
    rag_url = os.getenv("RAG_SERVER_URL")
    db_url = os.getenv("DATABASE_SERVER_URL")

    if not rag_url or not db_url:
        raise ValueError(
            "Missing MCP server URLs. Set RAG_SERVER_URL and DATABASE_SERVER_URL."
        )

    servers = [
        MCPServerConfig(name="energy-rag", url=rag_url, description="RAG server"),
        MCPServerConfig(name="energy-database", url=db_url, description="Database server"),
    ]

    client = MCPClient(servers)
    await client.connect()
    return client


def build_tool_executor(
    std_registry,
    mcp_client: MCPClient,
) -> Callable[[str, Dict[str, Any]], str]:
    """Build a tool executor that combines standard tools and MCP tools."""
    std_tools = {tool.name for tool in std_registry.get_all_tools()}
    mcp_tools = {tool.name for tool in mcp_client.list_tools()}

    async def executor(tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name in std_tools:
            result = await std_registry.execute(tool_name, **arguments)
            return result.to_json()
        if tool_name in mcp_tools:
            return await mcp_client.call_tool(tool_name, arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    return executor


async def test_agent_with_mcp_tools(
    provider_name: str,
    prompt: str,
    prompt_name: str,
    model: Optional[str] = None,
) -> MetricsTestResult:
    """Test agent with real MCP tools from remote servers."""
    test_name = f"MCP Tools - {prompt_name}"
    mcp_client = None

    try:
        # Initialize provider
        if model:
            provider = get_provider(provider_name, model=model)
        else:
            provider = get_provider(provider_name)

        actual_model = provider.model

        # Set up MCP client
        mcp_client = await setup_mcp_client()
        mcp_tools = mcp_client.list_tools()

        print(f"       MCP servers connected, {len(mcp_tools)} tools available")

        # Create agent with MCP tools only
        agent = ReActAgent(
            provider=provider,
            tools=mcp_tools,
            tool_executor=mcp_client.get_executor(),
            max_iterations=5,
        )

        # Run agent
        run = await agent.run(prompt)

        # Validate metrics
        valid, issues = validate_metrics(run)

        metrics = {
            "input_tokens": run.total_input_tokens,
            "output_tokens": run.total_output_tokens,
            "cached_tokens": run.total_cached_tokens,
            "total_tokens": run.total_tokens,
            "latency_ms": f"{run.total_latency_ms:.0f}",
            "iterations": run.iterations,
            "tool_calls": run.tool_calls_count,
            "steps": len(run.steps),
            "success": run.success,
        }

        if not valid:
            return MetricsTestResult(
                provider=provider_name,
                model=actual_model,
                test_name=test_name,
                passed=False,
                message="; ".join(issues),
                metrics=metrics,
            )

        if not run.success:
            return MetricsTestResult(
                provider=provider_name,
                model=actual_model,
                test_name=test_name,
                passed=False,
                message=f"Agent run failed: {run.error}",
                metrics=metrics,
            )

        if run.tool_calls_count == 0:
            return MetricsTestResult(
                provider=provider_name,
                model=actual_model,
                test_name=test_name,
                passed=False,
                message="No tools were called",
                metrics=metrics,
            )

        return MetricsTestResult(
            provider=provider_name,
            model=actual_model,
            test_name=test_name,
            passed=True,
            message="Metrics collected correctly",
            metrics=metrics,
        )

    except Exception as e:
        return MetricsTestResult(
            provider=provider_name,
            model=model or "default",
            test_name=test_name,
            passed=False,
            message=str(e),
        )
    finally:
        if mcp_client:
            await mcp_client.disconnect()


async def test_step_level_metrics(
    provider_name: str,
    model: Optional[str] = None,
) -> MetricsTestResult:
    """Test that individual step metrics are tracked correctly."""
    test_name = "Step-Level Metrics"
    mcp_client = None

    try:
        if model:
            provider = get_provider(provider_name, model=model)
        else:
            provider = get_provider(provider_name)

        actual_model = provider.model

        # Set up MCP client
        mcp_client = await setup_mcp_client()
        mcp_tools = mcp_client.list_tools()

        agent = ReActAgent(
            provider=provider,
            tools=mcp_tools,
            tool_executor=mcp_client.get_executor(),
            max_iterations=5,
        )

        # Run agent with a simple prompt
        run = await agent.run("Show me the available databases.")

        # Analyze step metrics
        step_types = [step.step_type.value for step in run.steps]
        action_steps = [s for s in run.steps if s.step_type == StepType.ACTION]
        observation_steps = [s for s in run.steps if s.step_type == StepType.OBSERVATION]
        answer_steps = [s for s in run.steps if s.step_type == StepType.ANSWER]

        issues = []

        if len(action_steps) != len(observation_steps):
            issues.append(
                f"Action/Observation mismatch: {len(action_steps)} actions, "
                f"{len(observation_steps)} observations"
            )

        for i, step in enumerate(action_steps):
            if not step.tool_name:
                issues.append(f"Action step {i} missing tool_name")
            if step.tool_input is None:
                issues.append(f"Action step {i} missing tool_input")

        for i, step in enumerate(observation_steps):
            if not step.tool_output:
                issues.append(f"Observation step {i} missing tool_output")

        if run.success and len(answer_steps) == 0:
            issues.append("Successful run has no answer step")

        metrics = {
            "total_steps": len(run.steps),
            "step_types": step_types,
            "action_steps": len(action_steps),
            "observation_steps": len(observation_steps),
            "answer_steps": len(answer_steps),
        }

        if issues:
            return MetricsTestResult(
                provider=provider_name,
                model=actual_model,
                test_name=test_name,
                passed=False,
                message="; ".join(issues),
                metrics=metrics,
            )

        return MetricsTestResult(
            provider=provider_name,
            model=actual_model,
            test_name=test_name,
            passed=True,
            message="Step metrics tracked correctly",
            metrics=metrics,
        )

    except Exception as e:
        return MetricsTestResult(
            provider=provider_name,
            model=model or "default",
            test_name=test_name,
            passed=False,
            message=str(e),
        )
    finally:
        if mcp_client:
            await mcp_client.disconnect()


async def test_combined_tools(
    provider_name: str,
    model: Optional[str] = None,
) -> MetricsTestResult:
    """Test agent with both MCP tools and standard tools."""
    test_name = "Combined Tools (MCP + Standard)"
    mcp_client = None

    try:
        if model:
            provider = get_provider(provider_name, model=model)
        else:
            provider = get_provider(provider_name)

        actual_model = provider.model

        # Set up MCP client and standard tools
        mcp_client = await setup_mcp_client()
        std_registry = create_default_registry()

        # Combine tools
        all_tools = std_registry.get_all_tools() + mcp_client.list_tools()
        executor = build_tool_executor(std_registry, mcp_client)

        print(f"       Combined tools: {len(std_registry.get_all_tools())} standard + {len(mcp_client.list_tools())} MCP")

        agent = ReActAgent(
            provider=provider,
            tools=all_tools,
            tool_executor=executor,
            max_iterations=5,
        )

        # Run agent with a prompt that uses database tools
        run = await agent.run("What databases are available for energy market data?")

        valid, issues = validate_metrics(run)

        metrics = {
            "input_tokens": run.total_input_tokens,
            "output_tokens": run.total_output_tokens,
            "total_tokens": run.total_tokens,
            "latency_ms": f"{run.total_latency_ms:.0f}",
            "iterations": run.iterations,
            "tool_calls": run.tool_calls_count,
            "steps": len(run.steps),
            "success": run.success,
        }

        if not valid:
            return MetricsTestResult(
                provider=provider_name,
                model=actual_model,
                test_name=test_name,
                passed=False,
                message="; ".join(issues),
                metrics=metrics,
            )

        if not run.success:
            return MetricsTestResult(
                provider=provider_name,
                model=actual_model,
                test_name=test_name,
                passed=False,
                message=f"Agent run failed: {run.error}",
                metrics=metrics,
            )

        return MetricsTestResult(
            provider=provider_name,
            model=actual_model,
            test_name=test_name,
            passed=True,
            message="Combined tools work correctly",
            metrics=metrics,
        )

    except Exception as e:
        return MetricsTestResult(
            provider=provider_name,
            model=model or "default",
            test_name=test_name,
            passed=False,
            message=str(e),
        )
    finally:
        if mcp_client:
            await mcp_client.disconnect()


async def test_provider(
    provider_name: str,
    model: Optional[str] = None,
    custom_query: Optional[str] = None,
) -> list[MetricsTestResult]:
    """Run all tests for a specific provider."""
    results = []

    try:
        if model:
            provider = get_provider(provider_name, model=model)
        else:
            provider = get_provider(provider_name)
        actual_model = provider.model
        print_header(f"{provider_name.upper()} Provider ({actual_model})")
    except Exception as e:
        print_header(f"{provider_name.upper()} Provider")
        result = MetricsTestResult(
            provider=provider_name,
            model=model or "default",
            test_name="Provider Initialization",
            passed=False,
            message=str(e),
        )
        print_result(result)
        return [result]

    # Test 1: MCP Tools - Database Query
    print_subheader("Test 1: MCP Tools - Database Query")
    prompt = custom_query or ENERGY_PROMPTS["database_query"]
    result = await test_agent_with_mcp_tools(provider_name, prompt, "database_query", model)
    results.append(result)
    print_result(result)

    # Test 2: Step-Level Metrics
    print_subheader("Test 2: Step-Level Metrics")
    result = await test_step_level_metrics(provider_name, model)
    results.append(result)
    print_result(result)

    # Test 3: Combined Tools
    print_subheader("Test 3: Combined Tools")
    result = await test_combined_tools(provider_name, model)
    results.append(result)
    print_result(result)

    return results


async def run_all_tests(
    providers: Optional[list[str]] = None,
    custom_query: Optional[str] = None,
) -> dict[str, list[MetricsTestResult]]:
    """Run tests for all providers."""
    provider_list = providers or ["openai", "anthropic", "deepinfra"]

    all_results = {}

    print("\n" + "=" * 70)
    print("  ENERGBENCH AGENT METRICS TEST SUITE")
    print("=" * 70)

    # Check API keys
    print("\n  Checking API keys...")
    api_keys = {
        "openai": ("OPENAI_API_KEY", bool(os.getenv("OPENAI_API_KEY"))),
        "anthropic": ("ANTHROPIC_API_KEY", bool(os.getenv("ANTHROPIC_API_KEY"))),
        "deepinfra": ("DEEPINFRA_API_KEY", bool(os.getenv("DEEPINFRA_API_KEY"))),
    }

    for provider_name, (env_var, exists) in api_keys.items():
        if provider_name in provider_list:
            status = "[OK]" if exists else "[MISSING]"
            print(f"    {env_var}: {status}")

    # Check MCP URLs
    print("\n  Checking MCP server URLs...")
    rag_url = os.getenv("RAG_SERVER_URL")
    db_url = os.getenv("DATABASE_SERVER_URL")
    print(f"    RAG_SERVER_URL: {'[OK]' if rag_url else '[MISSING]'}")
    print(f"    DATABASE_SERVER_URL: {'[OK]' if db_url else '[MISSING]'}")

    if not rag_url or not db_url:
        print("\n  ERROR: MCP URLs not configured. Cannot run tests.")
        return {}

    # Run tests for each provider
    for provider_name in provider_list:
        env_var, exists = api_keys.get(provider_name, (None, False))
        if not exists:
            print_header(f"{provider_name.upper()} Provider")
            result = MetricsTestResult(
                provider=provider_name,
                model="default",
                test_name="API Key Check",
                passed=False,
                message=f"Missing {env_var}",
            )
            print_result(result)
            all_results[provider_name] = [result]
            continue

        results = await test_provider(provider_name, custom_query=custom_query)
        all_results[provider_name] = results

    return all_results


def print_summary(all_results: dict[str, list[MetricsTestResult]]) -> bool:
    """Print test summary and return overall success."""
    print_header("TEST SUMMARY")

    total_tests = 0
    total_passed = 0

    for provider_name, results in all_results.items():
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        total_tests += total
        total_passed += passed

        status = "[OK]" if passed == total else "[FAIL]"
        print(f"  {status} {provider_name}: {passed}/{total} tests passed")

        for result in results:
            if not result.passed:
                print(f"       [FAIL] {result.test_name}: {result.message[:60]}...")

    print(f"\n  Total: {total_passed}/{total_tests} tests passed")

    overall_success = total_passed == total_tests
    if overall_success:
        print("\n  All agent metrics tests passed!")
    else:
        print("\n  Some tests failed.")

    return overall_success


async def main():
    parser = argparse.ArgumentParser(
        description="Test agent metrics collection with different providers"
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        choices=["openai", "anthropic", "deepinfra"],
        help="Test only a specific provider",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Specify a model to test (only with --provider)",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Custom query to test",
    )

    args = parser.parse_args()

    providers = [args.provider] if args.provider else None

    all_results = await run_all_tests(
        providers=providers,
        custom_query=args.query,
    )

    success = print_summary(all_results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
