#!/usr/bin/env python3
"""Test suite to verify all model provider implementations.

Tests basic completion, tool calling, and streaming for each provider.

Usage:
    python -m tests.provider_tests.test_providers
    python -m tests.provider_tests.test_providers --provider openai
    python -m tests.provider_tests.test_providers --skip-streaming
"""

import asyncio
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from energbench.agent.providers import (
    get_provider,
    Message,
    ToolDefinition,
)


@dataclass
class TestResult:
    """Result of a provider test."""

    provider: str
    model: str
    test_name: str
    passed: bool
    message: str
    latency_ms: float = 0.0
    tokens: int = 0


# Energy-related tools for testing function calling
ENERGY_TOOLS = [
    ToolDefinition(
        name="get_electricity_price",
        description="Get real-time electricity prices for an ISO/RTO market",
        parameters={
            "type": "object",
            "properties": {
                "market": {
                    "type": "string",
                    "enum": ["ERCOT", "PJM", "CAISO", "NYISO", "MISO"],
                    "description": "The electricity market/ISO",
                },
                "zone": {
                    "type": "string",
                    "description": "The pricing zone or node",
                },
            },
            "required": ["market"],
        },
    ),
    ToolDefinition(
        name="search_energy_documents",
        description="Search energy market documents and reports",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for energy documents",
                },
                "market": {
                    "type": "string",
                    "description": "Filter by market (ERCOT, PJM, etc.)",
                },
            },
            "required": ["query"],
        },
    ),
]


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_result(result: TestResult) -> None:
    """Print a test result."""
    status = "PASS" if result.passed else "FAIL"
    print(f"  [{status}] {result.test_name}")
    if result.latency_ms > 0:
        print(f"         Latency: {result.latency_ms:.0f}ms, Tokens: {result.tokens}")
    if not result.passed:
        print(f"         Error: {result.message}")


async def test_basic_completion(provider, provider_name: str, model: str) -> TestResult:
    """Test basic text completion with an energy-related question."""
    test_name = "Basic Completion"

    try:
        messages = [
            Message(
                role="system",
                content="You are an energy market analyst. Be concise and factual.",
            ),
            Message(
                role="user",
                content="What does LMP stand for in electricity markets? Reply in one sentence.",
            ),
        ]

        response = await provider.complete(messages, max_tokens=100)

        content = response.content.strip().lower()
        # Check for key terms that should appear in the response
        has_relevant_content = any(
            term in content
            for term in ["locational", "marginal", "price", "lmp"]
        )

        if not has_relevant_content:
            return TestResult(
                provider=provider_name,
                model=model,
                test_name=test_name,
                passed=False,
                message=f"Response doesn't mention LMP concepts: {response.content[:100]}",
                latency_ms=response.latency_ms,
                tokens=response.input_tokens + response.output_tokens,
            )

        return TestResult(
            provider=provider_name,
            model=model,
            test_name=test_name,
            passed=True,
            message=f"Response: {response.content[:80]}...",
            latency_ms=response.latency_ms,
            tokens=response.input_tokens + response.output_tokens,
        )

    except Exception as e:
        return TestResult(
            provider=provider_name,
            model=model,
            test_name=test_name,
            passed=False,
            message=str(e),
        )


async def test_tool_calling(provider, provider_name: str, model: str) -> TestResult:
    """Test function/tool calling with energy market tools."""
    test_name = "Tool Calling"

    try:
        messages = [
            Message(
                role="system",
                content="You are an energy market analyst. Use available tools to get data.",
            ),
            Message(
                role="user",
                content="What are the current electricity prices in ERCOT?",
            ),
        ]

        response = await provider.complete(messages, tools=ENERGY_TOOLS, max_tokens=200)

        if not response.tool_calls or len(response.tool_calls) == 0:
            return TestResult(
                provider=provider_name,
                model=model,
                test_name=test_name,
                passed=False,
                message=f"No tool calls made. Response: {response.content[:100] if response.content else 'empty'}",
                latency_ms=response.latency_ms,
                tokens=response.input_tokens + response.output_tokens,
            )

        tool_call = response.tool_calls[0]
        if tool_call.name != "get_electricity_price":
            return TestResult(
                provider=provider_name,
                model=model,
                test_name=test_name,
                passed=False,
                message=f"Expected 'get_electricity_price', got: {tool_call.name}",
                latency_ms=response.latency_ms,
                tokens=response.input_tokens + response.output_tokens,
            )

        if "market" not in tool_call.arguments:
            return TestResult(
                provider=provider_name,
                model=model,
                test_name=test_name,
                passed=False,
                message=f"Missing 'market' argument: {tool_call.arguments}",
                latency_ms=response.latency_ms,
                tokens=response.input_tokens + response.output_tokens,
            )

        return TestResult(
            provider=provider_name,
            model=model,
            test_name=test_name,
            passed=True,
            message=f"Tool: {tool_call.name}, Args: {tool_call.arguments}",
            latency_ms=response.latency_ms,
            tokens=response.input_tokens + response.output_tokens,
        )

    except Exception as e:
        return TestResult(
            provider=provider_name,
            model=model,
            test_name=test_name,
            passed=False,
            message=str(e),
        )


async def test_streaming(provider, provider_name: str, model: str) -> TestResult:
    """Test streaming text generation."""
    test_name = "Streaming"

    try:
        messages = [
            Message(
                role="user",
                content="List 3 major US electricity markets (ISOs). Just the names, one per line.",
            ),
        ]

        chunks = []
        async for chunk in provider.stream(messages, max_tokens=100):
            chunks.append(chunk)

        full_response = "".join(chunks)

        if not full_response.strip():
            return TestResult(
                provider=provider_name,
                model=model,
                test_name=test_name,
                passed=False,
                message="Empty streaming response",
            )

        # Check for expected ISO names (abbreviations and common variations)
        iso_patterns = [
            "ercot", "pjm", "caiso", "nyiso", "miso", "spp", "iso-ne", "isone",
            "california iso", "new york iso", "texas", "midcontinent",
            "new england", "southwest power", "electric reliability"
        ]
        found_isos = sum(1 for pattern in iso_patterns if pattern in full_response.lower())

        return TestResult(
            provider=provider_name,
            model=model,
            test_name=test_name,
            passed=found_isos >= 2,
            message=f"Chunks: {len(chunks)}, Found {found_isos} ISOs",
        )

    except Exception as e:
        return TestResult(
            provider=provider_name,
            model=model,
            test_name=test_name,
            passed=False,
            message=str(e),
        )


async def test_provider(
    provider_name: str,
    model: Optional[str] = None,
    skip_streaming: bool = False,
) -> list[TestResult]:
    """Test a specific provider."""
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
        result = TestResult(
            provider=provider_name,
            model=model or "default",
            test_name="Provider Initialization",
            passed=False,
            message=str(e),
        )
        print_result(result)
        return [result]

    print("\n  Running tests...")

    # Test 1: Basic completion
    result = await test_basic_completion(provider, provider_name, actual_model)
    results.append(result)
    print_result(result)

    # Test 2: Tool calling
    result = await test_tool_calling(provider, provider_name, actual_model)
    results.append(result)
    print_result(result)

    # Test 3: Streaming (optional)
    if not skip_streaming:
        result = await test_streaming(provider, provider_name, actual_model)
        results.append(result)
        print_result(result)

    return results


async def run_all_tests(
    providers: Optional[list[str]] = None,
    skip_streaming: bool = False,
) -> dict[str, list[TestResult]]:
    """Run tests for all providers."""

    provider_list = providers or ["openai", "anthropic", "deepinfra"]

    all_results = {}

    print("\n" + "=" * 60)
    print("  ENERGBENCH PROVIDER TEST SUITE")
    print("=" * 60)

    # Check for API keys
    print("\n  Checking API keys...")
    api_keys = {
        "openai": ("OPENAI_API_KEY", bool(os.getenv("OPENAI_API_KEY"))),
        "anthropic": ("ANTHROPIC_API_KEY", bool(os.getenv("ANTHROPIC_API_KEY"))),
        "deepinfra": ("DEEPINFRA_API_KEY", bool(os.getenv("DEEPINFRA_API_KEY"))),
        "google": ("GOOGLE_API_KEY", bool(os.getenv("GOOGLE_API_KEY"))),
    }

    for provider_name, (env_var, exists) in api_keys.items():
        if provider_name in provider_list:
            status = "[OK]" if exists else "[MISSING]"
            print(f"    {env_var}: {status}")

    # Run tests for each provider
    for provider_name in provider_list:
        env_var, exists = api_keys.get(provider_name, (None, False))
        if not exists:
            print_header(f"{provider_name.upper()} Provider")
            result = TestResult(
                provider=provider_name,
                model="default",
                test_name="API Key Check",
                passed=False,
                message=f"Missing {env_var}",
            )
            print_result(result)
            all_results[provider_name] = [result]
            continue

        results = await test_provider(provider_name, skip_streaming=skip_streaming)
        all_results[provider_name] = results

    return all_results


def print_summary(all_results: dict[str, list[TestResult]]) -> bool:
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

    print(f"\n  Total: {total_passed}/{total_tests} tests passed")

    overall_success = total_passed == total_tests
    if overall_success:
        print("\n  All provider tests passed!")
    else:
        print("\n  Some tests failed.")

    return overall_success


async def main():
    parser = argparse.ArgumentParser(description="Test model provider implementations")
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        choices=["openai", "anthropic", "deepinfra", "google"],
        help="Test only a specific provider",
    )
    parser.add_argument(
        "--skip-streaming",
        action="store_true",
        help="Skip streaming tests",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Specify a model to test (only with --provider)",
    )

    args = parser.parse_args()

    providers = [args.provider] if args.provider else None

    all_results = await run_all_tests(
        providers=providers,
        skip_streaming=args.skip_streaming,
    )

    success = print_summary(all_results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
