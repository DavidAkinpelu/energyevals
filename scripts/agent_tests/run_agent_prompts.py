#!/usr/bin/env python
"""Run ReActAgent against prompts using MCP + standard tools.

Requires RAG_SERVER_URL and DATABASE_SERVER_URL to point to remote MCP servers.

Usage:
    python run_agent_prompts.py                    # No observability
    python run_agent_prompts.py --observe json     # JSON file tracing
    python run_agent_prompts.py --observe langfuse # Langfuse tracing
    python run_agent_prompts.py --observe both     # Both backends
"""
from __future__ import annotations

import argparse
import asyncio
import os
import json
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv

from energbench.agent.react_agent import ReActAgent
from energbench.agent.providers.openai_provider import OpenAIProvider
from energbench.mcp import create_mcp_client
from energbench.tools import create_default_registry
from energbench.observability import get_observer, BaseObserver


PROMPTS = [
    "Compare last week's ERCOT regulation up price distributions with the same period last year.",
    "Which of the zones in NYISO had the highest price volatility last week? ",
    "What tariffs are available to commercial customers in Arlington Virginia? ",
    "Show me the typical annual solar generation profile for Midlands TX vs Bakersfield CA, ",
    "and compare their capacity factors. Find recent FERC filings related to battery storage.",
]


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value


def _build_executor(
    std_registry,
    mcp_client,
) -> Callable[[str, Dict[str, Any]], str]:
    std_tools = {tool.name for tool in std_registry.get_all_tools()}
    mcp_tools = {tool.name for tool in mcp_client.list_tools()}

    async def executor(tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name in std_tools:
            result = await std_registry.execute(tool_name, **arguments)
            return result.to_json()
        if tool_name in mcp_tools:
            return await mcp_client.call_tool(tool_name, arguments)
        return "{\"error\": \"Unknown tool\", \"tool\": \"%s\"}" % tool_name

    return executor


def _count_tool_failures(run) -> tuple[int, list[dict[str, str]]]:
    failures = 0
    details: list[dict[str, str]] = []
    for step in run.steps:
        if step.tool_output is None:
            continue
        output = step.tool_output
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            continue

        error = None
        if isinstance(payload, dict):
            if payload.get("success") is False:
                error = payload.get("error") or "success=false"
            elif "error" in payload:
                error = payload.get("error")

        if error:
            failures += 1
            details.append({"tool": step.tool_name or "unknown", "error": str(error)})

    return failures, details


def _print_metrics(run) -> None:
    failures, details = _count_tool_failures(run)
    tool_calls = run.tool_calls_count or len([s for s in run.steps if s.tool_name])

    print("\n=== METRICS ===")
    print(f"input_tokens: {run.total_input_tokens}")
    print(f"cached_tokens: {run.total_cached_tokens}")
    print(f"output_tokens: {run.total_output_tokens}")
    print(f"total_tokens: {run.total_tokens}")
    print(f"tool_calls: {tool_calls}")
    print(f"tool_failures: {failures}")
    if details:
        for item in details:
            print(f"  - {item['tool']}: {item['error']}")
    print(f"iterations: {run.iterations}")
    print(f"success: {run.success}")
    if run.error:
        print(f"error: {run.error}")
    print(f"duration_seconds: {run.duration_seconds:.2f}")


def _setup_observer(observe_backend: Optional[str]) -> Optional[BaseObserver]:
    """Set up observability if requested."""
    if not observe_backend:
        return None

    try:
        observer = get_observer(
            backend=observe_backend,
            output_dir="./observability_logs",
        )
        print(f"Observability enabled: {type(observer).__name__}")
        return observer
    except ValueError as e:
        print(f"Warning: Could not enable observability: {e}")
        return None


async def main(observe_backend: Optional[str] = None) -> None:
    load_dotenv()
    _require_env("RAG_SERVER_URL")
    _require_env("DATABASE_SERVER_URL")

    # Set up observability
    observer = _setup_observer(observe_backend)

    provider = OpenAIProvider(model=os.getenv("OPENAI_MODEL", "gpt-4o"))

    std_registry = create_default_registry()
    mcp_client = await create_mcp_client()

    tools = std_registry.get_all_tools() + mcp_client.list_tools()
    executor = _build_executor(std_registry, mcp_client)

    try:
        for i, prompt in enumerate(PROMPTS):
            print("\n=== PROMPT ===")
            print(prompt)

            # Create a new agent for each prompt
            agent = ReActAgent(provider=provider, tools=tools, tool_executor=executor)

            run = await agent.run(prompt)
            print("\n=== RESPONSE ===")
            if run.final_answer is not None:
                print(run.final_answer)
            else:
                print(f"Run failed: {run.error}")
            _print_metrics(run)

            # Trace the run if observability is enabled
            if observer:
                trace_id = observer.trace_agent_run(
                    run=run,
                    metadata={"prompt_index": i, "model": provider.model},
                    tags=["benchmark", "mcp"],
                )
                if trace_id:
                    print(f"Trace ID: {trace_id}")
    finally:
        await mcp_client.disconnect()
        if observer:
            observer.flush()
            observer.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ReActAgent against prompts with optional observability"
    )
    parser.add_argument(
        "--observe",
        choices=["json", "langfuse", "both", "auto"],
        default=None,
        help="Enable observability tracing (json, langfuse, both, or auto)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(observe_backend=args.observe))
