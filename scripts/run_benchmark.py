#!/usr/bin/env python
"""Run benchmark questions against different model providers.

This script loads configuration from a YAML file and runs benchmark questions
through the ReAct agent with the specified provider.

Usage:
    python scripts/run_benchmark.py                              # Use default config
    python scripts/run_benchmark.py --config my_config.yaml      # Use custom config
    python scripts/run_benchmark.py --list-questions             # List available questions
    python scripts/run_benchmark.py --provider openai            # Override provider
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from energbench.agent.providers import get_provider
from energbench.agent.react_agent import ReActAgent
from energbench.agent.schema import AgentRun
from energbench.mcp import create_mcp_client
from energbench.tools import create_default_registry
from energbench.observability import get_observer, BaseObserver


# Default configuration
DEFAULT_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "questions_file": "data/AI Evals New Questions.xlsx - Q&As.csv",
    "questions": None,  # None means all questions
    "observability": {
        "enabled": True,
        "backend": "json",
        "output_dir": "./benchmark_traces",
    },
    "mcp": {
        "enabled": True,
    },
    "agent": {
        "max_iterations": 25,
    },
    "output": {
        "results_dir": "./benchmark_results",
        "save_answers": True,
    },
}

# Available providers and their models
PROVIDERS = {
    "openai": {
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o", "gpt-4o-mini"],
    },
    "anthropic": {
        "default_model": "claude-sonnet-4-20250514",
        "models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
    },
    "google": {
        "default_model": "gemini-2.0-flash",
        "models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    },
    "deepinfra": {
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-405B-Instruct",
        ],
    },
}


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    provider: str
    model: str
    questions_file: Path
    questions: Optional[List[int]]
    observability_enabled: bool
    observability_backend: str
    observability_output_dir: Path
    mcp_enabled: bool
    max_iterations: int
    results_dir: Path
    save_answers: bool
    is_reasoning_model: Optional[bool] = None  # Override reasoning model auto-detection

    @classmethod
    def from_dict(cls, data: dict, base_path: Path) -> "BenchmarkConfig":
        """Create config from dictionary."""
        obs = data.get("observability", {})
        mcp = data.get("mcp", {})
        agent = data.get("agent", {})
        output = data.get("output", {})

        # Parse questions
        questions = data.get("questions")
        if questions:
            questions = cls._parse_questions(questions)

        # Resolve paths relative to base_path
        questions_file = base_path / data.get("questions_file", DEFAULT_CONFIG["questions_file"])

        return cls(
            provider=data.get("provider", DEFAULT_CONFIG["provider"]),
            model=data.get("model", PROVIDERS[data.get("provider", "openai")]["default_model"]),
            questions_file=questions_file,
            questions=questions,
            observability_enabled=obs.get("enabled", True),
            observability_backend=obs.get("backend", "json"),
            observability_output_dir=Path(obs.get("output_dir", "./benchmark_traces")),
            mcp_enabled=mcp.get("enabled", True),
            max_iterations=agent.get("max_iterations", 10),
            results_dir=Path(output.get("results_dir", "./benchmark_results")),
            save_answers=output.get("save_answers", True),
            is_reasoning_model=data.get("is_reasoning_model"),  # None = auto-detect
        )

    @staticmethod
    def _parse_questions(questions: Union[str, List[int], None]) -> Optional[List[int]]:
        """Parse question specification into list of IDs."""
        if questions is None:
            return None
        if isinstance(questions, list):
            return questions
        if isinstance(questions, str):
            result = []
            for part in str(questions).split(","):
                part = part.strip()
                if "-" in part:
                    start, end = part.split("-")
                    result.extend(range(int(start), int(end) + 1))
                else:
                    result.append(int(part))
            return result
        return None


@dataclass
class Question:
    """A benchmark question."""
    id: int
    category: str
    question_type: str
    difficulty: str
    question: str


@dataclass
class BenchmarkResult:
    """Result of running a benchmark question."""
    question: Question
    provider: str
    model: str
    success: bool
    answer: Optional[str]
    error: Optional[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None


def load_config(config_path: Optional[Path], base_path: Path) -> BenchmarkConfig:
    """Load configuration from YAML file or use defaults."""
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        print(f"Loaded config from: {config_path}")
        return BenchmarkConfig.from_dict(data, base_path)
    else:
        print("Using default configuration")
        return BenchmarkConfig.from_dict(DEFAULT_CONFIG, base_path)


def load_questions(csv_path: Path) -> List[Question]:
    """Load questions from a CSV file."""
    questions = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(Question(
                id=int(row.get("S/N", 0)),
                category=row.get("Category", ""),
                question_type=row.get("Question type", ""),
                difficulty=row.get("Difficulty level", ""),
                question=row.get("Question", ""),
            ))

    return questions


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def print_config(config: BenchmarkConfig) -> None:
    """Print configuration summary."""
    reasoning_status = (
        "yes (override)" if config.is_reasoning_model is True
        else "no (override)" if config.is_reasoning_model is False
        else "auto-detect"
    )
    print(f"\n  Configuration:")
    print(f"    Provider: {config.provider}")
    print(f"    Model: {config.model}")
    print(f"    Reasoning model: {reasoning_status}")
    print(f"    Questions file: {config.questions_file}")
    print(f"    Questions: {config.questions or 'all'}")
    print(f"    MCP enabled: {config.mcp_enabled}")
    print(f"    Observability: {config.observability_backend if config.observability_enabled else 'disabled'}")
    print(f"    Max iterations: {config.max_iterations}")


def print_question(q: Question, index: int, total: int) -> None:
    """Print question details."""
    print(f"\n  [{index}/{total}] Question {q.id} | {q.category} | {q.difficulty}")
    print(f"  {q.question[:200]}..." if len(q.question) > 200 else f"  {q.question}")


def print_result(result: BenchmarkResult) -> None:
    """Print benchmark result."""
    status = "[PASS]" if result.success else "[FAIL]"
    print(f"\n  {status}")

    if result.answer:
        answer_preview = result.answer[:300] + "..." if len(result.answer) > 300 else result.answer
        print(f"  Answer: {answer_preview}")

    if result.error:
        print(f"  Error: {result.error}")

    if result.metrics:
        print(f"  Metrics: tokens={result.metrics.get('total_tokens', 0)}, "
              f"tools={result.metrics.get('tool_calls', 0)}, "
              f"time={result.metrics.get('duration_seconds', 0):.1f}s")

    if result.trace_id:
        print(f"  Trace: {result.trace_id}")


def build_tool_executor(
    std_registry,
    mcp_client=None,
) -> Callable[[str, Dict[str, Any]], str]:
    """Build a tool executor that combines standard tools and MCP tools."""
    std_tools = {tool.name for tool in std_registry.get_all_tools()}
    mcp_tools = {tool.name for tool in mcp_client.list_tools()} if mcp_client else set()

    async def executor(tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name in std_tools:
            result = await std_registry.execute(tool_name, **arguments)
            return result.to_json()
        if mcp_client and tool_name in mcp_tools:
            return await mcp_client.call_tool(tool_name, arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    return executor


async def run_question(
    agent: ReActAgent,
    question: Question,
    config: BenchmarkConfig,
    observer: Optional[BaseObserver] = None,
) -> BenchmarkResult:
    """Run a single benchmark question."""
    try:
        run: AgentRun = await agent.run(question.question)

        metrics = {
            "input_tokens": run.total_input_tokens,
            "output_tokens": run.total_output_tokens,
            "cached_tokens": run.total_cached_tokens,
            "total_tokens": run.total_tokens,
            "tool_calls": run.tool_calls_count,
            "iterations": run.iterations,
            "duration_seconds": run.duration_seconds,
            "latency_ms": run.total_latency_ms,
        }

        trace_id = None
        if observer:
            trace_id = observer.trace_agent_run(
                run=run,
                metadata={
                    "question_id": question.id,
                    "category": question.category,
                    "difficulty": question.difficulty,
                    "provider": config.provider,
                    "model": config.model,
                },
                tags=["benchmark", question.category, question.difficulty],
            )

        return BenchmarkResult(
            question=question,
            provider=config.provider,
            model=config.model,
            success=run.success,
            answer=run.final_answer if config.save_answers else None,
            error=run.error,
            metrics=metrics,
            trace_id=trace_id,
        )

    except Exception as e:
        return BenchmarkResult(
            question=question,
            provider=config.provider,
            model=config.model,
            success=False,
            answer=None,
            error=str(e),
        )


def save_results(results: List[BenchmarkResult], config: BenchmarkConfig) -> Path:
    """Save benchmark results to JSON."""
    config.results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = config.results_dir / f"benchmark_{config.provider}_{timestamp}.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "provider": config.provider,
            "model": config.model,
            "questions_file": str(config.questions_file),
            "mcp_enabled": config.mcp_enabled,
            "max_iterations": config.max_iterations,
        },
        "summary": {
            "total_questions": len(results),
            "passed": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "total_tokens": sum(r.metrics.get("total_tokens", 0) for r in results),
            "total_duration_seconds": sum(r.metrics.get("duration_seconds", 0) for r in results),
        },
        "results": [
            {
                "question_id": r.question.id,
                "category": r.question.category,
                "difficulty": r.question.difficulty,
                "question": r.question.question,
                "success": r.success,
                "answer": r.answer,
                "error": r.error,
                "metrics": r.metrics,
                "trace_id": r.trace_id,
            }
            for r in results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path


async def run_benchmark(config: BenchmarkConfig) -> int:
    """Run the benchmark with the given configuration."""

    # Load questions
    if not config.questions_file.exists():
        print(f"Error: Questions file not found: {config.questions_file}")
        return 1

    questions = load_questions(config.questions_file)
    if not questions:
        print(f"Error: No questions found in {config.questions_file}")
        return 1

    # Filter questions if specified
    if config.questions:
        questions = [q for q in questions if q.id in config.questions]
        if not questions:
            print(f"Error: No matching questions for IDs: {config.questions}")
            return 1

    print(f"  Loaded {len(questions)} questions")

    # Validate provider
    if config.provider not in PROVIDERS:
        print(f"Error: Unknown provider '{config.provider}'")
        print(f"Available: {', '.join(PROVIDERS.keys())}")
        return 1

    # Set up observability
    observer = None
    if config.observability_enabled:
        try:
            config.observability_output_dir.mkdir(parents=True, exist_ok=True)
            observer = get_observer(
                backend=config.observability_backend,
                output_dir=str(config.observability_output_dir),
            )
            print(f"  Observability: {type(observer).__name__}")
        except ValueError as e:
            print(f"  Warning: Could not enable observability: {e}")

    # Initialize provider
    provider_kwargs = {}
    if config.is_reasoning_model is not None and config.provider == "openai":
        provider_kwargs["is_reasoning_model_override"] = config.is_reasoning_model
    provider = get_provider(config.provider, model=config.model, **provider_kwargs)
    print(f"  Provider: {config.provider}/{config.model}")

    # Initialize tools
    std_registry = create_default_registry()
    tools = std_registry.get_all_tools()

    mcp_client = None
    if config.mcp_enabled:
        try:
            mcp_client = await create_mcp_client()
            mcp_tools = mcp_client.list_tools()
            tools = tools + mcp_tools
            print(f"  MCP tools: {len(mcp_tools)} loaded")
        except Exception as e:
            print(f"  Warning: MCP unavailable: {e}")

    print(f"  Total tools: {len(tools)}")

    executor = build_tool_executor(std_registry, mcp_client)

    # Run benchmark
    results: List[BenchmarkResult] = []

    print_header("Running Benchmark")

    try:
        for i, question in enumerate(questions, 1):
            print_question(question, i, len(questions))

            # Create a new agent for each question
            agent = ReActAgent(
                provider=provider,
                tools=tools,
                tool_executor=executor,
                max_iterations=config.max_iterations,
            )

            result = await run_question(
                agent=agent,
                question=question,
                config=config,
                observer=observer,
            )

            results.append(result)
            print_result(result)

    finally:
        if mcp_client:
            await mcp_client.disconnect()
        if observer:
            observer.flush()
            observer.shutdown()

    # Summary
    print_header("Summary")

    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed
    total_tokens = sum(r.metrics.get("total_tokens", 0) for r in results)
    total_duration = sum(r.metrics.get("duration_seconds", 0) for r in results)

    print(f"  Provider: {config.provider}/{config.model}")
    print(f"  Questions: {len(results)}")
    print(f"  Passed: {passed} ({passed/len(results)*100:.0f}%)")
    print(f"  Failed: {failed}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total time: {total_duration:.1f}s")

    # Save results
    output_path = save_results(results, config)
    print(f"\n  Results saved: {output_path}")

    return 0 if failed == 0 else 1


def list_questions(config: BenchmarkConfig) -> int:
    """List available questions."""
    if not config.questions_file.exists():
        print(f"Error: Questions file not found: {config.questions_file}")
        return 1

    questions = load_questions(config.questions_file)

    print_header("Available Questions")
    print(f"  File: {config.questions_file}")
    print(f"  Total: {len(questions)} questions\n")

    for q in questions:
        print(f"  {q.id}. [{q.category}] [{q.difficulty}]")
        print(f"     {q.question[:100]}...")
        print()

    return 0


async def main() -> int:
    """Main entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run benchmark questions against model providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmark.py                              # Use default config
  python scripts/run_benchmark.py -c configs/my_config.yaml    # Custom config
  python scripts/run_benchmark.py --provider anthropic         # Override provider
  python scripts/run_benchmark.py --list-questions             # List questions

Config file (YAML):
  provider: openai
  model: gpt-4o-mini
  questions_file: data/questions.csv
  questions: [1, 2, 3]  # or "1-5" or null for all
  observability:
    enabled: true
    backend: json
  mcp:
    enabled: true
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("configs/benchmark_config.yaml"),
        help="Path to config file (default: configs/benchmark_config.yaml)",
    )
    parser.add_argument(
        "--provider", "-p",
        choices=list(PROVIDERS.keys()),
        help="Override provider from config",
    )
    parser.add_argument(
        "--model", "-m",
        help="Override model from config",
    )
    parser.add_argument(
        "--questions", "-q",
        help="Override questions (e.g., '1,2,3' or '1-5')",
    )
    parser.add_argument(
        "--list-questions", "-l",
        action="store_true",
        help="List available questions and exit",
    )
    parser.add_argument(
        "--no-observe",
        action="store_true",
        help="Disable observability",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Disable MCP tools",
    )
    parser.add_argument(
        "--reasoning-model",
        choices=["true", "false", "auto"],
        default=None,
        help="Override reasoning model detection (true/false/auto). "
             "Reasoning models use different API parameters (e.g., reasoning_effort instead of temperature)",
    )

    args = parser.parse_args()

    # Determine base path
    base_path = Path(__file__).parent.parent

    # Load config
    config = load_config(args.config, base_path)

    # Apply overrides
    if args.provider:
        config.provider = args.provider
        if not args.model:
            config.model = PROVIDERS[args.provider]["default_model"]
    if args.model:
        config.model = args.model
    if args.questions:
        config.questions = BenchmarkConfig._parse_questions(args.questions)
    if args.no_observe:
        config.observability_enabled = False
    if args.no_mcp:
        config.mcp_enabled = False
    if args.reasoning_model:
        if args.reasoning_model == "true":
            config.is_reasoning_model = True
        elif args.reasoning_model == "false":
            config.is_reasoning_model = False
        else:  # "auto"
            config.is_reasoning_model = None

    print_header("energBench Benchmark Runner")
    print_config(config)

    # List questions or run benchmark
    if args.list_questions:
        return list_questions(config)
    else:
        return await run_benchmark(config)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
