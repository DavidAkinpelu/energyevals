#!/usr/bin/env python

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from energbench.agent.schema import ModelSpec
from energbench.benchmark import (
    PROVIDERS,
    BenchmarkConfig,
    list_questions,
    load_config,
    print_config,
    print_header,
    run_benchmark,
)
from energbench.benchmark import (
    list_tools as list_tools_func,
)
from energbench.mcp import create_mcp_client
from energbench.tools import create_default_registry


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmark questions against model providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmark.py                                    # Use default config
  python scripts/run_benchmark.py -c configs/my_config.yaml          # Custom config
  python scripts/run_benchmark.py --provider anthropic               # Single model override
  python scripts/run_benchmark.py --models openai:gpt-4o anthropic:claude-sonnet-4-20250514
  python scripts/run_benchmark.py --list-questions                   # List questions

Config file (YAML):
  models:
    - provider: openai
      model: gpt-4o-mini
    - provider: anthropic
      model: claude-sonnet-4-20250514
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/benchmark_config.yaml"),
        help="Path to config file (default: configs/benchmark_config.yaml)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=list(PROVIDERS.keys()),
        help="Override provider from config",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Override model from config (single model mode)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specify models as provider:model pairs (e.g., openai:gpt-4o anthropic:claude-sonnet-4)",
    )
    parser.add_argument(
        "--questions",
        "-q",
        help="Override questions (e.g., '1,2,3' or '1-5')",
    )
    parser.add_argument(
        "--list-questions",
        "-l",
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
    parser.add_argument(
        "--tools",
        help="Include only specific tools (comma-separated, e.g., 'battery,search,gridstatus')",
    )
    parser.add_argument(
        "--exclude-tools",
        help="Exclude specific tools (comma-separated, e.g., 'docket,tariffs')",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available tools and exit",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Number of independent trials per question (default: 1). "
        "Produces trial_N/ subdirectories in trace output.",
    )

    return parser.parse_args()


def apply_cli_overrides(config: BenchmarkConfig, args: argparse.Namespace) -> None:
    """Apply command line argument overrides to config.

    Args:
        config: Benchmark configuration to modify
        args: Parsed command line arguments
    """
    # Model overrides
    if args.models:
        # New --models flag takes precedence
        config.models = []
        for spec in args.models:
            if ":" not in spec:
                print(f"Error: Invalid model spec '{spec}'. Use format 'provider:model'")
                sys.exit(1)
            provider, model = spec.split(":", 1)
            if provider not in PROVIDERS:
                print(f"Error: Unknown provider '{provider}'")
                print(f"Available: {', '.join(PROVIDERS.keys())}")
                sys.exit(1)
            config.models.append(ModelSpec(provider=provider, model=model))
    elif args.provider or args.model:
        # Legacy single-model override
        if len(config.models) > 1:
            print("Warning: --provider/--model overrides multi-model config to single model")
        provider = args.provider or config.models[0].provider
        model = args.model or PROVIDERS[provider]["default_model"]
        is_reasoning = None
        if args.reasoning_model:
            is_reasoning = (
                args.reasoning_model == "true" if args.reasoning_model != "auto" else None
            )
        config.models = [
            ModelSpec(provider=provider, model=model, is_reasoning_model=is_reasoning)
        ]
    elif args.reasoning_model:
        # Apply reasoning model override to all models
        for m in config.models:
            if args.reasoning_model == "true":
                m.is_reasoning_model = True
            elif args.reasoning_model == "false":
                m.is_reasoning_model = False
            else:  # "auto"
                m.is_reasoning_model = None

    # Question overrides
    if args.questions:
        config.questions = BenchmarkConfig._parse_questions(args.questions)

    # Observability and MCP overrides
    if args.no_observe:
        config.observability_enabled = False
    if args.no_mcp:
        config.mcp_enabled = False

    # Trial override
    if args.num_trials is not None:
        config.num_trials = args.num_trials

    # Tool overrides
    if args.tools:
        tool_list = [t.strip() for t in args.tools.split(",")]
        config.tools_config.include = tool_list
        config.tools_config.exclude = []
    elif args.exclude_tools:
        exclude_list = [t.strip() for t in args.exclude_tools.split(",")]
        config.tools_config.exclude = exclude_list
        config.tools_config.include = []


async def handle_list_tools(config: BenchmarkConfig) -> int:
    """Handle --list-tools command.

    Args:
        config: Benchmark configuration

    Returns:
        Exit code
    """
    std_registry = create_default_registry()
    mcp_client = None
    if config.mcp_enabled:
        try:
            mcp_client = await create_mcp_client()
        except Exception as e:
            print(f"Warning: MCP unavailable: {e}")
    return list_tools_func(std_registry, mcp_client)


async def main() -> int:
    """Main entry point."""
    load_dotenv()

    args = parse_args()

    # Determine base path
    base_path = Path(__file__).parent.parent

    # Load config
    config = load_config(args.config, base_path)

    # Apply CLI overrides
    apply_cli_overrides(config, args)

    print_header("energBench Benchmark Runner")
    print_config(config)

    # Handle --list-tools
    if args.list_tools:
        return await handle_list_tools(config)

    # List questions or run benchmark
    if args.list_questions:
        return list_questions(config)
    else:
        return await run_benchmark(config)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
