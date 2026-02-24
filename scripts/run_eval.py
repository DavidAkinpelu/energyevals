#!/usr/bin/env python

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from energbench.evaluation.config import EvalConfig, load_eval_config
from energbench.evaluation.runner import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM-judge evaluation over benchmark traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/run_eval.py                                    # Default config
  python scripts/run_eval.py -c configs/eval_config.yaml        # Custom config
  python scripts/run_eval.py --run-name eval_samples            # Specify run
  python scripts/run_eval.py --questions 1,3,5                  # Subset
  python scripts/run_eval.py --model openai_gpt-4.1             # One model
  python scripts/run_eval.py --compare                          # Cross-model tests
""",
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("configs/eval_config.yaml"),
        help="Path to eval config YAML (default: configs/eval_config.yaml)",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Override path to benchmark trace directory",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Override path to dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override evaluation output directory",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name (matches benchmark run_name for pairing traces)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Evaluate only this model directory name (e.g. openai_gpt-4.1)",
    )
    parser.add_argument(
        "--questions", "-q",
        default=None,
        help="Comma-separated question IDs to evaluate (e.g. 1,3,5)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Run paired cross-model significance tests",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Override the judge LLM model (e.g. gpt-4o)",
    )

    return parser.parse_args()


def apply_cli_overrides(config: EvalConfig, args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to the loaded config."""
    if args.results_path is not None:
        config.results_path = args.results_path
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.model is not None:
        config.models = [args.model]
    if args.questions is not None:
        config.questions = [int(q.strip()) for q in args.questions.split(",")]
    if args.compare:
        config.compare = True
    if args.judge_model is not None:
        config.judge.model = args.judge_model


def main() -> int:
    load_dotenv()
    args = parse_args()
    base_path = Path(__file__).parent.parent

    config_path = args.config if args.config.exists() else None
    config = load_eval_config(config_path, base_path=base_path)
    apply_cli_overrides(config, args)

    print(f"{'=' * 70}")
    print("  energBench Evaluation Pipeline")
    print(f"{'=' * 70}")
    print(f"  Judge model:   {config.judge.model}")
    print(f"  Results path:  {config.results_path}")
    print(f"  Dataset:       {config.dataset_path}")
    print(f"  Output dir:    {config.output_dir}")
    print(f"  Run name:      {config.run_name or '(auto-discover)'}")
    print(f"  Models filter: {config.models or 'all'}")
    print(f"  Questions:     {config.questions or 'all'}")
    print(f"  Compare:       {config.compare}")
    print(f"  Strategy:      default={config.default_strategy}, categories={config.category_strategies or '{}'}")

    reports = asyncio.run(run_evaluation(config))

    if not reports:
        print("\n  No reports generated. Check your paths and trace directories.")
        return 1

    print(f"\n  Evaluation complete. {len(reports)} model(s) evaluated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
