"""Tests for energyevals/evaluation/data_loader.py using tmp_path fixtures."""

import json
import time
from pathlib import Path

import pytest

from energyevals.evaluation.data_loader import (
    _resolve_trace_file,
    discover_trials,
    load_benchmark_result,
    load_ground_truth,
)


class TestDiscoverTrials:
    def test_finds_trial_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "trial_1").mkdir()
        (tmp_path / "trial_2").mkdir()
        (tmp_path / "other_dir").mkdir()  # should be ignored

        result = discover_trials(tmp_path)
        assert result == [1, 2]

    def test_fallback_no_trials(self, tmp_path: Path) -> None:
        (tmp_path / "some_trace.json").touch()

        result = discover_trials(tmp_path)
        assert result == [None]

    def test_fallback_nonexistent_path(self, tmp_path: Path) -> None:
        result = discover_trials(tmp_path / "nonexistent")
        assert result == [None]

    def test_returns_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "trial_3").mkdir()
        (tmp_path / "trial_1").mkdir()

        result = discover_trials(tmp_path)
        assert result == [1, 3]


class TestLoadGroundTruth:
    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_parses_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dataset.csv"
        self._write_csv(csv_path, [
            {
                "S/N": "1",
                "Question": "What is ERCOT?",
                "Answer": "Texas grid operator",
                "Approach": "Look up grid operators",
                "Category": "Markets",
                "Question type": "Factual",
                "Difficulty level": "Easy",
            },
            {
                "S/N": "2",
                "Question": "What is LMP?",
                "Answer": "Locational Marginal Price",
                "Approach": "Look up pricing concepts",
                "Category": "Pricing",
                "Question type": "Factual",
                "Difficulty level": "Medium",
            },
        ])

        result = load_ground_truth(csv_path)

        assert 1 in result
        assert 2 in result
        assert result[1].answer == "Texas grid operator"
        assert result[1].category == "Markets"
        assert result[2].answer == "Locational Marginal Price"

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        import csv
        csv_path = tmp_path / "bad.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["S/N", "Question"])
            writer.writeheader()
            writer.writerow({"S/N": "1", "Question": "test"})

        with pytest.raises(ValueError, match="missing required columns"):
            load_ground_truth(csv_path)


class TestLoadBenchmarkResult:
    def _write_trace(self, trace_dir: Path, question_num: int, data: dict) -> Path:
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"trace_q{question_num}_abc123.json"
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_reads_trace(self, tmp_path: Path) -> None:
        trace_data = {
            "final_answer": "42",
            "steps": [
                {"step_type": "thought", "latency_ms": 100.0},
                {"step_type": "observation", "latency_ms": 50.0, "tool_name": "search"},
            ],
            "metrics": {
                "total_input_tokens": 500,
                "total_output_tokens": 100,
                "total_tokens": 600,
                "tool_calls_count": 1,
                "duration_seconds": 3.5,
            },
        }
        self._write_trace(tmp_path, 1, trace_data)

        result = load_benchmark_result(tmp_path, question_num=1, trial=None)

        assert result.answer == "42"
        assert result.metrics.latency.llm_thinking_ms == 100.0
        assert result.metrics.latency.tool_execution_ms == 50.0
        assert result.metrics.cost.input_tokens == 500

    def test_reads_trial_subdirectory(self, tmp_path: Path) -> None:
        trial_dir = tmp_path / "trial_1"
        trace_data = {"final_answer": "hello", "steps": [], "metrics": {}}
        self._write_trace(trial_dir, 1, trace_data)

        result = load_benchmark_result(tmp_path, question_num=1, trial=1)

        assert result.answer == "hello"

    def test_missing_trace_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_benchmark_result(tmp_path, question_num=99, trial=None)


class TestResolveTraceFile:
    def test_finds_single_file(self, tmp_path: Path) -> None:
        f = tmp_path / "trace_q1_abc.json"
        f.touch()

        result = _resolve_trace_file(tmp_path, 1)
        assert result == f

    def test_raises_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _resolve_trace_file(tmp_path, 99)

    def test_sorts_by_mtime_newest_first(self, tmp_path: Path) -> None:
        old = tmp_path / "trace_q1_old.json"
        old.touch()
        # Sleep briefly then create newer file
        time.sleep(0.01)
        new = tmp_path / "trace_q1_new.json"
        new.touch()

        result = _resolve_trace_file(tmp_path, 1)
        assert result == new
