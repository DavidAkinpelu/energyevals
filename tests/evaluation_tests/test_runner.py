import json
from pathlib import Path

import pytest

from energbench.evaluation import runner as eval_runner
from energbench.evaluation.config import EvalConfig, load_eval_config
from energbench.evaluation.models import (
    AccuracyResult,
    ApproachResult,
    AttributeAlignmentResult,
    BenchmarkResultEntry,
    CostEstimate,
    EvaluationReport,
    JudgeScore,
    LatencyBreakdown,
    MetricScore,
    ScoreStatistics,
    SourceResult,
    TrialEval,
)


class TestRowLookupBySerialNumber:
    """Regression tests for S/N-based row lookup (Issue #2)."""

    def test_noncontiguous_sn_selects_correct_row(self):
        """rows_by_id must match by S/N value, not list position."""
        eval_data = [
            {"S/N": "3", "Question": "Q3", "Answer": "A3", "Approach": "", "Category": "", "Difficulty level": ""},
            {"S/N": "7", "Question": "Q7", "Answer": "A7", "Approach": "", "Category": "", "Difficulty level": ""},
            {"S/N": "11", "Question": "Q11", "Answer": "A11", "Approach": "", "Category": "", "Difficulty level": ""},
        ]
        rows_by_id = {int(r["S/N"]): r for r in eval_data}

        assert rows_by_id[3]["Question"] == "Q3"
        assert rows_by_id[7]["Question"] == "Q7"
        assert rows_by_id[11]["Question"] == "Q11"

    def test_shuffled_sn_selects_correct_row(self):
        """S/N lookup must work regardless of row ordering."""
        eval_data = [
            {"S/N": "5", "Question": "Q5", "Answer": "A5", "Approach": "", "Category": "", "Difficulty level": ""},
            {"S/N": "2", "Question": "Q2", "Answer": "A2", "Approach": "", "Category": "", "Difficulty level": ""},
            {"S/N": "8", "Question": "Q8", "Answer": "A8", "Approach": "", "Category": "", "Difficulty level": ""},
        ]
        rows_by_id = {int(r["S/N"]): r for r in eval_data}

        assert rows_by_id[2]["Question"] == "Q2"
        assert rows_by_id[5]["Question"] == "Q5"
        assert rows_by_id[8]["Question"] == "Q8"

    def test_missing_sn_raises_keyerror(self):
        """Accessing a non-existent S/N must raise KeyError."""
        eval_data = [
            {"S/N": "1", "Question": "Q1"},
            {"S/N": "3", "Question": "Q3"},
        ]
        rows_by_id = {int(r["S/N"]): r for r in eval_data}

        with pytest.raises(KeyError):
            _ = rows_by_id[2]


class TestJudgeScoreScale:
    """Tests for raw score scale handling in evaluation runner."""

    @pytest.mark.asyncio
    async def test_evaluate_trial_keeps_raw_approach_and_source_scores(self, monkeypatch):
        """Approach/sources should remain on their native 1-5 scale."""
        monkeypatch.setattr(
            eval_runner,
            "load_benchmark_result",
            lambda *args, **kwargs: BenchmarkResultEntry(
                answer="answer",
                steps_trace="[]",
                metrics=MetricScore(),
            ),
        )
        monkeypatch.setattr(eval_runner, "get_strategy", lambda *args, **kwargs: "accuracy")

        async def _judge_approach(*args, **kwargs):
            return ApproachResult(approach_correctness=4, reasoning="good")

        async def _judge_sources(*args, **kwargs):
            return SourceResult(source_validity=2, reasoning="weak sources")

        async def _judge_accuracy(*args, **kwargs):
            return AccuracyResult(accuracy_score=0.75, reasoning="mostly correct")

        monkeypatch.setattr(eval_runner, "judge_approach", _judge_approach)
        monkeypatch.setattr(eval_runner, "judge_sources", _judge_sources)
        monkeypatch.setattr(eval_runner, "judge_accuracy", _judge_accuracy)

        result = await eval_runner._evaluate_trial(
            provider=object(),  # provider internals are mocked out via judge functions
            question_num=1,
            question_text="Q",
            expected_answer="A",
            suggested_steps="S",
            category="cat",
            trace_base=Path("."),
            trial=None,
            trial_index=1,
            config=EvalConfig(),
        )

        assert result.approach.score == 4.0
        assert result.sources.score == 2.0
        assert result.accuracy.score == 0.75
        assert result.accuracy.judge_type == "accuracy"

    @pytest.mark.asyncio
    async def test_evaluate_trial_attributes_path_keeps_alignment_score(self, monkeypatch):
        """Attributes strategy should pass through 0-1 alignment score unchanged."""
        monkeypatch.setattr(
            eval_runner,
            "load_benchmark_result",
            lambda *args, **kwargs: BenchmarkResultEntry(
                answer="answer",
                steps_trace="[]",
                metrics=MetricScore(),
            ),
        )
        monkeypatch.setattr(eval_runner, "get_strategy", lambda *args, **kwargs: "attributes")

        async def _judge_approach(*args, **kwargs):
            return ApproachResult(approach_correctness=5, reasoning="excellent")

        async def _judge_sources(*args, **kwargs):
            return SourceResult(source_validity=3, reasoning="adequate")

        async def _judge_attributes(*args, **kwargs):
            return AttributeAlignmentResult(
                total_attributes=2,
                matched_attributes=1,
                alignment_score=0.5,
                attribute_details=[],
                reasoning="partial match",
            )

        monkeypatch.setattr(eval_runner, "judge_approach", _judge_approach)
        monkeypatch.setattr(eval_runner, "judge_sources", _judge_sources)
        monkeypatch.setattr(eval_runner, "judge_attributes", _judge_attributes)

        result = await eval_runner._evaluate_trial(
            provider=object(),
            question_num=1,
            question_text="Q",
            expected_answer="A",
            suggested_steps="S",
            category="cat",
            trace_base=Path("."),
            trial=None,
            trial_index=1,
            config=EvalConfig(),
        )

        assert result.approach.score == 5.0
        assert result.sources.score == 3.0
        assert result.accuracy.score == 0.5
        assert result.accuracy.judge_type == "attributes"


class TestTrialFailureIsolation:
    """Broad exception handler must skip failed trials without aborting the run."""

    @pytest.mark.asyncio
    async def test_exception_in_trial_does_not_abort_model_eval(self, tmp_path, monkeypatch):
        """A JSONDecodeError on trial 1 must not prevent trial 2 from being evaluated."""
        good_trial = TrialEval(
            trial=2,
            approach=JudgeScore(score=3.0, reasoning="ok", judge_type="approach"),
            accuracy=JudgeScore(score=0.5, reasoning="ok", judge_type="accuracy"),
            sources=JudgeScore(score=3.0, reasoning="ok", judge_type="sources"),
        )
        call_count = 0

        async def _flaky_trial(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise json.JSONDecodeError("bad json", "", 0)
            return good_trial

        monkeypatch.setattr(eval_runner, "_evaluate_trial", _flaky_trial)
        monkeypatch.setattr(eval_runner, "discover_trials", lambda *a, **kw: [1, 2])

        eval_data = [{"S/N": "1", "Question": "Q", "Answer": "A", "Approach": "", "Category": "", "Difficulty level": ""}]
        ground_truths = {1: object()}

        report = await eval_runner._evaluate_model(
            provider=object(),
            model_dir=tmp_path / "model",
            ground_truths=ground_truths,
            eval_data=eval_data,
            config=EvalConfig(),
        )

        assert len(report.questions) == 1
        assert len(report.questions[0].trials) == 1
        assert report.questions[0].trials[0].trial == 2


class TestAggregateMetrics:
    """_aggregate_metrics must average latency and cost nested fields."""

    def test_latency_and_cost_are_averaged(self):
        m1 = MetricScore(
            tool_calls=2,
            total_tokens=100,
            duration_seconds=1.0,
            latency=LatencyBreakdown(
                wall_clock_ms=200.0,
                llm_thinking_ms=100.0,
                tool_execution_ms=50.0,
                per_tool_ms={"sql": 30.0},
            ),
            cost=CostEstimate(
                input_tokens=80,
                output_tokens=20,
                cached_tokens=0,
                reasoning_tokens=0,
                estimated_cost_usd=0.002,
            ),
        )
        m2 = MetricScore(
            tool_calls=4,
            total_tokens=200,
            duration_seconds=3.0,
            latency=LatencyBreakdown(
                wall_clock_ms=400.0,
                llm_thinking_ms=200.0,
                tool_execution_ms=100.0,
                per_tool_ms={"sql": 70.0},
            ),
            cost=CostEstimate(
                input_tokens=160,
                output_tokens=40,
                cached_tokens=10,
                reasoning_tokens=5,
                estimated_cost_usd=0.006,
            ),
        )

        result = eval_runner._aggregate_metrics([m1, m2])

        assert result.tool_calls == 3
        assert result.total_tokens == 150
        assert result.duration_seconds == pytest.approx(2.0)
        assert result.latency.wall_clock_ms == pytest.approx(300.0)
        assert result.latency.llm_thinking_ms == pytest.approx(150.0)
        assert result.latency.tool_execution_ms == pytest.approx(75.0)
        assert result.latency.per_tool_ms["sql"] == pytest.approx(50.0)
        assert result.cost.input_tokens == 120
        assert result.cost.output_tokens == 30
        assert result.cost.estimated_cost_usd == pytest.approx(0.004)


class TestEvalConfigFastFail:
    """load_eval_config must raise FileNotFoundError for an explicit nonexistent path."""

    def test_nonexistent_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_eval_config(Path("/nonexistent/path/config.yaml"))


class TestEvalConfigExtension:
    """Config snapshot must be written as eval_config.json, not eval_config.yaml."""

    @pytest.mark.asyncio
    async def test_config_snapshot_extension_is_json(self, tmp_path, monkeypatch):
        saved_paths: list[Path] = []
        monkeypatch.setattr(eval_runner, "_save_json", lambda p, d: saved_paths.append(p))
        monkeypatch.setattr(eval_runner, "create_judge_provider", lambda cfg: object())
        monkeypatch.setattr(eval_runner, "load_eval_data", lambda p: [])
        monkeypatch.setattr(eval_runner, "load_ground_truth", lambda p: {})
        monkeypatch.setattr(eval_runner, "_discover_model_dirs", lambda *a, **kw: [tmp_path / "model_a"])
        monkeypatch.setattr(eval_runner, "_write_report", lambda *a, **kw: None)
        monkeypatch.setattr(eval_runner, "_print_report", lambda *a, **kw: None)

        stat = ScoreStatistics(mean=0.5, std=0.1, ci_lower=0.3, ci_upper=0.7, n=1)
        fake_report = EvaluationReport(
            model="model_a",
            run_name="test",
            num_trials=1,
            questions=[],
            aggregate_approach=stat,
            aggregate_accuracy=stat,
            aggregate_sources=stat,
            aggregate_metrics=MetricScore(),
        )

        async def _fake_evaluate_model(*args, **kwargs):
            return fake_report

        monkeypatch.setattr(eval_runner, "_evaluate_model", _fake_evaluate_model)

        config = EvalConfig(output_dir=tmp_path)
        await eval_runner.run_evaluation(config)

        config_paths = [p for p in saved_paths if "eval_config" in p.name]
        assert config_paths, "No eval_config path was saved"
        assert all(p.suffix == ".json" for p in config_paths)
        assert not any(p.suffix == ".yaml" for p in config_paths)
