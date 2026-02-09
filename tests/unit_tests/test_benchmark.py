import json
from pathlib import Path

import pytest
import yaml

from energbench.agent.schema import ModelSpec
from energbench.benchmark.config import (
    PROVIDERS,
    BenchmarkConfig,
    ToolsConfig,
    load_config,
)
from energbench.benchmark.data_loader import load_questions
from energbench.benchmark.models import BenchmarkResult, Question
from energbench.benchmark.results import save_results
from energbench.core.errors import ConfigurationError


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_from_dict_multi_model(self, tmp_path):
        """Test config parsing with multi-model format."""
        questions_file = tmp_path / "test.csv"
        questions_file.write_text("S/N,Category,Question type,Difficulty level,Question\n1,Test,Type,Easy,Test?")

        data = {
            "models": [
                {"provider": "openai", "model": "gpt-4o-mini"},
                {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
            ],
            "questions_file": str(questions_file.name),
        }
        config = BenchmarkConfig.from_dict(data, tmp_path)

        assert len(config.models) == 2
        assert config.models[0].provider == "openai"
        assert config.models[0].model == "gpt-4o-mini"
        assert config.models[1].provider == "anthropic"

    def test_parse_questions_list(self):
        """Test parsing question list."""
        result = BenchmarkConfig._parse_questions([1, 2, 3])
        assert result == [1, 2, 3]

    def test_parse_questions_range(self):
        """Test parsing question range."""
        result = BenchmarkConfig._parse_questions("1-5")
        assert result == [1, 2, 3, 4, 5]

    def test_parse_questions_mixed(self):
        """Test parsing mixed question specification."""
        result = BenchmarkConfig._parse_questions("1,3,5-7")
        assert result == [1, 3, 5, 6, 7]

    def test_tools_config_defaults(self):
        """Test default tools configuration."""
        config = ToolsConfig()
        assert config.enabled is True
        assert config.include == []
        assert config.exclude == []

    def test_load_config_with_file(self, tmp_path):
        """Test loading config from YAML file."""
        questions_file = tmp_path / "test.csv"
        questions_file.write_text("S/N,Category,Question type,Difficulty level,Question\n1,Test,Type,Easy,Test?")

        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "models": [
                {"provider": "openai", "model": "gpt-4o-mini"},
            ],
            "questions_file": str(questions_file.name),
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file, tmp_path)
        assert config.models[0].provider == "openai"
        assert config.models[0].model == "gpt-4o-mini"

    def test_load_config_missing_file_raises(self):
        """Test loading config with explicit but missing path raises error."""
        with pytest.raises(ConfigurationError, match="Config file not found"):
            load_config(Path("nonexistent.yaml"), Path("."))

    def test_load_config_none_uses_defaults(self):
        """Test loading config with None path uses defaults."""
        config = load_config(None, Path("."))
        assert config.models[0].provider == "openai"
        assert config.models[0].model == "gpt-4o-mini"

    def test_from_dict_missing_models_raises(self, tmp_path):
        """Test that from_dict raises when 'models' key is missing."""
        questions_file = tmp_path / "test.csv"
        questions_file.write_text(
            "S/N,Category,Question type,Difficulty level,Question\n1,Test,Type,Easy,Test?"
        )
        data = {"provider": "openai", "model": "gpt-4o-mini"}
        with pytest.raises(ConfigurationError, match="must include a 'models' list"):
            BenchmarkConfig.from_dict(data, tmp_path)


class TestQuestionModels:
    """Tests for Question and related models."""

    def test_question_creation(self):
        """Test Question dataclass creation."""
        q = Question(
            id=1,
            category="Market Data",
            question_type="Retrieval",
            difficulty="Easy",
            question="What is the current price?",
        )
        assert q.id == 1
        assert q.category == "Market Data"
        assert q.question == "What is the current price?"

    def test_load_questions_from_csv(self, tmp_path):
        """Test loading questions from CSV file."""
        csv_file = tmp_path / "questions.csv"
        csv_content = """S/N,Category,Question type,Difficulty level,Question
1,Market Data,Retrieval,Easy,What is the price?
2,Analysis,Complex,Medium,Analyze the trend."""

        csv_file.write_text(csv_content)

        questions = load_questions(csv_file)
        assert len(questions) == 2
        assert questions[0].id == 1
        assert questions[0].category == "Market Data"
        assert questions[0].difficulty == "Easy"
        assert questions[1].id == 2
        assert questions[1].difficulty == "Medium"

    def test_load_questions_invalid_sn_raises(self, tmp_path):
        """Test that invalid S/N values in CSV raise a clear error."""
        csv_file = tmp_path / "bad_questions.csv"
        csv_content = """S/N,Category,Question type,Difficulty level,Question
abc,Market Data,Retrieval,Easy,What is the price?"""
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="Invalid S/N value 'abc' at row 2"):
            load_questions(csv_file)

    def test_load_questions_empty_sn_raises(self, tmp_path):
        """Test that empty S/N values in CSV raise a clear error."""
        csv_file = tmp_path / "empty_sn.csv"
        csv_content = """S/N,Category,Question type,Difficulty level,Question
,Market Data,Retrieval,Easy,What is the price?"""
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="Invalid S/N value"):
            load_questions(csv_file)


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_result_creation(self):
        """Test BenchmarkResult creation."""
        q = Question(1, "Test", "Type", "Easy", "Test question?")
        result = BenchmarkResult(
            question=q,
            provider="openai",
            model="gpt-4o-mini",
            success=True,
            answer="Test answer",
            error=None,
            metrics={"total_tokens": 100},
        )

        assert result.success is True
        assert result.provider == "openai"
        assert result.metrics["total_tokens"] == 100

    def test_result_with_error(self):
        """Test BenchmarkResult with error."""
        q = Question(1, "Test", "Type", "Easy", "Test question?")
        result = BenchmarkResult(
            question=q,
            provider="openai",
            model="gpt-4o-mini",
            success=False,
            answer=None,
            error="API Error",
        )

        assert result.success is False
        assert result.error == "API Error"


class TestResultsSaving:
    """Tests for results saving functionality."""

    def test_save_results_single_model(self, tmp_path):
        """Test saving results for single model."""
        questions_file = tmp_path / "test.csv"
        questions_file.write_text("S/N,Category,Question type,Difficulty level,Question\n1,Test,Type,Easy,Test?")

        q1 = Question(1, "Test", "Type", "Easy", "Question 1?")
        q2 = Question(2, "Test", "Type", "Medium", "Question 2?")

        results = [
            BenchmarkResult(
                question=q1,
                provider="openai",
                model="gpt-4o-mini",
                success=True,
                answer="Answer 1",
                error=None,
                metrics={"total_tokens": 100, "duration_seconds": 1.5},
            ),
            BenchmarkResult(
                question=q2,
                provider="openai",
                model="gpt-4o-mini",
                success=True,
                answer="Answer 2",
                error=None,
                metrics={"total_tokens": 150, "duration_seconds": 2.0},
            ),
        ]

        config = BenchmarkConfig(
            models=[ModelSpec(provider="openai", model="gpt-4o-mini")],
            questions_file=questions_file,
            questions=None,
            observability_enabled=False,
            observability_backend="json",
            observability_output_dir=Path("."),
            observability_run_name=None,
            mcp_enabled=False,
            max_iterations=10,
            results_dir=tmp_path,
            save_answers=True,
        )

        all_results = {"openai/gpt-4o-mini": results}
        output_path = save_results(all_results, config)

        assert output_path.exists()
        assert "benchmark_openai_" in output_path.name
        assert output_path.suffix == ".json"

        with open(output_path) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "config" in data
        assert "summary" in data
        assert "results_by_model" in data
        assert data["summary"]["total_questions"] == 2
        assert data["summary"]["models"]["openai/gpt-4o-mini"]["passed"] == 2
        assert data["summary"]["models"]["openai/gpt-4o-mini"]["total_tokens"] == 250

    def test_save_results_multi_model(self, tmp_path):
        """Test saving results for multiple models."""
        questions_file = tmp_path / "test.csv"
        questions_file.write_text("S/N,Category,Question type,Difficulty level,Question\n1,Test,Type,Easy,Test?")

        q = Question(1, "Test", "Type", "Easy", "Question?")

        results_openai = [
            BenchmarkResult(
                question=q,
                provider="openai",
                model="gpt-4o-mini",
                success=True,
                answer="Answer",
                error=None,
                metrics={"total_tokens": 100},
            )
        ]

        results_anthropic = [
            BenchmarkResult(
                question=q,
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                success=True,
                answer="Answer",
                error=None,
                metrics={"total_tokens": 120},
            )
        ]

        config = BenchmarkConfig(
            models=[
                ModelSpec(provider="openai", model="gpt-4o-mini"),
                ModelSpec(provider="anthropic", model="claude-sonnet-4-20250514"),
            ],
            questions_file=questions_file,
            questions=None,
            observability_enabled=False,
            observability_backend="json",
            observability_output_dir=Path("."),
            observability_run_name=None,
            mcp_enabled=False,
            max_iterations=10,
            results_dir=tmp_path,
            save_answers=True,
        )

        all_results = {
            "openai/gpt-4o-mini": results_openai,
            "anthropic/claude-sonnet-4-20250514": results_anthropic,
        }
        output_path = save_results(all_results, config)

        assert output_path.exists()
        assert "benchmark_multi_" in output_path.name

        with open(output_path) as f:
            data = json.load(f)

        assert "results_by_model" in data
        assert len(data["results_by_model"]) == 2
        assert "openai/gpt-4o-mini" in data["results_by_model"]
        assert "anthropic/claude-sonnet-4-20250514" in data["results_by_model"]


class TestProviders:
    """Tests for PROVIDERS constant."""

    def test_providers_structure(self):
        """Test PROVIDERS dict has correct structure."""
        assert "openai" in PROVIDERS
        assert "anthropic" in PROVIDERS
        assert "google" in PROVIDERS
        assert "deepinfra" in PROVIDERS

        for provider_name, provider_info in PROVIDERS.items():
            assert "default_model" in provider_info
            assert "models" in provider_info
            assert isinstance(provider_info["models"], list)
            assert len(provider_info["models"]) > 0
