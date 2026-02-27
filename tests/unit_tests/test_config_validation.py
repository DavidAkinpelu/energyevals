from pathlib import Path

import pytest

from energbench.agent.schema import ModelSpec
from energbench.benchmark.config import BenchmarkConfig
from energbench.core.errors import ConfigurationError


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig validation."""

    def test_valid_config(self, tmp_path):
        """Test creating a valid configuration."""
        questions_file = tmp_path / "questions.csv"
        questions_file.touch()

        config = BenchmarkConfig(
            models=[ModelSpec(provider="openai", model="gpt-4o-mini")],
            questions_file=questions_file,
            questions=None,
            observability_enabled=True,
            observability_backend="json",
            observability_output_dir=Path("./traces"),
            observability_run_name=None,
            mcp_enabled=True,
            max_iterations=25,
            results_dir=Path("./results"),
            save_answers=True,
        )

        assert len(config.models) == 1
        assert config.models[0].provider == "openai"

    def test_empty_models_raises_error(self, tmp_path):
        """Test that empty models list raises error."""
        questions_file = tmp_path / "questions.csv"
        questions_file.touch()

        with pytest.raises(ConfigurationError) as exc_info:
            BenchmarkConfig(
                models=[],
                questions_file=questions_file,
                questions=None,
                observability_enabled=True,
                observability_backend="json",
                observability_output_dir=Path("./traces"),
                observability_run_name=None,
                mcp_enabled=True,
                max_iterations=25,
                results_dir=Path("./results"),
                save_answers=True,
            )

        assert "At least one model must be specified" in str(exc_info.value)

    def test_missing_questions_file_raises_error(self):
        """Test that missing questions file raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            BenchmarkConfig(
                models=[ModelSpec(provider="openai", model="gpt-4o-mini")],
                questions_file=Path("/nonexistent/file.csv"),
                questions=None,
                observability_enabled=True,
                observability_backend="json",
                observability_output_dir=Path("./traces"),
                observability_run_name=None,
                mcp_enabled=True,
                max_iterations=25,
                results_dir=Path("./results"),
                save_answers=True,
            )

        assert "Questions file not found" in str(exc_info.value)

    def test_invalid_backend_raises_error(self, tmp_path):
        """Test that invalid observability backend raises error."""
        questions_file = tmp_path / "questions.csv"
        questions_file.touch()

        with pytest.raises(ConfigurationError) as exc_info:
            BenchmarkConfig(
                models=[ModelSpec(provider="openai", model="gpt-4o-mini")],
                questions_file=questions_file,
                questions=None,
                observability_enabled=True,
                observability_backend="invalid_backend",
                observability_output_dir=Path("./traces"),
                observability_run_name=None,
                mcp_enabled=True,
                max_iterations=25,
                results_dir=Path("./results"),
                save_answers=True,
            )

        assert "Invalid observability backend" in str(exc_info.value)

    @pytest.mark.parametrize("backend", ["json", "langfuse", "both", "auto"])
    def test_valid_observability_backends(self, backend, tmp_path):
        """All documented backends must be accepted."""
        questions_file = tmp_path / "questions.csv"
        questions_file.touch()

        config = BenchmarkConfig(
            models=[ModelSpec(provider="openai", model="gpt-4o-mini")],
            questions_file=questions_file,
            questions=None,
            observability_enabled=True,
            observability_backend=backend,
            observability_output_dir=Path("./traces"),
            observability_run_name=None,
            mcp_enabled=True,
            max_iterations=25,
            results_dir=Path("./results"),
            save_answers=True,
        )
        assert config.observability_backend == backend

    def test_invalid_max_iterations_raises_error(self, tmp_path):
        """Test that invalid max_iterations raises error."""
        questions_file = tmp_path / "questions.csv"
        questions_file.touch()

        with pytest.raises(ConfigurationError) as exc_info:
            BenchmarkConfig(
                models=[ModelSpec(provider="openai", model="gpt-4o-mini")],
                questions_file=questions_file,
                questions=None,
                observability_enabled=True,
                observability_backend="json",
                observability_output_dir=Path("./traces"),
                observability_run_name=None,
                mcp_enabled=True,
                max_iterations=0,
                results_dir=Path("./results"),
                save_answers=True,
            )

        assert "max_iterations must be at least 1" in str(exc_info.value)

    def test_invalid_tool_output_log_mode_raises_error(self, tmp_path):
        """Invalid tool output log mode should fail validation."""
        questions_file = tmp_path / "questions.csv"
        questions_file.touch()

        with pytest.raises(ConfigurationError) as exc_info:
            BenchmarkConfig(
                models=[ModelSpec(provider="openai", model="gpt-4o-mini")],
                questions_file=questions_file,
                questions=None,
                observability_enabled=True,
                observability_backend="json",
                observability_output_dir=Path("./traces"),
                observability_run_name=None,
                mcp_enabled=True,
                max_iterations=25,
                results_dir=Path("./results"),
                save_answers=True,
                tool_output_log_mode="invalid",
            )

        assert "tool_output_log_mode must be one of" in str(exc_info.value)

    def test_negative_tool_output_log_max_chars_raises_error(self, tmp_path):
        """Negative preview max chars should fail validation."""
        questions_file = tmp_path / "questions.csv"
        questions_file.touch()

        with pytest.raises(ConfigurationError) as exc_info:
            BenchmarkConfig(
                models=[ModelSpec(provider="openai", model="gpt-4o-mini")],
                questions_file=questions_file,
                questions=None,
                observability_enabled=True,
                observability_backend="json",
                observability_output_dir=Path("./traces"),
                observability_run_name=None,
                mcp_enabled=True,
                max_iterations=25,
                results_dir=Path("./results"),
                save_answers=True,
                tool_output_log_max_chars=-1,
            )

        assert "tool_output_log_max_chars must be non-negative" in str(exc_info.value)
