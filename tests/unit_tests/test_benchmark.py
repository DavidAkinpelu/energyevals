import json
from pathlib import Path

import pytest
import yaml

from energbench.agent.schema import ModelSpec, ToolDefinition
from energbench.benchmark.config import (
    BenchmarkConfig,
    ToolsConfig,
    load_config,
)
from energbench.benchmark.data_loader import load_questions
from energbench.benchmark.models import BenchmarkResult, Question
from energbench.benchmark.results import save_results
from energbench.benchmark.tools import _build_tool_groups, _expand_names, filter_tools, merge_tools
from energbench.core.errors import ConfigurationError
from energbench.tools.base_tool import ToolRegistry


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

    def test_load_config_none_raises(self):
        """Test loading config with None path raises (no implicit defaults)."""
        with pytest.raises(ConfigurationError, match="config file path is required"):
            load_config(None, Path("."))

    def test_from_dict_missing_models_raises(self, tmp_path):
        """Test that from_dict raises when 'models' key is missing."""
        questions_file = tmp_path / "test.csv"
        questions_file.write_text(
            "S/N,Category,Question type,Difficulty level,Question\n1,Test,Type,Easy,Test?"
        )
        data = {"provider": "openai", "model": "gpt-4o-mini"}
        with pytest.raises(ConfigurationError, match="must include a 'models' list"):
            BenchmarkConfig.from_dict(data, tmp_path)

    def test_validate_invalid_seed_mode(self, tmp_path):
        """Invalid seed_mode should fail validation."""
        questions_file = tmp_path / "test.csv"
        questions_file.write_text(
            "S/N,Category,Question type,Difficulty level,Question\n1,Test,Type,Easy,Test?"
        )
        with pytest.raises(ConfigurationError, match="seed_mode must be one of"):
            BenchmarkConfig(
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
                seed_mode="bad_mode",
            )

    def test_validate_seeds_length_matches_trials(self, tmp_path):
        """seeds list length must match num_trials."""
        questions_file = tmp_path / "test.csv"
        questions_file.write_text(
            "S/N,Category,Question type,Difficulty level,Question\n1,Test,Type,Easy,Test?"
        )
        with pytest.raises(ConfigurationError, match="seeds length"):
            BenchmarkConfig(
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
                shuffle=True,
                num_trials=3,
                seeds=[1, 2],
            )

    def test_validate_seeds_requires_shuffle(self, tmp_path):
        """Providing seeds requires shuffle=true."""
        questions_file = tmp_path / "test.csv"
        questions_file.write_text(
            "S/N,Category,Question type,Difficulty level,Question\n1,Test,Type,Easy,Test?"
        )
        with pytest.raises(ConfigurationError, match="seeds requires shuffle=true"):
            BenchmarkConfig(
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
                shuffle=False,
                num_trials=2,
                seeds=[1, 2],
            )


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

        all_results = {"openai/gpt-4o-mini": {1: results}}
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
            "openai/gpt-4o-mini": {1: results_openai},
            "anthropic/claude-sonnet-4-20250514": {1: results_anthropic},
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


class TestProviderValidation:
    """Tests for provider validation in BenchmarkConfig."""

    def test_known_provider_is_accepted(self, tmp_path):
        questions_file = tmp_path / "test.csv"
        questions_file.write_text(
            "S/N,Category,Question type,Difficulty level,Question\n1,Test,Type,Easy,Test?"
        )
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
        assert config.models[0].provider == "openai"

    def test_unknown_provider_is_rejected(self, tmp_path):
        questions_file = tmp_path / "test.csv"
        questions_file.write_text(
            "S/N,Category,Question type,Difficulty level,Question\n1,Test,Type,Easy,Test?"
        )
        with pytest.raises(ConfigurationError, match="Unknown provider"):
            BenchmarkConfig(
                models=[ModelSpec(provider="unknown-provider", model="whatever")],
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


class TestToolGroupFiltering:
    """Tests for group-based tool filtering (Issue #7)."""

    @pytest.fixture
    def mock_registry(self):
        """Build a ToolRegistry with a pre-populated _method_to_tool mapping."""
        reg = ToolRegistry()
        reg._method_to_tool = {
            "search_web": "search",
            "get_page_contents": "search",
            "battery_revenue_optimization": "battery_optimization",
            "list_gridstatus_datasets": "gridstatus_api_tool",
            "query_gridstatus_dataset": "gridstatus_api_tool",
        }
        return reg

    @pytest.fixture
    def sample_tools(self):
        return [
            ToolDefinition(name="search_web", description="Search the web"),
            ToolDefinition(name="get_page_contents", description="Get page"),
            ToolDefinition(name="battery_revenue_optimization", description="Battery opt"),
            ToolDefinition(name="list_gridstatus_datasets", description="List datasets"),
            ToolDefinition(name="query_gridstatus_dataset", description="Query data"),
        ]

    def test_include_group(self, sample_tools, mock_registry):
        cfg = ToolsConfig(enabled=True, include=["search"], exclude=[])
        result = filter_tools(sample_tools, cfg, registry=mock_registry)
        names = {t.name for t in result}
        assert names == {"search_web", "get_page_contents"}

    def test_exclude_group(self, sample_tools, mock_registry):
        cfg = ToolsConfig(enabled=True, include=[], exclude=["gridstatus_api_tool"])
        result = filter_tools(sample_tools, cfg, registry=mock_registry)
        names = {t.name for t in result}
        assert "list_gridstatus_datasets" not in names
        assert "query_gridstatus_dataset" not in names
        assert "search_web" in names

    def test_mix_group_and_individual(self, sample_tools, mock_registry):
        cfg = ToolsConfig(enabled=True, include=["search", "battery_revenue_optimization"], exclude=[])
        result = filter_tools(sample_tools, cfg, registry=mock_registry)
        names = {t.name for t in result}
        assert names == {"search_web", "get_page_contents", "battery_revenue_optimization"}

    def test_expand_names_group(self, mock_registry):
        groups = _build_tool_groups(mock_registry)
        expanded = _expand_names(["battery_optimization"], groups)
        assert expanded == {"battery_revenue_optimization"}

    def test_expand_names_individual(self, mock_registry):
        groups = _build_tool_groups(mock_registry)
        expanded = _expand_names(["search_web"], groups)
        assert expanded == {"search_web"}

    def test_expand_names_mixed(self, mock_registry):
        groups = _build_tool_groups(mock_registry)
        expanded = _expand_names(["battery_optimization", "search_web"], groups)
        assert expanded == {"battery_revenue_optimization", "search_web"}

    def test_build_tool_groups(self, mock_registry):
        groups = _build_tool_groups(mock_registry)
        assert groups["search"] == {"search_web", "get_page_contents"}
        assert groups["battery_optimization"] == {"battery_revenue_optimization"}
        assert groups["gridstatus_api_tool"] == {"list_gridstatus_datasets", "query_gridstatus_dataset"}


class TestMergeTools:
    """Tests for merge_tools() — MCP-first ordering with std-wins deduplication."""

    @pytest.fixture
    def std_tools(self):
        return [
            ToolDefinition(name="std_only", description="Standard exclusive tool"),
            ToolDefinition(name="shared_tool", description="Standard version of shared tool"),
        ]

    @pytest.fixture
    def mcp_tools(self):
        return [
            ToolDefinition(name="mcp_only", description="MCP exclusive tool"),
            ToolDefinition(name="shared_tool", description="MCP version of shared tool"),
        ]

    def test_no_collision_mcp_first(self):
        """MCP-unique tools appear before standard tools when no name collision."""
        std = [ToolDefinition(name="std_a", description="std")]
        mcp = [ToolDefinition(name="mcp_a", description="mcp")]
        result = merge_tools(std, mcp)
        names = [t.name for t in result]
        assert names.index("mcp_a") < names.index("std_a")

    def test_std_wins_on_collision(self, std_tools, mcp_tools):
        """On name collision the standard ToolDefinition is kept; MCP version is absent."""
        result = merge_tools(std_tools, mcp_tools)
        names = [t.name for t in result]
        # collision name appears exactly once
        assert names.count("shared_tool") == 1
        # total length == unique names
        assert len(result) == 3  # mcp_only, std_only, shared_tool
        # the retained object is the std version
        shared = next(t for t in result if t.name == "shared_tool")
        assert shared.description == "Standard version of shared tool"

    def test_collision_logs_warning(self, std_tools, mcp_tools, caplog):
        """A warning containing the colliding tool name is emitted for each collision."""
        import logging
        with caplog.at_level(logging.WARNING, logger="energbench.benchmark.tools"):
            merge_tools(std_tools, mcp_tools)
        assert any("shared_tool" in record.message for record in caplog.records)
        assert any("Standard tool takes priority" in record.message for record in caplog.records)

    def test_empty_mcp(self, std_tools):
        """With no MCP tools the result equals std_tools in original order."""
        result = merge_tools(std_tools, [])
        assert result == std_tools

    def test_empty_std(self, mcp_tools):
        """With no standard tools the result equals mcp_tools in original order."""
        result = merge_tools([], mcp_tools)
        assert result == mcp_tools
