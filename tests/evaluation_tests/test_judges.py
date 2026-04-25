from unittest.mock import AsyncMock, MagicMock

import pytest

from energyevals.agent.providers import BaseProvider
from energyevals.evaluation.config import JudgeConfig
from energyevals.evaluation.judges import (
    _build_schema_instruction,
    _judge_call,
    create_judge_provider,
    judge_approach,
)
from energyevals.evaluation.models import ApproachResult


class TestCreateJudgeProvider:
    """Tests for create_judge_provider (Issue #3)."""

    def test_openai_provider(self):
        cfg = JudgeConfig(provider="openai", model="gpt-4o")
        provider = create_judge_provider(cfg)
        assert provider.provider_name == "openai"
        assert provider.model == "gpt-4o"

    def test_anthropic_provider(self):
        cfg = JudgeConfig(provider="anthropic", model="claude-sonnet-4-20250514")
        provider = create_judge_provider(cfg)
        assert provider.provider_name == "anthropic"

    def test_google_provider(self):
        cfg = JudgeConfig(provider="google", model="gemini-2.0-flash")
        provider = create_judge_provider(cfg)
        assert provider.provider_name == "google"

    def test_reasoning_effort_does_not_force_non_reasoning(self):
        cfg = JudgeConfig(provider="openai", model="gpt-5-mini", reasoning_effort="low")
        provider = create_judge_provider(cfg)
        assert provider.provider_name == "openai"
        assert getattr(provider, "is_reasoning_model", False) is False

    def test_unsupported_provider_raises(self):
        cfg = JudgeConfig(provider="some_unknown", model="x")
        with pytest.raises(ValueError, match="Unknown provider"):
            create_judge_provider(cfg)


class TestBuildSchemaInstruction:
    def test_includes_schema(self):
        instruction = _build_schema_instruction(ApproachResult)
        assert "approach_correctness" in instruction
        assert "reasoning" in instruction
        assert "JSON" in instruction


class TestJudgeCall:
    """Tests for _judge_call structured output parsing."""

    @pytest.mark.asyncio
    async def test_parses_json_response(self):
        mock_provider = MagicMock(spec=BaseProvider)
        mock_response = MagicMock()
        mock_response.content = '{"approach_correctness": 4, "reasoning": "Good approach"}'
        mock_provider.complete = AsyncMock(return_value=mock_response)

        cfg = JudgeConfig(model="gpt-4o", temperature=0.0)
        result = await _judge_call(mock_provider, cfg, "test prompt", ApproachResult)

        assert isinstance(result, ApproachResult)
        assert result.approach_correctness == 4
        assert result.reasoning == "Good approach"

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        mock_provider = MagicMock(spec=BaseProvider)
        mock_response = MagicMock()
        mock_response.content = '```json\n{"approach_correctness": 3, "reasoning": "Ok"}\n```'
        mock_provider.complete = AsyncMock(return_value=mock_response)

        cfg = JudgeConfig(model="gpt-4o")
        result = await _judge_call(mock_provider, cfg, "test prompt", ApproachResult)

        assert result.approach_correctness == 3

    @pytest.mark.asyncio
    async def test_passes_temperature_and_max_tokens(self):
        mock_provider = MagicMock(spec=BaseProvider)
        mock_response = MagicMock()
        mock_response.content = '{"approach_correctness": 5, "reasoning": "Great"}'
        mock_provider.complete = AsyncMock(return_value=mock_response)

        cfg = JudgeConfig(model="gpt-4o", temperature=0.5, max_tokens=512)
        await _judge_call(mock_provider, cfg, "test prompt", ApproachResult)

        call_kwargs = mock_provider.complete.call_args
        assert call_kwargs.kwargs["temperature"] == 0.5
        assert call_kwargs.kwargs["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_reasoning_effort_only_for_openai_reasoning(self):
        mock_provider = MagicMock(spec=BaseProvider)
        mock_provider.provider_name = "openai"
        mock_provider.is_reasoning_model = True
        mock_response = MagicMock()
        mock_response.content = '{"approach_correctness": 4, "reasoning": "Good approach"}'
        mock_provider.complete = AsyncMock(return_value=mock_response)

        cfg = JudgeConfig(model="gpt-5-mini", reasoning_effort="low")
        await _judge_call(mock_provider, cfg, "test prompt", ApproachResult)

        call_kwargs = mock_provider.complete.call_args
        assert call_kwargs.kwargs["reasoning_effort"] == "low"

    @pytest.mark.asyncio
    async def test_reasoning_effort_not_passed_to_non_openai(self):
        mock_provider = MagicMock(spec=BaseProvider)
        mock_provider.provider_name = "deepinfra"
        mock_response = MagicMock()
        mock_response.content = '{"approach_correctness": 4, "reasoning": "Good approach"}'
        mock_provider.complete = AsyncMock(return_value=mock_response)

        cfg = JudgeConfig(model="gpt-5-mini", reasoning_effort="low")
        await _judge_call(mock_provider, cfg, "test prompt", ApproachResult)

        call_kwargs = mock_provider.complete.call_args
        assert "reasoning_effort" not in call_kwargs.kwargs


class TestJudgeApproach:
    """Verify judge_approach forwards to the provider correctly."""

    @pytest.mark.asyncio
    async def test_returns_approach_result(self):
        mock_provider = MagicMock(spec=BaseProvider)
        mock_response = MagicMock()
        mock_response.content = '{"approach_correctness": 4, "reasoning": "Solid methodology"}'
        mock_provider.complete = AsyncMock(return_value=mock_response)

        cfg = JudgeConfig(model="gpt-4o", temperature=0.0)
        result = await judge_approach(
            mock_provider, "question", "steps", "trace", judge_config=cfg,
        )

        assert isinstance(result, ApproachResult)
        assert result.approach_correctness == 4
        mock_provider.complete.assert_awaited_once()
