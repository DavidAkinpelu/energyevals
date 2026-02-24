import json
import logging
from typing import Any

from pydantic import BaseModel

from energbench.agent.providers import BaseProvider, Message, get_provider
from energbench.agent.providers.openai_provider import is_reasoning_model

from .config import JudgeConfig
from .models import (
    AccuracyResult,
    ApproachResult,
    AttributeAlignmentResult,
    SourceResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (shared across all judges)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluator of answers relating to energy markets analysis
Follow expert industry standards.
Your output MUST exactly match the provided output schema.
Do not add extra fields or surrounding text.
"""

# ---------------------------------------------------------------------------
# Per-judge prompt templates
# ---------------------------------------------------------------------------

APPROACH_PROMPT = """\
You are evaluating the approach correctness of how an AI agent obtained answers to an energy-market related question
and not the correctness of the answer itself.

In addition to question, you also have a summary of the suggested approach provided by an expert and a trace \
of the steps the agent took to answer the question which you can use to infer the agent's approach to answering \
the question

Question:
{question}

Suggested Approach (Ground Truth):
{suggested_steps}

Agent's Steps:
{agent_steps_trace}

Evaluate:
- Correct problem framing
- Appropriate data sources (ISO postings, tariffs, settlement data, APIs)
- Logical analytical steps
- Correct tool usage (if applicable)

Rating scale:
5=expert-like, 4=minor issues, 3=notable gaps, 2=major flaws, 1=wrong approach
"""

ACCURACY_PROMPT = """\
You are evaluating the factual and numerical accuracy of an AI agent's answer to a question relating to energy markets \
analysis.

Question:
{question}

Expected Answer (Ground Truth):
{expected_answer}

Agent's Answer:
{agent_answer}

Evaluate:
- Numerical correctness (values, sign, magnitude, units, time basis)
- Factual alignment (market/ISO, node/zone, product, settlement type etc.)
- Completeness of key facts

Tolerance:
- Allow <= {abs_tol} absolute error OR <= {rel_tol}% relative error unless exactness is required.
"""

SOURCE_PROMPT = """\
You are evaluating the source validity of an AI agent's answer to a question relating to energy markets analysis.

You can extract or infer relevant sources from the question iteself or from the suggested approach ground truth

Question:
{question}

Suggested Steps:
{suggested_steps}

Agent's Answer:
{agent_answer}

Evaluate:
- Authority of sources
- Alignment with expected sources
- Appropriateness for the claim
- Missing citations when required
"""

ATTRIBUTE_PROMPT = """\
You are evaluating attribute alignment of an AI agent's answer against expected attributes.

You should extract attributes from the expected answer. There should be no more than 5 attributes.

Question:
{question}

Expected Answer:
{expected_answer}

Agent's Answer:
{agent_answer}

For each expected attribute, decide whether the agent answer contains the correct value
or a reasonable equivalent, respecting units and time basis.

Tolerance:
- For numeric attributes, allow <= {abs_tol} absolute error OR <= {rel_tol}% relative error \
  unless exactness is required.

"""


# ---------------------------------------------------------------------------
# Provider factory & helpers
# ---------------------------------------------------------------------------

def create_judge_provider(config: JudgeConfig) -> BaseProvider:
    """Create a provider instance from the judge configuration.

    Uses the same ``get_provider`` factory that the benchmark agent uses, so
    all supported providers (openai, anthropic, google, deepinfra) work
    out of the box.
    """
    kwargs: dict[str, Any] = {}
    if config.provider == "openai" and config.reasoning_effort:
        if is_reasoning_model(config.model):
            kwargs["reasoning_effort"] = config.reasoning_effort
    return get_provider(config.provider, model=config.model, **kwargs)


def _build_schema_instruction(result_type: type[BaseModel]) -> str:
    """Build a JSON-schema instruction string for the prompt."""
    schema = result_type.model_json_schema()
    return (
        "You MUST respond with ONLY a valid JSON object matching this schema "
        "(no markdown fences, no extra text):\n"
        f"{json.dumps(schema, indent=2)}"
    )


async def _judge_call(
    provider: BaseProvider,
    config: JudgeConfig,
    prompt: str,
    result_type: type[BaseModel],
) -> Any:
    """Run a judge call through the provider and parse the structured result."""
    schema_instruction = _build_schema_instruction(result_type)
    system_content = f"{JUDGE_SYSTEM_PROMPT}\n\n{schema_instruction}"

    messages = [
        Message(role="system", content=system_content),
        Message(role="user", content=prompt),
    ]

    kwargs: dict[str, Any] = {}
    if (
        config.reasoning_effort
        and provider.provider_name == "openai"
        and getattr(provider, "is_reasoning_model", False)
    ):
        kwargs["reasoning_effort"] = config.reasoning_effort

    response = await provider.complete(
        messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        **kwargs,
    )

    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0].strip()

    return result_type.model_validate_json(text)


# ---------------------------------------------------------------------------
# Judge functions
# ---------------------------------------------------------------------------

async def judge_approach(
    provider: BaseProvider,
    question: str,
    suggested_steps: str,
    agent_steps_trace: str,
    *,
    judge_config: JudgeConfig,
) -> ApproachResult:
    """Evaluate the agent's approach to answering a question."""
    prompt = APPROACH_PROMPT.format(
        question=question,
        suggested_steps=suggested_steps,
        agent_steps_trace=agent_steps_trace,
    )
    return await _judge_call(provider, judge_config, prompt, ApproachResult)


async def judge_accuracy(
    provider: BaseProvider,
    question: str,
    expected_answer: str,
    agent_answer: str,
    *,
    abs_tol: float = 0.01,
    rel_tol: float = 0.5,
    judge_config: JudgeConfig,
) -> AccuracyResult:
    """Evaluate factual and numerical accuracy of the agent's answer."""
    prompt = ACCURACY_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        agent_answer=agent_answer,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    return await _judge_call(provider, judge_config, prompt, AccuracyResult)


async def judge_sources(
    provider: BaseProvider,
    question: str,
    suggested_steps: str,
    agent_answer: str,
    *,
    judge_config: JudgeConfig,
) -> SourceResult:
    """Evaluate source validity of the agent's answer."""
    prompt = SOURCE_PROMPT.format(
        question=question,
        suggested_steps=suggested_steps,
        agent_answer=agent_answer,
    )
    return await _judge_call(provider, judge_config, prompt, SourceResult)


async def judge_attributes(
    provider: BaseProvider,
    question: str,
    expected_answer: str,
    agent_answer: str,
    *,
    abs_tol: float = 0.01,
    rel_tol: float = 0.5,
    judge_config: JudgeConfig,
) -> AttributeAlignmentResult:
    """Evaluate attribute alignment of the agent's answer."""
    prompt = ATTRIBUTE_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        agent_answer=agent_answer,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    return await _judge_call(provider, judge_config, prompt, AttributeAlignmentResult)
