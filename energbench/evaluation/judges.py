from openai import OpenAI

from .models import (
    AccuracyResult,
    ApproachResult,
    AttributeAlignmentResult,
    SourceResult,
)

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
# Judge functions
# ---------------------------------------------------------------------------

def judge_approach(
    client: OpenAI,
    question: str,
    suggested_steps: str,
    agent_steps_trace: str,
    *,
    model: str = "gpt-4o",
) -> ApproachResult:
    """Evaluate the agent's approach to answering a question."""
    prompt = APPROACH_PROMPT.format(
        question=question,
        suggested_steps=suggested_steps,
        agent_steps_trace=agent_steps_trace,
    )
    resp = client.responses.parse(
        model=model,
        temperature=0,
        input=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        text_format=ApproachResult,
    )
    return resp.output_parsed


def judge_accuracy(
    client: OpenAI,
    question: str,
    expected_answer: str,
    agent_answer: str,
    *,
    abs_tol: float = 0.01,
    rel_tol: float = 0.5,
    model: str = "gpt-4o",
) -> AccuracyResult:
    """Evaluate factual and numerical accuracy of the agent's answer."""
    prompt = ACCURACY_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        agent_answer=agent_answer,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    resp = client.responses.parse(
        model=model,
        temperature=0,
        input=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        text_format=AccuracyResult,
    )
    return resp.output_parsed


def judge_sources(
    client: OpenAI,
    question: str,
    suggested_steps: str,
    agent_answer: str,
    *,
    model: str = "gpt-4o",
) -> SourceResult:
    """Evaluate source validity of the agent's answer."""
    prompt = SOURCE_PROMPT.format(
        question=question,
        suggested_steps=suggested_steps,
        agent_answer=agent_answer,
    )
    resp = client.responses.parse(
        model=model,
        temperature=0,
        input=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        text_format=SourceResult,
    )
    return resp.output_parsed


def judge_attributes(
    client: OpenAI,
    question: str,
    expected_answer: str,
    agent_answer: str,
    *,
    abs_tol: float = 0.01,
    rel_tol: float = 0.5,
    model: str = "gpt-4o",
) -> AttributeAlignmentResult:
    """Evaluate attribute alignment of the agent's answer."""
    prompt = ATTRIBUTE_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        agent_answer=agent_answer,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    resp = client.responses.parse(
        model=model,
        temperature=0,
        input=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        text_format=AttributeAlignmentResult,
    )
    return resp.output_parsed
