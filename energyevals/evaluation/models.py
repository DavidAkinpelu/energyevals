from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Judge output models (structured output schemas for OpenAI responses.parse)
# ---------------------------------------------------------------------------

class ApproachResult(BaseModel):
    approach_correctness: int = Field(ge=1, le=5)
    reasoning: str


class AccuracyResult(BaseModel):
    accuracy_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class SourceResult(BaseModel):
    source_validity: int = Field(ge=1, le=5)
    reasoning: str


class AttributeDetail(BaseModel):
    name: str
    expected: str
    found: bool
    agent_value: str | None = None


class AttributeAlignmentResult(BaseModel):
    total_attributes: int
    matched_attributes: int
    alignment_score: float = Field(ge=0.0, le=1.0)
    attribute_details: list[AttributeDetail]
    reasoning: str


class ExtractedAttribute(BaseModel):
    name: str
    expected: str


class ExtractedAttributesResult(BaseModel):
    attributes: list[ExtractedAttribute]
    reasoning: str


class QuestionAttributes(BaseModel):
    question_id: int
    question: str
    category: str = ""
    difficulty: str = ""
    attributes: list[ExtractedAttribute]


class AttributesFile(BaseModel):
    version: str = "1"
    generated_at: str
    judge_provider: str
    judge_model: str
    dataset_path: str
    questions: list[QuestionAttributes]


# ---------------------------------------------------------------------------
# Normalized score wrapper
# ---------------------------------------------------------------------------

class JudgeScore(BaseModel):
    """Judge score from a single judge call.

    Scale depends on the judge type:
    - approach: 1-5
    - sources: 1-5
    - accuracy/attributes: 0-1
    """
    score: float = Field(ge=0.0, le=5.0)
    reasoning: str
    judge_type: str


# ---------------------------------------------------------------------------
# Operational metrics from agent traces
# ---------------------------------------------------------------------------

class LatencyBreakdown(BaseModel):
    """Wall-clock and per-phase latency parsed from agent trace steps."""
    wall_clock_ms: float = 0.0
    llm_thinking_ms: float = 0.0
    tool_execution_ms: float = 0.0
    per_tool_ms: dict[str, float] = Field(default_factory=dict)


class CostEstimate(BaseModel):
    """Token-based cost estimate for an agent run."""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float = 0.0


class MetricScore(BaseModel):
    """Operational metrics extracted from a single agent trace."""
    tool_calls: int = 0
    iterations: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    cost: CostEstimate = Field(default_factory=CostEstimate)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

class GroundTruthAttribute(BaseModel):
    """Single expected attribute in the ground truth."""
    name: str
    value: str


class GroundTruth(BaseModel):
    """Ground truth for one benchmark question."""
    answer: str
    approach: str
    question_type: str
    category: str


# ---------------------------------------------------------------------------
# Benchmark result entry (loaded from trace JSON)
# ---------------------------------------------------------------------------

class BenchmarkResultEntry(BaseModel):
    """One question's agent trace data loaded from a benchmark trace file."""
    answer: str | None = None
    steps_trace: str = ""
    metrics: MetricScore = Field(default_factory=MetricScore)


# ---------------------------------------------------------------------------
# Trial-level evaluation
# ---------------------------------------------------------------------------

class TrialEval(BaseModel):
    """Results from judging one agent answer for one question (one trial)."""
    trial: int
    approach: JudgeScore
    accuracy: JudgeScore
    sources: JudgeScore
    raw_approach: ApproachResult | None = None
    raw_accuracy: AccuracyResult | None = None
    raw_attributes: AttributeAlignmentResult | None = None
    raw_sources: SourceResult | None = None
    metrics: MetricScore = Field(default_factory=MetricScore)


# ---------------------------------------------------------------------------
# Score statistics (aggregated across N trials)
# ---------------------------------------------------------------------------

class ScoreStatistics(BaseModel):
    """Aggregated stats for a score across N trials."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n: int


# ---------------------------------------------------------------------------
# Question-level evaluation (N trials aggregated)
# ---------------------------------------------------------------------------

class QuestionEval(BaseModel):
    """Full evaluation of one question across N trials."""
    question_id: int
    category: str
    difficulty: str
    accuracy_strategy: str
    trials: list[TrialEval]
    approach_stats: ScoreStatistics
    accuracy_stats: ScoreStatistics
    sources_stats: ScoreStatistics
    aggregated_metrics: MetricScore = Field(default_factory=MetricScore)


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------

class ModelComparison(BaseModel):
    """Result of a paired statistical test between two models."""
    model_a: str
    model_b: str
    dimension: str
    test_name: str
    p_value: float
    significant: bool
    effect_size: float
    direction: str


# ---------------------------------------------------------------------------
# Top-level evaluation report
# ---------------------------------------------------------------------------

class EvaluationReport(BaseModel):
    """Top-level report for one model's evaluation run."""
    model: str
    run_name: str
    num_trials: int
    questions: list[QuestionEval]
    aggregate_approach: ScoreStatistics
    aggregate_accuracy: ScoreStatistics
    aggregate_sources: ScoreStatistics
    aggregate_metrics: MetricScore = Field(default_factory=MetricScore)
    comparisons: list[ModelComparison] = Field(default_factory=list)
    config_snapshot: dict = Field(default_factory=dict)
