from dataclasses import dataclass, field
from typing import Any


@dataclass
class Question:
    """A benchmark question."""

    id: int
    category: str
    question_type: str
    difficulty: str
    question: str


@dataclass
class BenchmarkResult:
    """Result of running a benchmark question."""

    question: Question
    provider: str
    model: str
    success: bool
    answer: str | None
    error: str | None
    metrics: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
