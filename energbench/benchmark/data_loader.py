from __future__ import annotations

import csv
from pathlib import Path

from .models import Question


def load_questions(csv_path: Path) -> list[Question]:
    """Load benchmark questions from a CSV file.

    Args:
        csv_path: Path to the CSV file containing questions.

    Returns:
        List of Question objects.

    Raises:
        ValueError: If a row has an invalid or missing S/N value.
    """
    questions = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # row 1 is the header
            raw_id = row.get("S/N", "").strip()
            try:
                question_id = int(raw_id)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid S/N value '{raw_id}' at row {row_num} in {csv_path}. "
                    f"Expected an integer."
                )
            questions.append(
                Question(
                    id=question_id,
                    category=row.get("Category", ""),
                    question_type=row.get("Question type", ""),
                    difficulty=row.get("Difficulty level", ""),
                    question=row.get("Question", ""),
                )
            )

    return questions
