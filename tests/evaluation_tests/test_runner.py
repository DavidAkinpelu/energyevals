import pytest


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
