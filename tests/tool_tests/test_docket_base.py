import csv
import os

from energyevals.tools.dockets._base import DocketBaseTool


class TestDocketBaseToolUnit:
    """Unit tests for DocketBaseTool._save_csv()."""

    def test_save_csv_with_valid_rows(self, tmp_path):
        """Test CSV saving with valid rows produces a readable CSV."""
        csv_path = str(tmp_path / "test_output.csv")
        rows = [
            {"docket_number": "FC-1234", "title": "Rate Case", "date": "2025-01-15"},
            {"docket_number": "FC-5678", "title": "Tariff Filing", "date": "2025-02-01"},
        ]

        result = DocketBaseTool._save_csv(rows, csv_path)

        assert result == csv_path
        assert os.path.exists(csv_path)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            saved_rows = list(reader)

        assert len(saved_rows) == 2
        assert saved_rows[0]["docket_number"] == "FC-1234"
        assert saved_rows[1]["title"] == "Tariff Filing"

    def test_save_csv_with_empty_rows(self, tmp_path):
        """Test CSV saving with empty rows list creates file with headers only."""
        csv_path = str(tmp_path / "empty_output.csv")
        rows = []

        result = DocketBaseTool._save_csv(rows, csv_path)

        assert result == csv_path
        assert os.path.exists(csv_path)

    def test_save_csv_with_none_path(self):
        """Test _save_csv returns None when save_csv_path is None."""
        rows = [{"key": "value"}]
        result = DocketBaseTool._save_csv(rows, None)
        assert result is None

    def test_save_csv_with_empty_string_path(self):
        """Test _save_csv returns None when save_csv_path is empty string."""
        rows = [{"key": "value"}]
        result = DocketBaseTool._save_csv(rows, "")
        assert result is None

    def test_save_csv_error_handling(self, tmp_path):
        """Test _save_csv returns None on write error (e.g. bad directory)."""
        csv_path = "/nonexistent_dir/sub/test.csv"
        rows = [{"key": "value"}]

        result = DocketBaseTool._save_csv(rows, csv_path)

        assert result is None

    def test_save_csv_with_nested_data(self, tmp_path):
        """Test CSV saving handles rows with list/dict values gracefully."""
        csv_path = str(tmp_path / "nested.csv")
        rows = [
            {
                "id": "1",
                "tags": ["energy", "solar"],
                "meta": {"source": "FERC"},
            },
        ]

        result = DocketBaseTool._save_csv(rows, csv_path)

        assert result == csv_path
        assert os.path.exists(csv_path)
