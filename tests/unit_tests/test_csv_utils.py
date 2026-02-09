from pathlib import Path

import pandas as pd

from energbench.utils.csv_utils import (
    csv_string_to_dataframe,
    dataframe_to_csv_string,
    generate_timestamp,
    process_large_dataframe_result,
    save_dataframe_to_csv,
)


class TestGenerateTimestamp:
    """Tests for generate_timestamp function."""

    def test_timestamp_format(self):
        """Test that timestamp has correct format."""
        ts = generate_timestamp()

        # Should be YYYYMMDD_HHMMSS format
        assert len(ts) == 15
        assert ts[8] == "_"
        assert ts[:8].isdigit()  # YYYYMMDD
        assert ts[9:].isdigit()  # HHMMSS

    def test_timestamp_uniqueness(self):
        """Test that consecutive timestamps are different (or very close)."""
        ts1 = generate_timestamp()
        ts2 = generate_timestamp()

        # Should be same or differ by at most 1 second
        assert ts1 <= ts2


class TestSaveDataframeToCsv:
    """Tests for save_dataframe_to_csv function."""

    def test_save_simple_dataframe(self, tmp_path):
        """Test saving a simple dataframe."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        path = save_dataframe_to_csv(df, "test", tmp_path)

        assert path.exists()
        assert path.parent == tmp_path
        assert "test_" in path.name
        assert path.suffix == ".csv"

    def test_saved_csv_content(self, tmp_path):
        """Test that saved CSV has correct content."""
        df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})

        path = save_dataframe_to_csv(df, "data", tmp_path)

        loaded_df = pd.read_csv(path)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_filename_with_timestamp(self, tmp_path):
        """Test that filename includes timestamp."""
        df = pd.DataFrame({"col": [1]})

        path = save_dataframe_to_csv(df, "prefix", tmp_path)

        # Should have format: prefix_YYYYMMDD_HHMMSS.csv
        parts = path.stem.split("_")
        assert parts[0] == "prefix"
        assert len(parts) >= 3  # prefix, date, time


class TestProcessLargeDataframeResult:
    """Tests for process_large_dataframe_result function."""

    def test_small_dataframe_inline(self, tmp_path):
        """Test that small dataframes are returned inline."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        result = process_large_dataframe_result(df, "test", tmp_path, csv_threshold=100)

        assert isinstance(result, dict)
        assert result["row_count"] == 3
        assert "data" in result
        assert "csv_saved" not in result  # Small results don't have csv_saved

    def test_large_dataframe_saved_to_csv(self, tmp_path):
        """Test that large dataframes are saved to CSV."""
        df = pd.DataFrame({"x": range(200), "y": range(200, 400)})

        result = process_large_dataframe_result(df, "large", tmp_path, csv_threshold=100)

        assert result["csv_saved"] is True
        assert "csv_path" in result
        assert result["row_count"] == 200
        assert Path(result["csv_path"]).exists()

    def test_threshold_boundary(self, tmp_path):
        """Test behavior at threshold boundary."""
        df = pd.DataFrame({"a": range(50)})
        # len(df) == 50, threshold == 50 → 50 <= 50 is True → inline (not CSV)
        result = process_large_dataframe_result(df, "test", tmp_path, csv_threshold=50)

        assert "data" in result
        assert result["row_count"] == 50

    def test_empty_dataframe(self, tmp_path):
        """Test handling of empty dataframe."""
        df = pd.DataFrame()

        result = process_large_dataframe_result(df, "empty", tmp_path, csv_threshold=10)

        assert result["row_count"] == 0


class TestCsvStringToDataframe:
    """Tests for csv_string_to_dataframe function."""

    def test_parse_simple_csv(self):
        """Test parsing a simple CSV string."""
        csv_str = "a,b\n1,2\n3,4\n"

        df = csv_string_to_dataframe(csv_str)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2
        assert df["a"].tolist() == [1, 3]
        assert df["b"].tolist() == [2, 4]

    def test_parse_csv_with_quotes(self):
        """Test parsing CSV with quoted values."""
        csv_str = 'name,value\n"John Doe",100\n"Jane Smith",200\n'

        df = csv_string_to_dataframe(csv_str)

        assert df["name"].tolist() == ["John Doe", "Jane Smith"]
        assert df["value"].tolist() == [100, 200]

    def test_parse_empty_csv(self):
        """Test parsing empty CSV string."""
        csv_str = "col1,col2\n"

        df = csv_string_to_dataframe(csv_str)

        assert len(df) == 0
        assert list(df.columns) == ["col1", "col2"]


class TestDataframeToCsvString:
    """Tests for dataframe_to_csv_string function."""

    def test_convert_simple_dataframe(self):
        """Test converting simple dataframe to CSV string."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

        csv_str = dataframe_to_csv_string(df)

        assert "x,y" in csv_str
        assert "1,3" in csv_str or "1" in csv_str
        assert isinstance(csv_str, str)

    def test_roundtrip_conversion(self):
        """Test converting to CSV and back preserves data."""
        original = pd.DataFrame({"a": [10, 20, 30], "b": ["x", "y", "z"]})

        csv_str = dataframe_to_csv_string(original)
        restored = csv_string_to_dataframe(csv_str)

        pd.testing.assert_frame_equal(original, restored)

    def test_convert_empty_dataframe(self):
        """Test converting empty dataframe."""
        df = pd.DataFrame(columns=["col1", "col2"])

        csv_str = dataframe_to_csv_string(df)

        assert "col1,col2" in csv_str
        assert isinstance(csv_str, str)
