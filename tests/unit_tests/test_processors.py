import json
import tempfile
from pathlib import Path

from energyevals.agent.processors import CSVProcessor, ImageProcessor, ResultProcessor


class TestCSVProcessor:
    """Tests for CSVProcessor."""

    def test_process_non_json_result(self):
        """Test processing non-JSON result."""
        processor = CSVProcessor()
        result, csv_path = processor.process("test_tool", "plain text")
        assert result == "plain text"
        assert csv_path is None

    def test_process_result_without_rows(self):
        """Test processing JSON without rows."""
        processor = CSVProcessor()
        data = {"status": "success", "value": 42}
        result, csv_path = processor.process("test_tool", json.dumps(data))
        assert result == json.dumps(data)
        assert csv_path is None

    def test_process_result_below_threshold(self):
        """Test processing result with rows below threshold."""
        processor = CSVProcessor(threshold=10)
        data = {
            "rows": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            "columns": ["a", "b"]
        }
        result, csv_path = processor.process("test_tool", json.dumps(data))
        assert result == json.dumps(data)
        assert csv_path is None

    def test_process_result_above_threshold(self):
        """Test processing result with rows above threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = CSVProcessor(threshold=2, output_dir=tmpdir)
            data = {
                "rows": [{"a": i, "b": i*2} for i in range(5)],
                "columns": ["a", "b"]
            }
            result, csv_path = processor.process("test_tool", json.dumps(data))

            # Should return a reference to CSV
            result_data = json.loads(result)
            assert result_data["status"] == "success"
            assert result_data["row_count"] == 5
            assert "csv_file" in result_data
            assert csv_path is not None
            assert Path(csv_path).exists()

            # Check CSV file was created
            with open(csv_path) as f:
                content = f.read()
                assert "a,b" in content  # Header


class TestImageProcessor:
    """Tests for ImageProcessor."""

    def test_extract_images_empty_result(self):
        """Test extracting images from empty result."""
        processor = ImageProcessor()
        images = processor.extract_images("{}")
        assert images == []

    def test_extract_images_no_images(self):
        """Test extracting images from result without images."""
        processor = ImageProcessor()
        data = {"status": "success", "data": [1, 2, 3]}
        images = processor.extract_images(json.dumps(data))
        assert images == []


class TestResultProcessor:
    """Tests for ResultProcessor."""

    def test_process_result_delegates_to_csv(self):
        """Test that process_result delegates to CSVProcessor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ResultProcessor(csv_threshold=2, csv_output_dir=tmpdir)
            data = {
                "rows": [{"a": i} for i in range(5)],
                "columns": ["a"]
            }
            result, csv_path = processor.process_result("test_tool", json.dumps(data))

            # Should have saved to CSV
            assert csv_path is not None
            assert Path(csv_path).exists()

    def test_extract_images_delegates_to_image(self):
        """Test that extract_images delegates to ImageProcessor."""
        processor = ResultProcessor()
        images = processor.extract_images("{}")
        assert images == []
