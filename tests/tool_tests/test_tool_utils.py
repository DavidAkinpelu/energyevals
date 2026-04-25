import json

import pandas as pd
import pytest

from energyevals.utils.csv_utils import (
    generate_timestamp,
    save_to_csv,
)
from energyevals.utils.formatting import (
    create_error_response,
    require_api_key,
)
from energyevals.utils.http import HTTPClient


class TestRequireApiKey:
    """Tests for require_api_key function."""

    def test_api_key_present(self, monkeypatch):
        """Test when API key is present."""
        monkeypatch.setenv("TEST_API_KEY", "secret_key_123")

        result = require_api_key("TEST_API_KEY", "TestTool")

        assert result == "secret_key_123"

    def test_api_key_missing(self, monkeypatch):
        """Test when API key is missing."""
        monkeypatch.delenv("TEST_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            require_api_key("TEST_API_KEY", "TestTool")

        assert "TEST_API_KEY" in str(exc_info.value)
        assert "TestTool" in str(exc_info.value)

    def test_api_key_empty_string(self, monkeypatch):
        """Test when API key is empty string."""
        monkeypatch.setenv("TEST_API_KEY", "")

        with pytest.raises(ValueError):
            require_api_key("TEST_API_KEY", "TestTool")


class TestGenerateTimestamp:
    """Tests for generate_timestamp function."""

    def test_timestamp_format(self):
        """Test timestamp format."""
        ts = generate_timestamp()

        assert len(ts) == 15
        assert ts[8] == "_"
        assert ts[:8].isdigit()
        assert ts[9:].isdigit()

    def test_timestamp_components(self):
        """Test timestamp components."""
        ts = generate_timestamp()

        year = int(ts[:4])
        month = int(ts[4:6])
        day = int(ts[6:8])
        hour = int(ts[9:11])
        minute = int(ts[11:13])
        second = int(ts[13:15])

        assert year >= 2020
        assert 1 <= month <= 12
        assert 1 <= day <= 31
        assert 0 <= hour <= 23
        assert 0 <= minute <= 59
        assert 0 <= second <= 59


class TestSaveToCsv:
    """Tests for save_to_csv function."""

    def test_save_simple_dataframe(self, tmp_path):
        """Test saving simple dataframe."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        path = save_to_csv(df, "test_data", tmp_path)

        assert path.exists()
        assert "test_data" in path.name
        assert path.suffix == ".csv"

    def test_saved_file_readable(self, tmp_path):
        """Test that saved file can be read back."""
        df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})

        path = save_to_csv(df, "data", tmp_path)
        loaded = pd.read_csv(path)

        pd.testing.assert_frame_equal(df, loaded)

    def test_filename_includes_timestamp(self, tmp_path):
        """Test that filename includes timestamp."""
        df = pd.DataFrame({"col": [1]})

        path = save_to_csv(df, "prefix", tmp_path)

        assert "prefix_" in path.name
        parts = path.stem.split("_")
        assert len(parts) >= 3  # prefix + date + time


class TestHTTPClient:
    """Tests for HTTPClient class."""

    def test_init_default(self):
        """Test initialization with defaults."""
        client = HTTPClient()

        assert client.session is not None
        assert client.auth_method == "header"
        assert client.auth_param_name == "x-api-key"

    def test_init_with_custom_auth(self):
        """Test initialization with custom auth settings."""
        client = HTTPClient(
            auth_method="param",
            auth_param_name="api_key",
            timeout=60,
            retries=5,
        )

        assert client.auth_method == "param"
        assert client.auth_param_name == "api_key"
        assert client.timeout == 60
        assert client.retries == 5


class TestCreateErrorResponse:
    """Tests for create_error_response function."""

    def test_simple_error(self):
        """Test creating simple error response."""
        result = create_error_response("Something went wrong", source="test_tool")
        data = json.loads(result)

        assert "error" in data
        assert data["error"] == "Something went wrong"
        assert data["source"] == "test_tool"

    def test_error_with_context(self):
        """Test creating error with additional context."""
        result = create_error_response(
            "API failed",
            source="api_tool",
            context={"status_code": 404, "endpoint": "/test"},
        )
        data = json.loads(result)

        assert data["error"] == "API failed"
        assert data["context"]["status_code"] == 404
        assert data["context"]["endpoint"] == "/test"

    def test_error_response_is_json_string(self):
        """Test that error response is valid JSON string."""
        result = create_error_response("Test error", source="test")

        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, dict)
