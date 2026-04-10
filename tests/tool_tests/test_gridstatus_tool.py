import json
import os
from unittest.mock import Mock

import pytest
import requests

from energyevals.tools.gridstatus_tool import GridStatusAPITool


class TestGridStatusAPIToolUnit:
    """Unit tests for GridStatusAPITool."""

    def test_init_with_api_key(self, monkeypatch):
        """Test initialization with API key."""
        monkeypatch.setenv("GRIDSTATUS_API_KEY", "test_key")
        tool = GridStatusAPITool()
        assert tool.name == "gridstatus_api_tool"
        assert tool.api_key == "test_key"

    def test_init_without_api_key(self, monkeypatch):
        """Test initialization without API key."""
        monkeypatch.delenv("GRIDSTATUS_API_KEY", raising=False)
        tool = GridStatusAPITool()
        assert tool.api_key is None

    def test_get_tools_definition(self):
        """Test tool definitions are properly structured."""
        tool = GridStatusAPITool(api_key="test_key")
        tools = tool.get_tools()

        assert len(tools) == 3
        tool_names = {t.name for t in tools}
        assert "list_gridstatus_datasets" in tool_names
        assert "inspect_gridstatus_dataset" in tool_names
        assert "query_gridstatus_dataset" in tool_names

    def test_set_api_key(self, monkeypatch):
        """Test setting API key after initialization."""
        monkeypatch.delenv("GRIDSTATUS_API_KEY", raising=False)
        tool = GridStatusAPITool()
        assert tool.api_key is None

        tool.set_api_key("new_key")
        assert tool.api_key == "new_key"

    def test_list_datasets_success(self, mocker):
        """Test listing datasets with mocked API."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "ercot_lmp", "name": "ERCOT LMP", "description": "ERCOT prices"},
                {"id": "caiso_lmp", "name": "CAISO LMP", "description": "CAISO prices"},
            ]
        }

        mocker.patch("energyevals.tools.gridstatus_tool.requests.get", return_value=mock_response)

        tool = GridStatusAPITool(api_key="test_key")
        result = tool.list_gridstatus_datasets()

        result_data = json.loads(result)
        assert isinstance(result_data, list)
        assert len(result_data) == 2
        assert result_data[0]["id"] == "ercot_lmp"

    def test_inspect_dataset_success(self, mocker):
        """Test dataset inspection with mocked API."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "id": "ercot_lmp",
            "schema": {"columns": ["time", "price", "location"]},
            "row_count": 1000,
        }

        mocker.patch("energyevals.tools.gridstatus_tool.requests.get", return_value=mock_response)

        tool = GridStatusAPITool(api_key="test_key")
        result = tool.inspect_gridstatus_dataset(dataset_id="ercot_lmp")

        result_data = json.loads(result)
        assert result_data["id"] == "ercot_lmp"
        assert "schema" in result_data

    def test_query_dataset_success(self, mocker):
        """Test dataset query with mocked API."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "data": [
                {"time": "2024-01-01T00:00:00Z", "price": 25.5, "location": "HB_NORTH"},
                {"time": "2024-01-01T01:00:00Z", "price": 27.3, "location": "HB_NORTH"},
            ],
        }

        mocker.patch("energyevals.tools.gridstatus_tool.requests.get", return_value=mock_response)

        tool = GridStatusAPITool(api_key="test_key")
        result = tool.query_gridstatus_dataset(
            dataset_id="ercot_lmp",
            limit=10,
        )

        result_data = json.loads(result)
        assert isinstance(result_data, dict)
        assert "preview" in result_data or "data" in result_data or "error" not in result_data

    def test_query_dataset_api_error(self, mocker):
        """Test handling of API errors."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_response.text = "Dataset not found"

        mocker.patch("energyevals.tools.gridstatus_tool.requests.get", return_value=mock_response)

        tool = GridStatusAPITool(api_key="test_key")
        result = tool.query_gridstatus_dataset(dataset_id="invalid")

        result_data = json.loads(result)
        assert "error" in result_data


@pytest.mark.integration
@pytest.mark.requires_api_key
class TestGridStatusAPIToolIntegration:
    """Integration tests with real GridStatus API."""

    @pytest.mark.asyncio
    async def test_list_datasets_real_api(self):
        """Test listing datasets with real API."""
        if not os.getenv("GRIDSTATUS_API_KEY"):
            pytest.skip("GRIDSTATUS_API_KEY not set")

        tool = GridStatusAPITool()
        result = tool.list_gridstatus_datasets()

        result_data = json.loads(result)
        assert isinstance(result_data, list) or "error" in result_data

    @pytest.mark.asyncio
    async def test_query_dataset_real_api(self):
        """Test querying dataset with real API."""
        if not os.getenv("GRIDSTATUS_API_KEY"):
            pytest.skip("GRIDSTATUS_API_KEY not set")

        tool = GridStatusAPITool()

        list_result = tool.list_gridstatus_datasets()
        list_data = json.loads(list_result)

        if isinstance(list_data, list) and list_data:
            dataset_id = list_data[0].get("id")

            if dataset_id:
                query_result = tool.query_gridstatus_dataset(
                    dataset_id=dataset_id,
                    limit=1,
                )

                query_data = json.loads(query_result)
                assert isinstance(query_data, dict)
