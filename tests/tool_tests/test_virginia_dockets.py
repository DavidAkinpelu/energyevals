import json
from unittest.mock import Mock

import pytest
import requests

from energbench.tools.dockets import VirginiaDocketTool


class TestVirginiaDocketToolUnit:
    """Unit tests for VirginiaDocketTool."""

    def test_init(self):
        """Test initialization sets name and registers method."""
        tool = VirginiaDocketTool()
        assert tool.name == "virginia_dockets"
        assert "search_virginia_dockets" in tool._methods

    def test_get_tools_definition(self):
        """Test tool definitions are properly structured."""
        tool = VirginiaDocketTool()
        tools = tool.get_tools()

        assert len(tools) == 1
        assert tools[0].name == "search_virginia_dockets"
        assert "virginia" in tools[0].description.lower()
        assert "start_date" in tools[0].parameters["required"]
        assert "end_date" in tools[0].parameters["required"]

    def test_search_virginia_success(self, mocker):
        """Test successful Virginia SCC search with mocked JSON API."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = [
            {
                "CaseNumber": "PUR-2025-00001",
                "DocName": "Application for Rate Adjustment",
                "Month": 1,
                "Day": 15,
                "Year": 2025,
                "DocID": "DOC001",
                "FileName": "PUR-2025-00001/app.pdf",
            },
            {
                "CaseNumber": "PUR-2025-00002",
                "DocName": "Staff Report on Grid Modernization",
                "Month": 1,
                "Day": 20,
                "Year": 2025,
                "DocID": "DOC002",
                "FileName": "PUR-2025-00002/report.pdf",
            },
        ]

        mocker.patch(
            "energbench.tools.dockets.virginia_tool.requests.get",
            return_value=mock_response,
        )
        mocker.patch.object(VirginiaDocketTool, "_save_csv", return_value=None)

        tool = VirginiaDocketTool()
        result = tool.search_virginia(
            start_date="2025-01-01",
            end_date="2025-01-31",
        )

        result_data = json.loads(result)
        assert "error" not in result_data
        assert result_data["num_results"] == 2
        assert result_data["results"][0]["case_number"] == "PUR-2025-00001"
        assert result_data["results"][0]["filed_date"] == "2025-01-15"
        assert result_data["results"][0]["document_url"] is not None
        assert result_data["results"][1]["doc_name"] == "Staff Report on Grid Modernization"

    def test_search_virginia_with_docname_filter(self, mocker):
        """Test Virginia SCC search filters results by docname_contains."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = [
            {
                "CaseNumber": "PUR-2025-00001",
                "DocName": "Application for Rate Adjustment",
                "Month": 1,
                "Day": 15,
                "Year": 2025,
                "DocID": "DOC001",
                "FileName": "app.pdf",
            },
            {
                "CaseNumber": "PUR-2025-00002",
                "DocName": "Staff Report",
                "Month": 1,
                "Day": 20,
                "Year": 2025,
                "DocID": "DOC002",
                "FileName": "report.pdf",
            },
        ]

        mocker.patch(
            "energbench.tools.dockets.virginia_tool.requests.get",
            return_value=mock_response,
        )
        mocker.patch.object(VirginiaDocketTool, "_save_csv", return_value=None)

        tool = VirginiaDocketTool()
        result = tool.search_virginia(
            start_date="2025-01-01",
            end_date="2025-01-31",
            docname_contains="Rate",
        )

        result_data = json.loads(result)
        assert result_data["num_results"] == 1
        assert result_data["results"][0]["case_number"] == "PUR-2025-00001"

    def test_search_virginia_with_case_filter(self, mocker):
        """Test Virginia SCC search filters by case_contains."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = [
            {
                "CaseNumber": "PUR-2025-00001",
                "DocName": "Doc A",
                "Month": 1,
                "Day": 15,
                "Year": 2025,
                "DocID": "D1",
                "FileName": "a.pdf",
            },
            {
                "CaseNumber": "PUE-2025-00001",
                "DocName": "Doc B",
                "Month": 1,
                "Day": 20,
                "Year": 2025,
                "DocID": "D2",
                "FileName": "b.pdf",
            },
        ]

        mocker.patch(
            "energbench.tools.dockets.virginia_tool.requests.get",
            return_value=mock_response,
        )
        mocker.patch.object(VirginiaDocketTool, "_save_csv", return_value=None)

        tool = VirginiaDocketTool()
        result = tool.search_virginia(
            start_date="2025-01-01",
            end_date="2025-01-31",
            case_contains="PUE",
        )

        result_data = json.loads(result)
        assert result_data["num_results"] == 1
        assert result_data["results"][0]["case_number"] == "PUE-2025-00001"

    def test_search_virginia_empty_results(self, mocker):
        """Test Virginia SCC search with no results."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = []

        mocker.patch(
            "energbench.tools.dockets.virginia_tool.requests.get",
            return_value=mock_response,
        )
        mocker.patch.object(VirginiaDocketTool, "_save_csv", return_value=None)

        tool = VirginiaDocketTool()
        result = tool.search_virginia(
            start_date="2025-01-01",
            end_date="2025-01-31",
        )

        result_data = json.loads(result)
        assert result_data["num_results"] == 0
        assert result_data["results"] == []

    def test_search_virginia_api_error(self, mocker):
        """Test handling of HTTP errors from Virginia SCC API."""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "503 Service Unavailable"
        )

        mocker.patch(
            "energbench.tools.dockets.virginia_tool.requests.get",
            return_value=mock_response,
        )

        tool = VirginiaDocketTool()
        result = tool.search_virginia(
            start_date="2025-01-01",
            end_date="2025-01-31",
        )

        result_data = json.loads(result)
        assert "error" in result_data

    def test_search_virginia_exception_handling(self, mocker):
        """Test generic exception handling in Virginia SCC search."""
        mocker.patch(
            "energbench.tools.dockets.virginia_tool.requests.get",
            side_effect=Exception("Connection timed out"),
        )

        tool = VirginiaDocketTool()
        result = tool.search_virginia(
            start_date="2025-01-01",
            end_date="2025-01-31",
        )

        result_data = json.loads(result)
        assert "error" in result_data
        assert "Connection timed out" in result_data["error"]


@pytest.mark.integration
@pytest.mark.slow
class TestVirginiaDocketToolIntegration:
    """Integration tests for VirginiaDocketTool with real API."""

    def test_search_virginia_real_api(self):
        """Test Virginia SCC search against the real API."""
        tool = VirginiaDocketTool()
        try:
            result = tool.search_virginia(
                start_date="2025-01-01",
                end_date="2025-01-07",
                timeout=15,
            )
        except Exception:
            pytest.skip("Virginia SCC API unavailable")

        result_data = json.loads(result)
        assert isinstance(result_data, dict)
        assert "results" in result_data or "error" in result_data
