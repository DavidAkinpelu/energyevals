import json
from unittest.mock import Mock

import pytest
import requests

from energbench.tools.dockets import DCDocketTool


class TestDCDocketToolUnit:
    """Unit tests for DCDocketTool."""

    def test_init(self):
        """Test initialization sets name and registers method."""
        tool = DCDocketTool()
        assert tool.name == "dc_dockets"
        assert "search_dc_dockets" in tool._methods

    def test_get_tools_definition(self):
        """Test tool definitions are properly structured."""
        tool = DCDocketTool()
        tools = tool.get_tools()

        assert len(tools) == 1
        assert tools[0].name == "search_dc_dockets"
        assert "dc" in tools[0].description.lower()
        assert "start_date" in tools[0].parameters["properties"]
        assert "end_date" in tools[0].parameters["properties"]
        assert "start_date" in tools[0].parameters["required"]
        assert "end_date" in tools[0].parameters["required"]

    def test_search_dc_success(self, mocker):
        """Test successful DC PSC search with mocked API."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "resultsSet": [
                {
                    "filingId": 1001,
                    "docketNumber": "FC-1234",
                    "companyOrIndividual": "Pepco Holdings",
                    "filingType": "Application",
                    "receivedDate": "2025-01-15",
                    "description": "<p>Rate case filing</p>",
                    "attachmentFileName": "filing_001.pdf",
                    "attachmentId": "ABC123",
                    "isConfidential": False,
                },
            ],
        }

        mocker.patch(
            "energbench.tools.dockets.dc_tool.requests.get",
            return_value=mock_response,
        )
        mocker.patch.object(DCDocketTool, "_save_csv", return_value=None)

        tool = DCDocketTool()
        result = tool.search_dc(
            start_date="01/01/2025",
            end_date="01/31/2025",
        )

        result_data = json.loads(result)
        assert "error" not in result_data
        assert result_data["num_results"] == 1
        assert result_data["results"][0]["docket_number"] == "FC-1234"
        assert result_data["results"][0]["description"] == "Rate case filing"
        assert result_data["results"][0]["download_url"] is not None

    def test_search_dc_empty_results(self, mocker):
        """Test DC PSC search with no results."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"resultsSet": []}

        mocker.patch(
            "energbench.tools.dockets.dc_tool.requests.get",
            return_value=mock_response,
        )
        mocker.patch.object(DCDocketTool, "_save_csv", return_value=None)

        tool = DCDocketTool()
        result = tool.search_dc(start_date="01/01/2025", end_date="01/31/2025")

        result_data = json.loads(result)
        assert "error" not in result_data
        assert result_data["num_results"] == 0
        assert result_data["results"] == []

    def test_search_dc_api_error(self, mocker):
        """Test handling of HTTP errors from DC PSC API."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )

        mocker.patch(
            "energbench.tools.dockets.dc_tool.requests.get",
            return_value=mock_response,
        )

        tool = DCDocketTool()
        result = tool.search_dc(start_date="01/01/2025", end_date="01/31/2025")

        result_data = json.loads(result)
        assert "error" in result_data
        assert "500" in result_data["error"]

    def test_search_dc_exception_handling(self, mocker):
        """Test generic exception handling in DC PSC search."""
        mocker.patch(
            "energbench.tools.dockets.dc_tool.requests.get",
            side_effect=Exception("Connection refused"),
        )

        tool = DCDocketTool()
        result = tool.search_dc(start_date="01/01/2025", end_date="01/31/2025")

        result_data = json.loads(result)
        assert "error" in result_data
        assert "Connection refused" in result_data["error"]

    def test_search_dc_confidential_filing(self, mocker):
        """Test that confidential filings have no download URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "resultsSet": [
                {
                    "filingId": 1002,
                    "docketNumber": "FC-5678",
                    "companyOrIndividual": "WGL",
                    "filingType": "Confidential",
                    "receivedDate": "2025-02-01",
                    "description": "Confidential filing",
                    "attachmentFileName": "secret.pdf",
                    "attachmentId": "XYZ",
                    "isConfidential": True,
                },
            ],
        }

        mocker.patch(
            "energbench.tools.dockets.dc_tool.requests.get",
            return_value=mock_response,
        )
        mocker.patch.object(DCDocketTool, "_save_csv", return_value=None)

        tool = DCDocketTool()
        result = tool.search_dc(start_date="01/01/2025", end_date="02/28/2025")

        result_data = json.loads(result)
        assert result_data["results"][0]["download_url"] is None
        assert result_data["results"][0]["is_confidential"] is True


@pytest.mark.integration
@pytest.mark.slow
class TestDCDocketToolIntegration:
    """Integration tests for DCDocketTool with real API."""

    def test_search_dc_real_api(self):
        """Test DC PSC search against the real API."""
        tool = DCDocketTool()
        try:
            result = tool.search_dc(
                start_date="01/01/2025",
                end_date="01/07/2025",
                records_to_show=5,
            )
        except Exception:
            pytest.skip("DC PSC API unavailable")

        result_data = json.loads(result)
        assert isinstance(result_data, dict)
        assert "results" in result_data or "error" in result_data
