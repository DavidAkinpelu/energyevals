import json
from unittest.mock import Mock

import pytest
import requests

from energyevals.tools.dockets import FERCDocketTool


class TestFERCDocketToolUnit:
    """Unit tests for FERCDocketTool."""

    def test_init(self):
        """Test initialization sets name and registers method."""
        tool = FERCDocketTool()
        assert tool.name == "ferc_dockets"
        assert "search_ferc_dockets" in tool._methods

    def test_get_tools_definition(self):
        """Test tool definitions are properly structured."""
        tool = FERCDocketTool()
        tools = tool.get_tools()

        assert len(tools) == 1
        assert tools[0].name == "search_ferc_dockets"
        assert "ferc" in tools[0].description.lower()
        assert "start_date" in tools[0].parameters["properties"]
        assert "keyword" in tools[0].parameters["properties"]
        assert "start_date" in tools[0].parameters["required"]
        assert "end_date" in tools[0].parameters["required"]

    def test_search_ferc_success(self, mocker):
        """Test successful FERC eLibrary search with mocked API."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "searchHits": [
                {
                    "description": "Application for Market-Based Rate Authority",
                    "filedDate": "2025-01-10",
                    "docketNumbers": ["ER25-1234"],
                    "category": "Electric",
                    "libraries": ["eLibrary"],
                    "accessionNumber": "20250110-5001",
                    "affiliations": [
                        {"afType": "Applicant", "affiliation": "Test Energy Corp"},
                    ],
                    "transmittals": [
                        {
                            "fileType": "PDF",
                            "fileName": "application.pdf",
                            "fileDesc": "Main filing",
                            "fileSize": 102400,
                            "fileId": "FILE001",
                        },
                        {
                            "fileType": "DOC",
                            "fileName": "cover_letter.doc",
                            "fileDesc": "Cover",
                            "fileSize": 2048,
                            "fileId": "FILE002",
                        },
                    ],
                },
            ],
        }

        mocker.patch(
            "energyevals.tools.dockets.ferc_tool.requests.post",
            return_value=mock_response,
        )
        mocker.patch.object(FERCDocketTool, "_save_csv", return_value=None)

        tool = FERCDocketTool()
        result = tool.search_ferc(
            start_date="2025-01-01",
            end_date="2025-01-31",
            keyword="market-based rate",
        )

        result_data = json.loads(result)
        assert "error" not in result_data
        assert result_data["source"] == "FERC"
        assert result_data["num_results"] == 1
        assert result_data["keyword"] == "market-based rate"
        assert result_data["results"][0]["docket_numbers"] == ["ER25-1234"]
        # Only PDF files should be included
        assert len(result_data["results"][0]["pdf_files"]) == 1
        assert result_data["results"][0]["pdf_files"][0]["file_name"] == "application.pdf"
        # Affiliations are formatted as strings
        assert "Applicant: Test Energy Corp" in result_data["results"][0]["affiliations"]

    def test_search_ferc_empty_results(self, mocker):
        """Test FERC search with no hits."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"searchHits": []}

        mocker.patch(
            "energyevals.tools.dockets.ferc_tool.requests.post",
            return_value=mock_response,
        )
        mocker.patch.object(FERCDocketTool, "_save_csv", return_value=None)

        tool = FERCDocketTool()
        result = tool.search_ferc(
            start_date="2025-01-01",
            end_date="2025-01-31",
            keyword="nonexistent topic",
        )

        result_data = json.loads(result)
        assert result_data["num_results"] == 0
        assert result_data["results"] == []

    def test_search_ferc_api_error(self, mocker):
        """Test handling of HTTP errors from FERC API."""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "503 Service Unavailable"
        )

        mocker.patch(
            "energyevals.tools.dockets.ferc_tool.requests.post",
            return_value=mock_response,
        )

        tool = FERCDocketTool()
        result = tool.search_ferc(
            start_date="2025-01-01",
            end_date="2025-01-31",
            keyword="test",
        )

        result_data = json.loads(result)
        assert "error" in result_data
        assert result_data["source"] == "FERC"

    def test_search_ferc_exception_handling(self, mocker):
        """Test generic exception handling in FERC search."""
        mocker.patch(
            "energyevals.tools.dockets.ferc_tool.requests.post",
            side_effect=Exception("Network timeout"),
        )

        tool = FERCDocketTool()
        result = tool.search_ferc(
            start_date="2025-01-01",
            end_date="2025-01-31",
            keyword="test",
        )

        result_data = json.loads(result)
        assert "error" in result_data
        assert "Network timeout" in result_data["error"]


@pytest.mark.integration
@pytest.mark.slow
class TestFERCDocketToolIntegration:
    """Integration tests for FERCDocketTool with real API."""

    def test_search_ferc_real_api(self):
        """Test FERC search against the real API."""
        tool = FERCDocketTool()
        try:
            result = tool.search_ferc(
                start_date="2025-01-01",
                end_date="2025-01-07",
                keyword="solar",
                results_per_page=5,
            )
        except Exception:
            pytest.skip("FERC API unavailable")

        result_data = json.loads(result)
        assert isinstance(result_data, dict)
        assert "results" in result_data or "error" in result_data
