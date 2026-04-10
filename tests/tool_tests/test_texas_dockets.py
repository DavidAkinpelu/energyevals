import json
from unittest.mock import Mock

import pytest
import requests

from energyevals.tools.dockets import TexasDocketTool


class TestTexasDocketToolUnit:
    """Unit tests for TexasDocketTool."""

    def test_init(self):
        """Test initialization sets name and registers method."""
        tool = TexasDocketTool()
        assert tool.name == "texas_dockets"
        assert "search_texas_dockets" in tool._methods

    def test_get_tools_definition(self):
        """Test tool definitions are properly structured."""
        tool = TexasDocketTool()
        tools = tool.get_tools()

        assert len(tools) == 1
        assert tools[0].name == "search_texas_dockets"
        assert "texas" in tools[0].description.lower()
        assert "date_from" in tools[0].parameters["required"]
        assert "date_to" in tools[0].parameters["required"]

    def test_search_texas_success(self, mocker):
        """Test successful Texas PUC docket search with mocked HTML response."""
        html = """
        <html><body>
            <table>
                <tr><th>Control</th><th>Filings</th><th>Utility</th><th>Description</th></tr>
                <tr>
                    <td><a href="/search/dockets/55001">55001</a></td>
                    <td>12</td>
                    <td>Oncor Electric</td>
                    <td>Application for Rate Increase</td>
                </tr>
                <tr>
                    <td><a href="/search/dockets/55002">55002</a></td>
                    <td>5</td>
                    <td>CenterPoint Energy</td>
                    <td>Transmission Cost of Service</td>
                </tr>
            </table>
        </body></html>
        """

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.text = html

        mocker.patch(
            "energyevals.tools.dockets.texas_tool.requests.get",
            return_value=mock_response,
        )
        mocker.patch.object(TexasDocketTool, "_save_csv", return_value=None)

        tool = TexasDocketTool()
        result = tool.search_texas(
            date_from="01/01/2025",
            date_to="01/31/2025",
        )

        result_data = json.loads(result)
        assert "error" not in result_data
        assert result_data["source"] == "Texas PUC"
        assert result_data["num_results"] == 2
        assert result_data["results"][0]["control_number"] == "55001"
        assert result_data["results"][0]["utility"] == "Oncor Electric"
        assert result_data["results"][1]["description"] == "Transmission Cost of Service"

    def test_search_texas_missing_dates(self, mocker):
        """Test Texas PUC search raises error when dates are empty."""
        mocker.patch.object(TexasDocketTool, "_save_csv", return_value=None)

        tool = TexasDocketTool()
        result = tool.search_texas(date_from="", date_to="")

        result_data = json.loads(result)
        assert "error" in result_data

    def test_search_texas_filing_search(self, mocker):
        """Test Texas PUC filing search (with control_number) uses filings URL."""
        html = "<html><body><table><tr><th>A</th></tr></table></body></html>"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.text = html

        mock_get = mocker.patch(
            "energyevals.tools.dockets.texas_tool.requests.get",
            return_value=mock_response,
        )
        mocker.patch.object(TexasDocketTool, "_save_csv", return_value=None)

        tool = TexasDocketTool()
        tool.search_texas(
            date_from="01/01/2025",
            date_to="01/31/2025",
            control_number="55001",
        )

        call_url = mock_get.call_args[0][0]
        assert "/search/filings/" in call_url

    def test_search_texas_api_error(self, mocker):
        """Test handling of HTTP errors from Texas PUC."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )

        mocker.patch(
            "energyevals.tools.dockets.texas_tool.requests.get",
            return_value=mock_response,
        )

        tool = TexasDocketTool()
        result = tool.search_texas(
            date_from="01/01/2025",
            date_to="01/31/2025",
        )

        result_data = json.loads(result)
        assert "error" in result_data

    def test_search_texas_exception_handling(self, mocker):
        """Test generic exception handling in Texas PUC search."""
        mocker.patch(
            "energyevals.tools.dockets.texas_tool.requests.get",
            side_effect=Exception("Read timed out"),
        )

        tool = TexasDocketTool()
        result = tool.search_texas(
            date_from="01/01/2025",
            date_to="01/31/2025",
        )

        result_data = json.loads(result)
        assert "error" in result_data
        assert "Read timed out" in result_data["error"]


@pytest.mark.integration
@pytest.mark.slow
class TestTexasDocketToolIntegration:
    """Integration tests for TexasDocketTool with real API."""

    def test_search_texas_real_api(self):
        """Test Texas PUC search against the real website."""
        tool = TexasDocketTool()
        result = tool.search_texas(
            date_from="01/01/2025",
            date_to="01/07/2025",
            timeout=15,
        )
        # try:
        #     result = tool.search_texas(
        #         date_from="01/01/2025",
        #         date_to="01/07/2025",
        #         timeout=15,
        #     )
        # except Exception:
        #     pytest.skip("Texas PUC website unavailable")

        result_data = json.loads(result)
        assert isinstance(result_data, dict)
        assert "results" in result_data #or "error" in result_data
