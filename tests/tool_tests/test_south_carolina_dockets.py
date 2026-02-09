import json
from unittest.mock import Mock

import pytest
import requests

from energbench.tools.dockets import SouthCarolinaDocketTool


class TestSouthCarolinaDocketToolUnit:
    """Unit tests for SouthCarolinaDocketTool."""

    def test_init(self):
        """Test initialization sets name and registers method."""
        tool = SouthCarolinaDocketTool()
        assert tool.name == "south_carolina_dockets"
        assert "search_south_carolina_dockets" in tool._methods

    def test_get_tools_definition(self):
        """Test tool definitions are properly structured."""
        tool = SouthCarolinaDocketTool()
        tools = tool.get_tools()

        assert len(tools) == 1
        assert tools[0].name == "search_south_carolina_dockets"
        assert "south carolina" in tools[0].description.lower()
        assert "start_date" in tools[0].parameters["required"]
        assert "end_date" in tools[0].parameters["required"]

    def test_search_south_carolina_success(self, mocker):
        """Test successful SC PSC search with mocked HTML response."""
        html = """
        <html><body>
            <table class="datatable-standard-savestate">
                <tbody>
                    <tr>
                        <td><a class="detailNumber" href="/Web/Dockets/Detail/2025-1-E">2025-1-E</a></td>
                        <td>
                            <span><strong>Application for Rate Increase</strong></span>
                            <div class="parties">
                                <a>Duke Energy Carolinas</a>
                                <a>SC ORS</a>
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <td><a class="detailNumber" href="/Web/Dockets/Detail/2025-2-G">2025-2-G</a></td>
                        <td>
                            <span><strong>Gas Pipeline Application</strong></span>
                            <div class="parties">
                                <a>Piedmont Natural Gas</a>
                            </div>
                        </td>
                    </tr>
                </tbody>
            </table>
        </body></html>
        """

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.text = html
        mock_response.url = "https://dms.psc.sc.gov/Web/Dockets/Search?StartDate=2025-01-01"

        mocker.patch(
            "energbench.tools.dockets.south_carolina_tool.requests.get",
            return_value=mock_response,
        )
        mocker.patch.object(SouthCarolinaDocketTool, "_save_csv", return_value=None)

        tool = SouthCarolinaDocketTool()
        result = tool.search_south_carolina(
            start_date="2025-01-01",
            end_date="2025-01-31",
        )

        result_data = json.loads(result)
        assert "error" not in result_data
        assert len(result_data["items"]) == 2
        assert result_data["items"][0]["docket_number"] == "2025-1-E"
        assert result_data["items"][0]["summary"] == "Application for Rate Increase"
        assert "Duke Energy Carolinas" in result_data["items"][0]["parties"]
        assert result_data["items"][1]["docket_number"] == "2025-2-G"

    def test_search_south_carolina_no_table(self, mocker):
        """Test SC PSC search when no results table is present."""
        html = "<html><body><p>No results found.</p></body></html>"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.text = html
        mock_response.url = "https://dms.psc.sc.gov/Web/Dockets/Search"

        mocker.patch(
            "energbench.tools.dockets.south_carolina_tool.requests.get",
            return_value=mock_response,
        )
        mocker.patch.object(SouthCarolinaDocketTool, "_save_csv", return_value=None)

        tool = SouthCarolinaDocketTool()
        result = tool.search_south_carolina(
            start_date="2025-01-01",
            end_date="2025-01-31",
        )

        result_data = json.loads(result)
        assert "error" not in result_data
        assert result_data["items"] == []

    def test_search_south_carolina_api_error(self, mocker):
        """Test handling of HTTP errors from SC PSC."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )

        mocker.patch(
            "energbench.tools.dockets.south_carolina_tool.requests.get",
            return_value=mock_response,
        )

        tool = SouthCarolinaDocketTool()
        result = tool.search_south_carolina(
            start_date="2025-01-01",
            end_date="2025-01-31",
        )

        result_data = json.loads(result)
        assert "error" in result_data

    def test_search_south_carolina_exception_handling(self, mocker):
        """Test generic exception handling in SC PSC search."""
        mocker.patch(
            "energbench.tools.dockets.south_carolina_tool.requests.get",
            side_effect=Exception("Timeout exceeded"),
        )

        tool = SouthCarolinaDocketTool()
        result = tool.search_south_carolina(
            start_date="2025-01-01",
            end_date="2025-01-31",
        )

        result_data = json.loads(result)
        assert "error" in result_data
        assert "Timeout exceeded" in result_data["error"]


@pytest.mark.integration
@pytest.mark.slow
class TestSouthCarolinaDocketToolIntegration:
    """Integration tests for SouthCarolinaDocketTool with real API."""

    def test_search_south_carolina_real_api(self):
        """Test SC PSC search against the real website."""
        tool = SouthCarolinaDocketTool()
        try:
            result = tool.search_south_carolina(
                start_date="2025-01-01",
                end_date="2025-01-07",
                timeout=15,
            )
        except Exception:
            pytest.skip("SC PSC website unavailable")

        result_data = json.loads(result)
        assert isinstance(result_data, dict)
        assert "items" in result_data #or "error" in result_data
