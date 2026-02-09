import json
from unittest.mock import MagicMock, Mock

import pytest
import requests

from energbench.tools.dockets import NorthCarolinaDocketTool


class TestNorthCarolinaDocketToolUnit:
    """Unit tests for NorthCarolinaDocketTool."""

    def test_init(self):
        """Test initialization sets name and registers method."""
        tool = NorthCarolinaDocketTool()
        assert tool.name == "north_carolina_dockets"
        assert "search_north_carolina_dockets" in tool._methods

    def test_get_tools_definition(self):
        """Test tool definitions are properly structured."""
        tool = NorthCarolinaDocketTool()
        tools = tool.get_tools()

        assert len(tools) == 1
        assert tools[0].name == "search_north_carolina_dockets"
        assert "north carolina" in tools[0].description.lower()
        assert "date_from" in tools[0].parameters["required"]
        assert "date_to" in tools[0].parameters["required"]

    def test_search_north_carolina_success(self, mocker):
        """Test successful NC UC search with mocked session (GET + POST)."""
        # Landing page with hidden fields
        landing_html = """
        <html><body>
            <input type="hidden" name="__VIEWSTATE" value="vs123" />
            <input type="hidden" name="__EVENTVALIDATION" value="ev456" />
        </body></html>
        """

        # Results page with docket rows
        results_html = """
        <html><body>
            <tr class="SearchResultsItem">
                <td class="width-full"><a href="/NCUC/ViewDocket/E-7-1234">E-7, Sub 1234</a></td>
                <td class="width-full">Duke Energy Rate Case</td>
                <td class="text-left width-full">Date Filed: 01/10/2025</td>
            </tr>
            <tr class="SearchResultsAlternatingItem">
                <td class="width-full"><a href="/NCUC/ViewDocket/E-2-5678">E-2, Sub 5678</a></td>
                <td class="width-full">Dominion Energy IRP</td>
                <td class="text-left width-full">Date Filed: 01/12/2025</td>
            </tr>
        </body></html>
        """

        mock_get_resp = Mock()
        mock_get_resp.status_code = 200
        mock_get_resp.raise_for_status = Mock()
        mock_get_resp.text = landing_html

        mock_post_resp = Mock()
        mock_post_resp.status_code = 200
        mock_post_resp.raise_for_status = Mock()
        mock_post_resp.text = results_html

        mock_session_instance = MagicMock()
        mock_session_instance.get.return_value = mock_get_resp
        mock_session_instance.post.return_value = mock_post_resp

        mocker.patch(
            "energbench.tools.dockets.north_carolina_tool.requests.Session",
            return_value=mock_session_instance,
        )
        mocker.patch.object(NorthCarolinaDocketTool, "_save_csv", return_value=None)

        tool = NorthCarolinaDocketTool()
        result = tool.search_north_carolina(
            date_from="01/01/2025",
            date_to="01/31/2025",
        )

        result_data = json.loads(result)
        assert "error" not in result_data
        assert "items" in result_data
        assert len(result_data["items"]) == 2
        assert result_data["items"][0]["docket_number"] == "E-7, Sub 1234"
        assert result_data["items"][0]["date_filed"] == "01/10/2025"
        assert result_data["pages_fetched"] == 1

    def test_search_north_carolina_api_error(self, mocker):
        """Test handling of non-403 HTTP errors from NC UC."""
        mock_response = Mock()
        mock_response.status_code = 500
        http_error = requests.exceptions.HTTPError("500 Internal Server Error")
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error

        mock_session_instance = MagicMock()
        mock_session_instance.get.return_value = mock_response

        mocker.patch(
            "energbench.tools.dockets.north_carolina_tool.requests.Session",
            return_value=mock_session_instance,
        )

        tool = NorthCarolinaDocketTool()
        result = tool.search_north_carolina(
            date_from="01/01/2025",
            date_to="01/31/2025",
        )

        result_data = json.loads(result)
        assert "error" in result_data

    def test_search_north_carolina_exception_handling(self, mocker):
        """Test generic exception handling in NC UC search."""
        mock_session_instance = MagicMock()
        mock_session_instance.get.side_effect = Exception("Connection reset")

        mocker.patch(
            "energbench.tools.dockets.north_carolina_tool.requests.Session",
            return_value=mock_session_instance,
        )

        tool = NorthCarolinaDocketTool()
        result = tool.search_north_carolina(
            date_from="01/01/2025",
            date_to="01/31/2025",
        )

        result_data = json.loads(result)
        assert "error" in result_data
        assert "Connection reset" in result_data["error"]


@pytest.mark.integration
@pytest.mark.slow
class TestNorthCarolinaDocketToolIntegration:
    """Integration tests for NorthCarolinaDocketTool with real API."""

    def test_search_north_carolina_real_api(self):
        """Test NC UC search against the real website."""
        tool = NorthCarolinaDocketTool()
        try:
            result = tool.search_north_carolina(
                date_from="01/01/2025",
                date_to="01/07/2025",
                max_pages=1,
                timeout=15,
            )
        except Exception:
            pytest.skip("NC UC website unavailable")

        result_data = json.loads(result)
        assert isinstance(result_data, dict)
        assert "items" in result_data or "error" in result_data
