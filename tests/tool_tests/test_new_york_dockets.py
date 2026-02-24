import json
import os
from unittest.mock import MagicMock, Mock

import pytest
import requests

from energbench.tools.dockets import NewYorkDocketTool


class TestNewYorkDocketToolUnit:
    """Unit tests for NewYorkDocketTool."""

    def test_init(self):
        """Test initialization sets name and registers method."""
        tool = NewYorkDocketTool()
        assert tool.name == "new_york_dockets"
        assert "search_new_york_dockets" in tool._methods

    def test_get_tools_definition(self):
        """Test tool definitions are properly structured."""
        tool = NewYorkDocketTool()
        tools = tool.get_tools()

        assert len(tools) == 1
        assert tools[0].name == "search_new_york_dockets"
        assert "new york" in tools[0].description.lower()
        assert "start_date" in tools[0].parameters["required"]
        assert "end_date" in tools[0].parameters["required"]
        assert "mode" in tools[0].parameters["properties"]

    def test_search_new_york_success(self, mocker, monkeypatch):
        """Test successful NY DPS search with mocked session (two-step flow)."""
        monkeypatch.setenv("NY_DPS_TOKEN", "test-token")
        # Step 1: search results page returns HTML with hidden fields
        search_html = """
        <html><body>
            <input type="hidden" id="GridPlaceHolder_hdnQueryString"
                   value="MC=1&amp;SDF=01/01/2025&amp;SDT=01/31/2025" />
            <input type="hidden" id="GridPlaceHolder_hdnbIsMatter" value="True" />
        </body></html>
        """

        # Step 2: data endpoint returns JSON list
        data_json = json.dumps([
            {
                "MatterID": 12345,
                "MatterType": "Electric",
                "MatterSubType": "Rate Case",
                "MatterTitle": "Con Edison Rate Proceeding",
                "MatterCompanies": "Consolidated Edison",
                "strSubmitDate": "01/15/2025",
                "TotalRecords": 1,
                "StartDate": "/Date(1736899200000)/",
                "CaseOrMatterNumber": '<a href="./MatterManagement/CaseMaster.aspx?MatterCaseNo=25-E-0001">25-E-0001</a>',
            },
        ])

        mock_search_resp = Mock()
        mock_search_resp.status_code = 200
        mock_search_resp.raise_for_status = Mock()
        mock_search_resp.text = search_html

        mock_data_resp = Mock()
        mock_data_resp.status_code = 200
        mock_data_resp.raise_for_status = Mock()
        mock_data_resp.text = data_json

        mock_session_instance = MagicMock()
        mock_session_instance.get.side_effect = [mock_search_resp, mock_data_resp]

        mocker.patch(
            "energbench.tools.dockets.new_york_tool.requests.Session",
            return_value=mock_session_instance,
        )
        mocker.patch.object(NewYorkDocketTool, "_save_csv", return_value=None)

        tool = NewYorkDocketTool()
        result = tool.search_new_york(
            start_date="01/01/2025",
            end_date="01/31/2025",
            keyword="rate case",
            mode="cases",
        )

        result_data = json.loads(result)
        assert "error" not in result_data
        assert result_data["mode"] == "cases"
        assert len(result_data["records"]) == 1
        assert result_data["records"][0]["MatterID"] == 12345
        assert result_data["records"][0]["CaseOrMatterNumber"] == "25-E-0001"

    def test_search_new_york_missing_hidden_fields(self, mocker):
        """Test NY DPS search when hidden fields are missing."""
        search_html = "<html><body><p>Unexpected page content</p></body></html>"

        mock_search_resp = Mock()
        mock_search_resp.status_code = 200
        mock_search_resp.raise_for_status = Mock()
        mock_search_resp.text = search_html

        mock_session_instance = MagicMock()
        mock_session_instance.get.return_value = mock_search_resp

        mocker.patch(
            "energbench.tools.dockets.new_york_tool.requests.Session",
            return_value=mock_session_instance,
        )

        tool = NewYorkDocketTool()
        result = tool.search_new_york(
            start_date="01/01/2025",
            end_date="01/31/2025",
        )

        result_data = json.loads(result)
        assert "error" in result_data

    def test_search_new_york_api_error(self, mocker):
        """Test handling of HTTP errors from NY DPS."""
        mock_session_instance = MagicMock()
        mock_session_instance.get.side_effect = requests.exceptions.HTTPError(
            "503 Service Unavailable"
        )

        mocker.patch(
            "energbench.tools.dockets.new_york_tool.requests.Session",
            return_value=mock_session_instance,
        )

        tool = NewYorkDocketTool()
        result = tool.search_new_york(
            start_date="01/01/2025",
            end_date="01/31/2025",
        )

        result_data = json.loads(result)
        assert "error" in result_data

    def test_search_new_york_missing_token(self, monkeypatch):
        """Missing NY_DPS_TOKEN should return a clear error."""
        monkeypatch.delenv("NY_DPS_TOKEN", raising=False)

        tool = NewYorkDocketTool()
        result = tool.search_new_york(
            start_date="01/01/2025",
            end_date="01/31/2025",
        )

        result_data = json.loads(result)
        assert "error" in result_data
        assert "NY_DPS_TOKEN" in result_data["error"]

    def test_search_new_york_exception_handling(self, mocker):
        """Test generic exception handling in NY DPS search."""
        mock_session_instance = MagicMock()
        mock_session_instance.get.side_effect = Exception("SSL certificate error")

        mocker.patch(
            "energbench.tools.dockets.new_york_tool.requests.Session",
            return_value=mock_session_instance,
        )

        tool = NewYorkDocketTool()
        result = tool.search_new_york(
            start_date="01/01/2025",
            end_date="01/31/2025",
        )

        result_data = json.loads(result)
        assert "error" in result_data
        assert "SSL certificate error" in result_data["error"]


@pytest.mark.integration
@pytest.mark.slow
class TestNewYorkDocketToolIntegration:
    """Integration tests for NewYorkDocketTool with real API."""

    def test_search_new_york_real_api(self):
        """Test NY DPS search against the real website."""
        if not os.getenv("NY_DPS_TOKEN"):
            pytest.skip("NY_DPS_TOKEN not set")
        tool = NewYorkDocketTool()
        try:
            result = tool.search_new_york(
                start_date="01/01/2025",
                end_date="01/07/2025",
                mode="cases",
                timeout=15,
            )
        except Exception:
            pytest.skip("NY DPS website unavailable")

        result_data = json.loads(result)
        assert isinstance(result_data, dict)
        assert "records" in result_data or "error" in result_data
