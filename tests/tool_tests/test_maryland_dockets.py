import json
from unittest.mock import MagicMock, Mock

import pytest
import requests

from energyevals.tools.dockets import MarylandDocketTool


class TestMarylandDocketToolUnit:
    """Unit tests for MarylandDocketTool."""

    def test_init(self):
        """Test initialization sets name and registers both methods."""
        tool = MarylandDocketTool()
        assert tool.name == "maryland_dockets"
        assert "get_maryland_psc_item" in tool._methods
        assert "get_maryland_official_filings" in tool._methods

    def test_get_tools_definition(self):
        """Test tool definitions are properly structured."""
        tool = MarylandDocketTool()
        tools = tool.get_tools()

        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert "get_maryland_psc_item" in tool_names
        assert "get_maryland_official_filings" in tool_names

        item_tool = next(t for t in tools if t.name == "get_maryland_psc_item")
        assert "kind" in item_tool.parameters["required"]
        assert "number" in item_tool.parameters["required"]

        filings_tool = next(t for t in tools if t.name == "get_maryland_official_filings")
        assert "start_date" in filings_tool.parameters["required"]
        assert "end_date" in filings_tool.parameters["required"]

    def test_get_maryland_psc_item_success(self, mocker):
        """Test successful Maryland PSC item fetch with mocked HTML response."""
        html = """
        <html><body>
            <span id="ctl00_hCaseNum">Case Number: 9876</span>
            <span id="ctl00_hFiledDate">Filed Date: 01/15/2025</span>
            <span id="ctl00_hCaseCaption">Application for Rate Adjustment</span>
            <table id="caserulepublicdata">
                <tbody>
                    <tr>
                        <td><span data-pdf="/DMS/pdf/12345">1</span></td>
                        <td>Initial Application</td>
                        <td>01/15/2025</td>
                    </tr>
                </tbody>
            </table>
        </body></html>
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.text = html

        mocker.patch(
            "energyevals.tools.dockets.maryland_tool.requests.get",
            return_value=mock_response,
        )
        mocker.patch.object(MarylandDocketTool, "_save_csv", return_value=None)

        tool = MarylandDocketTool()
        result = tool.get_maryland_psc_item(kind="case", number="9876")

        result_data = json.loads(result)
        assert "error" not in result_data
        assert result_data["case_number"] == "9876"
        assert result_data["filed_date"] == "01/15/2025"
        assert result_data["caption"] == "Application for Rate Adjustment"
        assert len(result_data["entries"]) == 1
        assert result_data["entries"][0]["index"] == "1"
        assert result_data["entries"][0]["subject"] == "Initial Application"

    def test_get_maryland_psc_item_rulemaking(self, mocker):
        """Test Maryland PSC item fetch for rulemaking normalizes RM prefix."""
        html = """
        <html><body>
            <span id="ctl00_hCaseNum">Rulemaking Number: RM56</span>
            <span id="ctl00_hFiledDate">Filed: 02/01/2025</span>
            <span id="ctl00_hCaseCaption">Net Metering Rules</span>
        </body></html>
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.text = html

        mock_get = mocker.patch(
            "energyevals.tools.dockets.maryland_tool.requests.get",
            return_value=mock_response,
        )
        mocker.patch.object(MarylandDocketTool, "_save_csv", return_value=None)

        tool = MarylandDocketTool()
        tool.get_maryland_psc_item(kind="rulemaking", number="56")

        call_url = mock_get.call_args[0][0]
        assert "/DMS/rm/RM56" in call_url

    def test_get_maryland_psc_item_api_error(self, mocker):
        """Test handling of HTTP errors from Maryland PSC."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Not Found"
        )

        mocker.patch(
            "energyevals.tools.dockets.maryland_tool.requests.get",
            return_value=mock_response,
        )

        tool = MarylandDocketTool()
        result = tool.get_maryland_psc_item(kind="case", number="999999")

        result_data = json.loads(result)
        assert "error" in result_data

    def test_get_maryland_official_filings_success(self, mocker):
        """Test successful Maryland official filings search with mocked session."""
        landing_html = """
        <html><body>
            <input type="hidden" id="__VIEWSTATE" value="abc123" />
            <input type="hidden" id="__EVENTVALIDATION" value="def456" />
            <input type="hidden" id="__VIEWSTATEGENERATOR" value="ghi789" />
        </body></html>
        """
        results_html = """
        <html><body>
            <table id="maillogdata">
                <tbody>
                    <tr>
                        <td><span class="btnOpenPdf" data-pdf="/DMS/pdf/99001">ML 99001</span></td>
                        <td>Order Approving Rate Increase</td>
                    </tr>
                    <tr>
                        <td><span class="btnOpenPdf" data-pdf="/DMS/pdf/99002">ML 99002</span></td>
                        <td>Staff Report on Energy Efficiency</td>
                    </tr>
                </tbody>
            </table>
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
        mock_session_instance.headers = MagicMock()

        mocker.patch(
            "energyevals.tools.dockets.maryland_tool.requests.Session",
            return_value=mock_session_instance,
        )
        mocker.patch.object(MarylandDocketTool, "_save_csv", return_value=None)

        tool = MarylandDocketTool()
        result = tool.get_maryland_official_filings(
            start_date="01/01/2025",
            end_date="01/31/2025",
        )

        result_data = json.loads(result)
        assert "error" not in result_data
        assert len(result_data["results"]) == 2
        assert result_data["results"][0]["maillog_number"] == "99001"
        assert result_data["results"][1]["maillog_number"] == "99002"

    def test_get_maryland_official_filings_no_results(self, mocker):
        """Test Maryland filings search when no table is returned."""
        landing_html = """
        <html><body>
            <input type="hidden" id="__VIEWSTATE" value="abc" />
            <input type="hidden" id="__EVENTVALIDATION" value="def" />
            <input type="hidden" id="__VIEWSTATEGENERATOR" value="ghi" />
        </body></html>
        """
        results_html = "<html><body><p>No results found.</p></body></html>"

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
        mock_session_instance.headers = MagicMock()

        mocker.patch(
            "energyevals.tools.dockets.maryland_tool.requests.Session",
            return_value=mock_session_instance,
        )

        tool = MarylandDocketTool()
        result = tool.get_maryland_official_filings(
            start_date="01/01/2025",
            end_date="01/31/2025",
        )

        result_data = json.loads(result)
        assert result_data["results"] == []

    def test_get_maryland_official_filings_exception(self, mocker):
        """Test exception handling in Maryland official filings search."""
        mock_session_instance = MagicMock()
        mock_session_instance.get.side_effect = Exception("DNS resolution failed")
        mock_session_instance.headers = MagicMock()

        mocker.patch(
            "energyevals.tools.dockets.maryland_tool.requests.Session",
            return_value=mock_session_instance,
        )

        tool = MarylandDocketTool()
        result = tool.get_maryland_official_filings(
            start_date="01/01/2025",
            end_date="01/31/2025",
        )

        result_data = json.loads(result)
        assert "error" in result_data
        assert "DNS resolution failed" in result_data["error"]


@pytest.mark.integration
@pytest.mark.slow
class TestMarylandDocketToolIntegration:
    """Integration tests for MarylandDocketTool with real API."""

    def test_get_maryland_psc_item_real_api(self):
        """Test Maryland PSC item fetch against the real website."""
        tool = MarylandDocketTool()
        try:
            result = tool.get_maryland_psc_item(kind="case", number="9645", timeout=15)
        except Exception:
            pytest.skip("Maryland PSC website unavailable")

        result_data = json.loads(result)
        assert isinstance(result_data, dict)
        assert "entries" in result_data or "error" in result_data

    def test_get_maryland_official_filings_real_api(self):
        """Test Maryland official filings search against the real API."""
        tool = MarylandDocketTool()
        try:
            result = tool.get_maryland_official_filings(
                start_date="01/01/2025",
                end_date="01/07/2025",
                timeout=15,
            )
        except Exception:
            pytest.skip("Maryland PSC API unavailable")

        result_data = json.loads(result)
        assert isinstance(result_data, dict)
        assert "results" in result_data or "error" in result_data
