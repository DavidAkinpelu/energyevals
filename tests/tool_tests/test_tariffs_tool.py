import json
import os
from unittest.mock import Mock

import pytest

from energbench.tools.tariffs_tool import TariffsTool


class TestTariffsToolUnit:
    """Unit tests for TariffsTool."""

    def test_init_with_api_key(self, monkeypatch):
        """Test initialization with API key."""
        monkeypatch.setenv("OPEN_EI_API_KEY", "test_key")
        tool = TariffsTool()
        assert tool.name == "tariffs"
        assert tool.api_key == "test_key"

    def test_init_without_api_key(self, monkeypatch):
        """Test initialization without API key logs warning."""
        monkeypatch.delenv("OPEN_EI_API_KEY", raising=False)
        tool = TariffsTool()
        assert tool.api_key is None

    def test_get_tools_definition(self):
        """Test tool definitions are properly structured."""
        tool = TariffsTool(api_key="test_key")
        tools = tool.get_tools()

        assert len(tools) == 1
        assert tools[0].name == "get_utility_tariffs"
        assert "tariff" in tools[0].description.lower()
        assert "address" in tools[0].parameters["properties"]
        assert "sector" in tools[0].parameters["properties"]
        assert "address" in tools[0].parameters["required"]

    def test_get_utility_tariffs_success(self, mocker):
        """Test successful tariff lookup with mocked API."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({
            "items": [
                {
                    "label": "Test Utility Rate",
                    "utility": "Test Electric",
                    "energyratestructure": [[{"rate": 0.12}]],
                }
            ]
        })

        mocker.patch("energbench.tools.tariffs_tool.requests.get", return_value=mock_response)

        tool = TariffsTool(api_key="test_key")
        result = tool.get_utility_tariffs(
            address="123 Test St, City, ST 12345",
            sector="Residential",
        )

        result_data = json.loads(result)
        assert isinstance(result_data, list)
        assert len(result_data) > 0
        assert result_data[0]["label"] == "Test Utility Rate"

    def test_get_utility_tariffs_api_error(self, mocker):
        """Test handling of API errors."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not found"

        mocker.patch("energbench.tools.tariffs_tool.requests.get", return_value=mock_response)

        tool = TariffsTool(api_key="test_key")
        result = tool.get_utility_tariffs(
            address="Invalid Address",
            sector="Residential",
        )

        result_data = json.loads(result)
        assert "error" in result_data

    def test_get_utility_tariffs_no_results(self, mocker):
        """Test handling when no tariffs found."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({"items": []})

        mocker.patch("energbench.tools.tariffs_tool.requests.get", return_value=mock_response)

        tool = TariffsTool(api_key="test_key")
        result = tool.get_utility_tariffs(
            address="123 Test St",
            sector="Residential",
        )

        result_data = json.loads(result)
        assert "error" in result_data

    def test_get_utility_tariffs_exception_handling(self, mocker):
        """Test exception handling in tariff lookup."""
        mocker.patch(
            "energbench.tools.tariffs_tool.requests.get",
            side_effect=Exception("Connection error"),
        )

        tool = TariffsTool(api_key="test_key")
        result = tool.get_utility_tariffs(
            address="123 Test St",
            sector="Residential",
        )

        result_data = json.loads(result)
        assert "error" in result_data
        assert "Connection error" in result_data["error"]


@pytest.mark.integration
@pytest.mark.requires_api_key
class TestTariffsToolIntegration:
    """Integration tests with real OpenEI API."""

    @pytest.mark.asyncio
    async def test_get_utility_tariffs_real_api(self):
        """Test tariff lookup with real API."""
        if not os.getenv("OPEN_EI_API_KEY"):
            pytest.skip("OPEN_EI_API_KEY not set")

        tool = TariffsTool()
        result = tool.get_utility_tariffs(
            address="1600 Pennsylvania Ave NW, Washington, DC 20500",
            sector="Commercial",
        )

        result_data = json.loads(result)
        assert isinstance(result_data, (list, dict))
        if isinstance(result_data, dict):
            assert "error" in result_data
        else:
            assert len(result_data) > 0
            assert "label" in result_data[0] or "utility" in result_data[0]
