import json
import os
from unittest.mock import Mock

import pytest
import requests

from energbench.tools.renewables_tool import RenewablesTool


class TestRenewablesToolUnit:
    """Unit tests for RenewablesTool."""

    def test_init_with_api_key(self, monkeypatch):
        """Test initialization with API key."""
        monkeypatch.setenv("RENEWABLES_NINJA_API_KEY", "test_key")
        tool = RenewablesTool()
        assert tool.name == "renewables"
        assert tool.api_key == "test_key"

    def test_init_without_api_key(self, monkeypatch):
        """Test initialization without API key."""
        monkeypatch.delenv("RENEWABLES_NINJA_API_KEY", raising=False)
        tool = RenewablesTool()
        assert tool.api_key is None

    def test_get_tools_definition(self):
        """Test tool definitions are properly structured."""
        tool = RenewablesTool(api_key="test_key")
        tools = tool.get_tools()

        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert "get_solar_profile" in tool_names
        assert "get_wind_profile" in tool_names

        solar_tool = next(t for t in tools if t.name == "get_solar_profile")
        assert "lat" in solar_tool.parameters["required"]
        assert "lon" in solar_tool.parameters["required"]

    def test_get_solar_profile_success(self, mocker):
        """Test successful solar profile generation with mocked API."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "outputs": {
                "ac": [100, 200, 300, 400, 500, 400, 300, 200]
            }
        }

        mocker.patch("energbench.tools.renewables_tool.requests.get", return_value=mock_response)

        tool = RenewablesTool(api_key="test_key")
        result = tool.get_solar_profile(
            lat=30.0,
            lon=-97.0,
            date_from="2019-01-01",
            date_to="2019-01-01",
            capacity=1.0,
        )

        result_data = json.loads(result)
        assert "error" not in result_data

    def test_get_wind_profile_success(self, mocker):
        """Test successful wind profile generation with mocked API."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "outputs": {
                "gen": [150, 250, 350, 450, 550, 450, 350, 250]
            }
        }

        mocker.patch("energbench.tools.renewables_tool.requests.get", return_value=mock_response)

        tool = RenewablesTool(api_key="test_key")
        result = tool.get_wind_profile(
            lat=30.0,
            lon=-97.0,
            date_from="2019-01-01",
            date_to="2019-01-01",
            capacity=1.0,
        )

        result_data = json.loads(result)
        assert "error" not in result_data

    def test_solar_profile_api_error(self, mocker):
        """Test handling of API errors in solar profile."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Bad Request")
        mock_response.text = "Invalid parameters"

        mocker.patch("energbench.tools.renewables_tool.requests.get", return_value=mock_response)

        tool = RenewablesTool(api_key="test_key")
        result = tool.get_solar_profile(
            lat=200.0,  # Invalid latitude
            lon=-97.0,
            date_from="2019-01-01",
            date_to="2019-01-01",
            capacity=1.0,
        )

        result_data = json.loads(result)
        assert "error" in result_data

    def test_wind_profile_exception_handling(self, mocker):
        """Test exception handling in wind profile."""
        mocker.patch(
            "energbench.tools.renewables_tool.requests.get",
            side_effect=requests.exceptions.ConnectionError("Network error"),
        )

        tool = RenewablesTool(api_key="test_key")
        result = tool.get_wind_profile(
            lat=30.0,
            lon=-97.0,
            date_from="2019-01-01",
            date_to="2019-01-01",
            capacity=1.0,
        )

        result_data = json.loads(result)
        assert "error" in result_data
        assert "Network error" in result_data["error"]


@pytest.mark.integration
@pytest.mark.requires_api_key
@pytest.mark.slow
class TestRenewablesToolIntegration:
    """Integration tests with real NREL API."""

    @pytest.mark.asyncio
    async def test_get_solar_profile_real_api(self):
        """Test solar profile with real NREL API."""
        if not os.getenv("RENEWABLES_NINJA_API_KEY"):
            pytest.skip("RENEWABLES_NINJA_API_KEY not set")

        tool = RenewablesTool()
        result = tool.get_solar_profile(
            lat=30.2672,
            lon=-97.7431,
            date_from="2019-01-01",
            date_to="2019-01-02",
            capacity=1.0,
        )

        result_data = json.loads(result)
        assert isinstance(result_data, dict)
        assert "error" in result_data or "location" in result_data

    @pytest.mark.asyncio
    async def test_get_wind_profile_real_api(self):
        """Test wind profile with real NREL API."""
        if not os.getenv("RENEWABLES_NINJA_API_KEY"):
            pytest.skip("RENEWABLES_NINJA_API_KEY not set")

        tool = RenewablesTool()
        result = tool.get_wind_profile(
            lat=30.2672,
            lon=-97.7431,
            date_from="2019-01-01",
            date_to="2019-01-02",
            capacity=1.0,
        )

        result_data = json.loads(result)
        assert isinstance(result_data, dict)
