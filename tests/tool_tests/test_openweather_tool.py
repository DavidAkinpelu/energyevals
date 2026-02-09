import json
import os
from unittest.mock import Mock

import pytest
import requests

from energbench.tools.openweather_tool import OpenWeatherTool


class TestOpenWeatherToolUnit:
    """Unit tests for OpenWeatherTool."""

    def test_init_with_api_key(self, monkeypatch):
        """Test initialization with API key."""
        monkeypatch.setenv("OPENWEATHER_API_KEY", "test_key")
        tool = OpenWeatherTool()
        assert tool.name == "openweather"
        assert tool.api_key == "test_key"

    def test_init_without_api_key(self, monkeypatch):
        """Test initialization without API key."""
        monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
        tool = OpenWeatherTool()
        assert tool.api_key is None

    def test_get_tools_definition(self):
        """Test tool definitions are properly structured."""
        tool = OpenWeatherTool(api_key="test_key")
        tools = tool.get_tools()

        assert len(tools) >= 1
        tool_names = {t.name for t in tools}
        assert "get_current_weather" in tool_names

    def test_get_current_weather_success(self, mocker):
        """Test successful weather retrieval with mocked API."""
        geo_response = Mock()
        geo_response.status_code = 200
        geo_response.raise_for_status = Mock()
        geo_response.json.return_value = [
            {"name": "Austin", "lat": 30.2672, "lon": -97.7431, "country": "US"}
        ]

        weather_response = Mock()
        weather_response.status_code = 200
        weather_response.raise_for_status = Mock()
        weather_response.json.return_value = {
            "name": "Austin",
            "main": {"temp": 75.2, "humidity": 65, "pressure": 1013},
            "weather": [{"description": "clear sky"}],
            "wind": {"speed": 5.5},
        }

        mocker.patch(
            "energbench.tools.openweather_tool.requests.get",
            side_effect=[geo_response, weather_response],
        )

        tool = OpenWeatherTool(api_key="test_key")
        result = tool.get_current_weather(location="Austin, TX")

        result_data = json.loads(result)
        assert "main" in result_data
        assert result_data["main"]["temp"] == 75.2

    def test_get_current_weather_api_error(self, mocker):
        """Test handling of API errors."""
        geo_response = Mock()
        geo_response.status_code = 200
        geo_response.raise_for_status = Mock()
        geo_response.json.return_value = [
            {"name": "Test", "lat": 0, "lon": 0, "country": "US"}
        ]

        weather_response = Mock()
        weather_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")

        mocker.patch(
            "energbench.tools.openweather_tool.requests.get",
            side_effect=[geo_response, weather_response],
        )

        tool = OpenWeatherTool(api_key="test_key")
        result = tool.get_current_weather(location="InvalidCityName12345")

        result_data = json.loads(result)
        assert "error" in result_data

    def test_get_current_weather_exception_handling(self, mocker):
        """Test exception handling."""
        mocker.patch(
            "energbench.tools.openweather_tool.requests.get",
            side_effect=requests.exceptions.ConnectionError("Connection timeout"),
        )

        tool = OpenWeatherTool(api_key="test_key")
        result = tool.get_current_weather(location="Austin, TX")

        result_data = json.loads(result)
        assert "error" in result_data
        assert "Connection timeout" in result_data["error"]


@pytest.mark.integration
@pytest.mark.requires_api_key
class TestOpenWeatherToolIntegration:
    """Integration tests with real OpenWeatherMap API."""

    @pytest.mark.asyncio
    async def test_get_current_weather_real_api(self):
        """Test weather retrieval with real API."""
        if not os.getenv("OPENWEATHER_API_KEY"):
            pytest.skip("OPENWEATHER_API_KEY not set")

        tool = OpenWeatherTool()
        result = tool.get_current_weather(location="Austin, TX")

        result_data = json.loads(result)
        assert isinstance(result_data, dict)
        assert "main" in result_data or "error" in result_data

    @pytest.mark.asyncio
    async def test_get_air_pollution_real_api(self):
        """Test air pollution retrieval with real API."""
        if not os.getenv("OPENWEATHER_API_KEY"):
            pytest.skip("OPENWEATHER_API_KEY not set")

        tool = OpenWeatherTool()
        result = tool.get_air_pollution(location="Austin, TX")

        result_data = json.loads(result)
        assert isinstance(result_data, dict)
        assert "list" in result_data or "error" in result_data
