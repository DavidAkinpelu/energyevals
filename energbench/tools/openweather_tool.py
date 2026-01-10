"""OpenWeather API tool for weather data."""

import json
import os
from typing import Optional

import requests
from loguru import logger

from energbench.agent.providers import ToolDefinition

from .base_tool import BaseTool


class OpenWeatherTool(BaseTool):
    """Tool for accessing OpenWeather API data.
    
    Provides access to current weather, forecasts, historical weather,
    air pollution data, and geocoding services.
    """

    def __init__(self, api_key: Optional[str] = None, units: str = "metric"):
        """Initialize the OpenWeather tool.

        Args:
            api_key: OpenWeather API key. Defaults to OPENWEATHER_API_KEY env var.
            units: Units of measurement. "metric" (default), "imperial", or "standard".
        """
        super().__init__(
            name="openweather",
            description="Access weather, forecast, air pollution, and geocoding data",
        )

        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")

        if not self.api_key:
            logger.warning("OPENWEATHER_API_KEY not set. Tool will not function.")

        self.units = units
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.geo_url = "https://api.openweathermap.org/geo/1.0"

        # Register methods
        self.register_method("geocode_location", self.geocode_location)
        self.register_method("get_current_weather", self.get_current_weather)
        self.register_method("get_forecast", self.get_forecast)
        self.register_method("get_historical_weather", self.get_historical_weather)
        self.register_method("get_air_pollution", self.get_air_pollution)

    def _make_request(self, url: str, params: dict) -> dict:
        """Make a request to the OpenWeather API."""
        if not self.api_key:
            return {"error": "OPENWEATHER_API_KEY not configured"}

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenWeather API request failed: {e}")
            return {"error": str(e)}

    def _geocode(self, location: str, limit: int = 1) -> dict:
        """Helper to geocode a location."""
        params = {
            "q": location,
            "limit": limit,
            "appid": self.api_key
        }
        return self._make_request(f"{self.geo_url}/direct", params)

    def geocode_location(self, location: str, limit: int = 1) -> str:
        """Convert location name to coordinates.
        
        Args:
            location: The location to geocode (e.g., "Austin, TX, USA").
            limit: The number of results to return (default: 1).
            
        Returns:
            JSON string with geocoding results.
        """
        result = self._geocode(location, limit)
        return json.dumps(result, indent=2)

    def get_current_weather(self, location: str) -> str:
        """Get current weather for a location.
        
        Args:
            location: The location to get weather for (e.g., "Austin, TX, USA").
            
        Returns:
            JSON string with current weather data.
        """
        geo_result = self._geocode(location)
        
        if "error" in geo_result:
            return json.dumps(geo_result, indent=2)
        
        if not geo_result:
            return json.dumps({"error": "Location not found"}, indent=2)
        
        lat = geo_result[0]["lat"]
        lon = geo_result[0]["lon"]
        
        params = {
            "lat": lat,
            "lon": lon,
            "units": self.units,
            "appid": self.api_key
        }
        
        weather = self._make_request(f"{self.base_url}/weather", params)
        
        if "error" not in weather:
            weather["location_name"] = geo_result[0].get("name", location)
            weather["country"] = geo_result[0].get("country", "")
        
        return json.dumps(weather, indent=2)

    def get_forecast(self, location: str, days: int = 5) -> str:
        """Get weather forecast for a location.
        
        Args:
            location: The location to get forecast for (e.g., "Austin, TX, USA").
            days: Number of days to forecast (1-5, default: 5).
            
        Returns:
            JSON string with forecast data.
        """
        geo_result = self._geocode(location)
        
        if "error" in geo_result:
            return json.dumps(geo_result, indent=2)
        
        if not geo_result:
            return json.dumps({"error": "Location not found"}, indent=2)
        
        lat = geo_result[0]["lat"]
        lon = geo_result[0]["lon"]
        cnt = min(days * 8, 40)  # 5 days max = 40 entries (3-hour intervals)
        
        params = {
            "lat": lat,
            "lon": lon,
            "units": self.units,
            "cnt": cnt,
            "appid": self.api_key
        }
        
        forecast = self._make_request(f"{self.base_url}/forecast", params)
        
        if "error" not in forecast:
            forecast["location_name"] = geo_result[0].get("name", location)
            forecast["country"] = geo_result[0].get("country", "")
        
        return json.dumps(forecast, indent=2)

    def get_historical_weather(
        self, 
        location: str, 
        start: int, 
        end: int, 
        type_inp: str = "hour"
    ) -> str:
        """Get historical weather data for a location.
        
        Args:
            location: The location to get historical weather for.
            start: Start date (Unix timestamp, UTC), e.g., 1369728000.
            end: End date (Unix timestamp, UTC), e.g., 1369789200.
            type_inp: Type of the call, keep as "hour".
            
        Returns:
            JSON string with historical weather data.
        """
        geo_result = self._geocode(location)
        
        if "error" in geo_result:
            return json.dumps(geo_result, indent=2)
        
        if not geo_result:
            return json.dumps({"error": "Location not found"}, indent=2)
        
        lat = geo_result[0]["lat"]
        lon = geo_result[0]["lon"]
        
        params = {
            "lat": lat,
            "lon": lon,
            "units": self.units,
            "start": start,
            "end": end,
            "type": type_inp,
            "appid": self.api_key
        }
        
        historical = self._make_request(f"{self.base_url}/history", params)
        
        if "error" not in historical:
            historical["location_name"] = geo_result[0].get("name", location)
            historical["country"] = geo_result[0].get("country", "")
        
        return json.dumps(historical, indent=2)

    def get_air_pollution(self, location: str) -> str:
        """Get current air pollution data for a location.
        
        Args:
            location: The location to get air pollution for.
            
        Returns:
            JSON string with air pollution data.
        """
        geo_result = self._geocode(location)
        
        if "error" in geo_result:
            return json.dumps(geo_result, indent=2)
        
        if not geo_result:
            return json.dumps({"error": "Location not found"}, indent=2)
        
        lat = geo_result[0]["lat"]
        lon = geo_result[0]["lon"]
        
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key
        }
        
        pollution = self._make_request(f"{self.base_url}/air_pollution", params)
        
        if "error" not in pollution:
            pollution["location_name"] = geo_result[0].get("name", location)
            pollution["country"] = geo_result[0].get("country", "")
        
        return json.dumps(pollution, indent=2)

    def get_tools(self) -> list[ToolDefinition]:
        """Return tool definitions for OpenWeather tools."""
        return [
            ToolDefinition(
                name="geocode_location",
                description=(
                    "Convert a location name to geographic coordinates (latitude/longitude). "
                    "Useful for getting coordinates before making weather API calls."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location to geocode (e.g., 'Austin, TX, USA')",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of results to return (default: 1)",
                            "default": 1,
                        },
                    },
                    "required": ["location"],
                },
            ),
            ToolDefinition(
                name="get_current_weather",
                description=(
                    "Get current weather conditions for a location including temperature, "
                    "humidity, pressure, wind speed, and weather description."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location to get weather for (e.g., 'Austin, TX, USA')",
                        },
                    },
                    "required": ["location"],
                },
            ),
            ToolDefinition(
                name="get_forecast",
                description=(
                    "Get weather forecast for up to 5 days with 3-hour intervals. "
                    "Includes temperature, humidity, precipitation, and wind predictions."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location to get forecast for (e.g., 'Austin, TX, USA')",
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days to forecast (1-5, default: 5)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 5,
                        },
                    },
                    "required": ["location"],
                },
            ),
            ToolDefinition(
                name="get_historical_weather",
                description=(
                    "Get historical weather data for a specific time range. "
                    "Requires Unix timestamps for start and end dates."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location to get historical weather for",
                        },
                        "start": {
                            "type": "integer",
                            "description": "Start date as Unix timestamp (UTC), e.g., 1369728000",
                        },
                        "end": {
                            "type": "integer",
                            "description": "End date as Unix timestamp (UTC), e.g., 1369789200",
                        },
                        "type_inp": {
                            "type": "string",
                            "description": "Type of call, keep as 'hour' (default)",
                            "default": "hour",
                        },
                    },
                    "required": ["location", "start", "end"],
                },
            ),
            ToolDefinition(
                name="get_air_pollution",
                description=(
                    "Get current air quality data including pollutant concentrations "
                    "(CO, NO, NO2, O3, SO2, PM2.5, PM10, NH3) and Air Quality Index."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location to get air pollution data for",
                        },
                    },
                    "required": ["location"],
                },
            ),
        ]

