"""Solar and wind profile tool using Renewables.ninja API."""

import json
import os
from typing import Optional
from datetime import datetime
import requests
from loguru import logger

from energbench.agent.providers import ToolDefinition

from .base_tool import BaseTool


class RenewablesTool(BaseTool):
    """Tool for getting solar and wind generation profiles.

    Uses the Renewables.ninja API to get normalized hourly generation
    profiles for solar PV and wind turbines at any location.
    """

    BASE_URL = "https://www.renewables.ninja/api"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the renewables tool.

        Args:
            api_key: Renewables.ninja API key. Defaults to RENEWABLES_NINJA_API_KEY env var.
        """
        super().__init__(
            name="renewables",
            description="Get solar and wind generation profiles",
        )

        self.api_key = api_key or os.getenv("RENEWABLES_NINJA_API_KEY")

        if not self.api_key:
            logger.warning("RENEWABLES_NINJA_API_KEY not set. Tool will not function.")

        self.register_method("get_solar_profile", self.get_solar_profile)
        self.register_method("get_wind_profile", self.get_wind_profile)

    @staticmethod
    def _extract_numeric_values(data: object) -> list[float]:
        """Collect numeric values from nested data structures."""
        values: list[float] = []

        def _add(value: object) -> None:
            if isinstance(value, bool):
                return
            if isinstance(value, (int, float)):
                values.append(float(value))
                return
            if isinstance(value, dict):
                for nested in value.values():
                    _add(nested)
                return
            if isinstance(value, (list, tuple)):
                for nested in value:
                    _add(nested)

        _add(data)
        return values

    def get_tools(self) -> list[ToolDefinition]:
        """Return tool definitions for the renewables tool."""
        return [
            ToolDefinition(
                name="get_solar_profile",
                description=(
                    "Get an annual hourly solar PV generation profile for a location. "
                    "Returns normalized capacity factors (0-1) for each hour of a year. "
                    "Useful for solar project analysis and energy modeling."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "lat": {
                            "type": "number",
                            "description": "Latitude of the location",
                        },
                        "lon": {
                            "type": "number",
                            "description": "Longitude of the location",
                        },
                        "date_from": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD)",
                        },
                        "date_to": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD)"
                        },
                        "capacity": {
                            "type": "number",
                            "description": "System capacity in kW (default: 1.0)",
                            "default": 1.0,
                        },
                        "tilt": {
                            "type": "number",
                            "description": "Panel tilt angle in degrees (default: latitude)",
                        },
                        "azimuth": {
                            "type": "number",
                            "description": "Panel azimuth in degrees (default: 180 for south)",
                            "default": 180,
                        },
                        "system_loss": {
                            "type": "number",
                            "description": "System losses as decimal (default: 0.1 for 10%)",
                            "default": 0.1,
                        },
                        "tracking": {
                            "type": "integer",
                            "description": "Tracking mode (0 none, 1 azimuth, 2 tilt+azimuth)",
                            "default": 0,
                        },
                        "format": {
                            "type": "string",
                            "description": "Response format (json or csv)",
                            "default": "json",
                        },
                    },

                    "required": ["lat", "lon", "date_from", "date_to"],
                },
            ),
            ToolDefinition(
                name="get_wind_profile",
                description=(
                    "Get an annual hourly wind generation profile for a location. "
                    "Returns normalized capacity factors (0-1) for each hour of a year. "
                    "Useful for wind project analysis and energy modeling."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "lat": {
                            "type": "number",
                            "description": "Latitude of the location",
                        },
                        "lon": {
                            "type": "number",
                            "description": "Longitude of the location",
                        },
                        "date_from": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD)",
                        },
                        "date_to": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD)",
                        },
                        
                        "capacity": {
                            "type": "number",
                            "description": "Turbine capacity in kW (default: 1.0)",
                            "default": 1.0,
                        },
                        "height": {
                            "type": "number",
                            "description": "Hub height in meters (default: 100)",
                            "default": 100,
                        },
                        "turbine": {
                            "type": "string",
                            "description": "Turbine model (default: Vestas V90 2000)",
                            "default": "Vestas V90 2000",
                        },
                        "format": {
                            "type": "string",
                            "description": "Response format (json or csv)",
                            "default": "json",
                        },
                    },
                    "required": ["lat", "lon", "date_from", "date_to"],
                },
            ),
        ]

    def _make_request(self, endpoint: str, params: dict) -> dict:
        """Make a request to the Renewables.ninja API."""
        if not self.api_key:
            return {"error": "RENEWABLES_NINJA_API_KEY not configured"}

        headers = {
            "x-api-key": self.api_key,
            "Authorization": f"Token {self.api_key}",
        }
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, headers=headers, params=params, timeout=120)
            response.raise_for_status()
            if params.get("format") == "csv":
                return response.text
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Renewables.ninja API request failed: {e}")
            return {"error": str(e)}

    def get_solar_profile(
        self,
        lat: float,
        lon: float,
        date_from: str,
        date_to: str,
        capacity: float = 1.0,
        tilt: Optional[float] = None,
        azimuth: float = 180,
        system_loss: float = 0.1,
        tracking: int = 0,
        format: str = "json",
    ) -> str:
        """Get annual solar generation profile.

        Args:
            lat: Latitude of the location in degrees.
            lon: Longitude of the location in degrees.
            date_from: Start date in YYYY-MM-DD format.
            date_to: End date in YYYY-MM-DD format.
            capacity: System capacity in kW (default: 1.0).
            tilt: Panel tilt angle in degrees (defaults to latitude).
            azimuth: Panel azimuth in degrees (default: 180 for south).
            system_loss: System losses as decimal (default: 0.1 for 10%). 
            tracking: Tracking mode (0 none, 1 azimuth, 2 tilt+azimuth).
            format: Response format (json or csv).
        """
        params = {
            "lat": lat,
            "lon": lon,
            "date_from": date_from,
            "date_to": date_to,
            "dataset": "merra2",
            "capacity": capacity,
            "system_loss": system_loss,
            "tracking": tracking,
            "tilt": tilt if tilt is not None else abs(lat),
            "azim": azimuth,
            "format": format,
        }

        result = self._make_request("data/pv", params)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_csv_path = f"solar_profile_{timestamp}.csv"

        if "error" in result:
            return json.dumps(result)

        if format.lower() == "csv":
            try:
                import pandas as pd
                from io import StringIO

                data = result if isinstance(result, str) else result.get("data", "")
                df = pd.read_csv(StringIO(data))
                if save_csv_path:
                    df.to_csv(save_csv_path, index=False)
                return json.dumps(
                    {
                        "location": {"lat": lat, "lon": lon},
                        "rows": len(df),
                        "data_preview": df.head(5).to_dict(orient="records"),
                        "saved_csv": save_csv_path,
                    },
                    indent=2,
                    default=str,
                )
            except Exception as e:
                return json.dumps({"error": str(e), "raw": result}, indent=2)

        data = result.get("data", {})
        values = self._extract_numeric_values(data)
        summary = {
            "location": {"lat": lat, "lon": lon},
            "capacity_kw": capacity,
            "annual_capacity_factor": sum(values) / len(values) if values else 0,
            "peak_output": max(values) if values else 0,
            "annual_generation_kwh": sum(values) * capacity,
            "num_hours": len(values),
            "sample_data": dict(list(data.items())[:10]) if isinstance(data, dict) else values[:10],
        }

        if save_csv_path and isinstance(data, dict):
            try:
                import pandas as pd

                df = pd.DataFrame(list(data.items()), columns=["timestamp", "value"])
                df.to_csv(save_csv_path, index=False)
                summary["saved_csv"] = save_csv_path
            except Exception as e:
                summary["csv_error"] = str(e)

        return json.dumps(summary, indent=2, default=str)

    def get_wind_profile(
        self,
        lat: float,
        lon: float,
        date_from: str,
        date_to: str,
        capacity: float = 1.0,
        height: float = 100,
        turbine: str = "Vestas V90 2000",
        format: str = "json",
    ) -> str:
        """Get annual wind generation profile.

        Args:
            lat: Latitude of the location.
            lon: Longitude of the location.
            date_from: Start date in YYYY-MM-DD format.
            date_to: End date in YYYY-MM-DD format.
            capacity: Turbine capacity in kW.
            height: Hub height in meters.
            turbine: Turbine model (default: Vestas V90 2000).
            format: Response format (json or csv).

        Returns:
            JSON string with hourly capacity factors.
        """
        params = {
            "lat": lat,
            "lon": lon,
            "date_from": date_from,
            "date_to": date_to,
            "dataset": "merra2",
            "capacity": capacity,
            "height": height,
            "turbine": turbine,
            "format": format,
        }

        result = self._make_request("data/wind", params)

        if "error" in result:
            return json.dumps(result)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_csv_path = f"wind_profile_{timestamp}.csv"

        if format.lower() == "csv":
            try:
                import pandas as pd
                from io import StringIO

                data = result if isinstance(result, str) else result.get("data", "")
                df = pd.read_csv(StringIO(data))
                if save_csv_path:
                    df.to_csv(save_csv_path, index=False)
                return json.dumps(
                    {
                        "location": {"lat": lat, "lon": lon},
                        "rows": len(df),
                        "data_preview": df.head(5).to_dict(orient="records"),
                        "saved_csv": save_csv_path,
                    },
                    indent=2,
                    default=str,
                )
            except Exception as e:
                return json.dumps({"error": str(e), "raw": result}, indent=2)

        data = result.get("data", {})
        values = self._extract_numeric_values(data)
        summary = {
            "location": {"lat": lat, "lon": lon},
            "capacity_kw": capacity,
            "hub_height_m": height,
            "annual_capacity_factor": sum(values) / len(values) if values else 0,
            "peak_output": max(values) if values else 0,
            "annual_generation_kwh": sum(values) * capacity,
            "num_hours": len(values),
            "sample_data": dict(list(data.items())[:10]) if isinstance(data, dict) else values[:10],
        }

        if save_csv_path and isinstance(data, dict):
            try:
                import pandas as pd

                df = pd.DataFrame(list(data.items()), columns=["timestamp", "value"])
                df.to_csv(save_csv_path, index=False)
                summary["saved_csv"] = save_csv_path
            except Exception as e:
                summary["csv_error"] = str(e)

        return json.dumps(summary, indent=2, default=str)
