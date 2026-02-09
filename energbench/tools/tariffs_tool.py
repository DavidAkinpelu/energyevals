from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from loguru import logger

from energbench.agent.providers import ToolDefinition

from .base_tool import BaseTool


class TariffsTool(BaseTool):
    """Tool for looking up utility tariffs using the OpenEI API.

    Provides access to utility rate information including energy charges,
    demand charges, time-of-use rates, and more.
    """

    BASE_URL = "https://api.openei.org/utility_rates"

    def __init__(self, api_key: str | None = None):
        """Initialize the tariffs tool.

        Args:
            api_key: OpenEI API key. Defaults to OPEN_EI_API_KEY env var.
        """
        super().__init__(
            name="tariffs",
            description="Look up utility electricity tariffs and rate structures",
        )

        self.api_key = api_key or os.getenv("OPEN_EI_API_KEY")

        if not self.api_key:
            logger.warning("OPEN_EI_API_KEY not set. Tool will not function.")

        self.register_method("get_utility_tariffs", self.get_utility_tariffs)

    def get_tools(self) -> list[ToolDefinition]:
        """Return tool definitions for the tariffs tool."""
        return [
            ToolDefinition(
                name="get_utility_tariffs",
                description=(
                    "Look up electricity tariffs for a given address. Returns utility "
                    "rate structures including energy charges, demand charges, and "
                    "time-of-use periods. Useful for calculating electricity costs."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "address": {
                            "type": "string",
                            "description": "Street address to look up tariffs for, including state and zip code",
                        },
                        "sector": {
                            "type": "string",
                            "description": "Building type. Can be one of 'Residential', 'Commercial', 'Industrial', or 'Lighting'",
                            "enum": ["Residential", "Commercial", "Industrial", "Lighting"],
                        },
                        "return_format": {
                            "type": "string",
                            "description": "Fixed as 'json'",
                            "default": "json",
                        },
                        "detail": {
                            "type": "string",
                            "description": "Tag fixed as 'full' to get detailed responses from the api",
                            "default": "full",
                        },
                        "version": {
                            "type": "integer",
                            "description": "Fixed at 7 which is the current latest version of the api",
                            "default": 7,
                        },
                        "active_only": {
                            "type": "boolean",
                            "description": "Default is True. Selects only existing tariffs. Can be set to False to include retired tariffs",
                            "default": True,
                        },
                    },
                    "required": ["address", "sector"],
                },
            ),
        ]

    def get_utility_tariffs(
        self,
        address: str,
        sector: str,
        return_format: str = "json",
        detail: str = "full",
        version: int = 7,
        active_only: bool = True,
    ) -> str:
        """Calls an api and provides the response as a list of dictionaries with details about different utility tariffs.

        Args:
            address: str representing detailed address, including state and zip code of the building
            sector: str representing building type. Can be one of "Residential", "Commercial", "Industrial", or "Lighting"
            return_format: fixed as json
            detail: tag fixed as full to get detailed responses from the api
            version: fixed at 7 which is the current latest version of the api
            active_only: default is True. Selects only existing tariffs. Can be set to False to include retired tariffs

        Returns:
            JSON string with tariff information or error message.
        """
        if not self.api_key:
            return json.dumps({"error": "OPEN_EI_API_KEY not configured"})

        params: dict[str, str | int] = {
            "version": version,
            "format": return_format,
            "api_key": self.api_key,
            "sector": sector,
            "address": address,
            "detail": detail,
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)

            if response.status_code == 200:
                raw = response.text
                cleaned_text = re.sub(r'[\x00-\x1F\x7F]', '', raw)
                data = json.loads(cleaned_text)
                final_response: Any = []

                if len(data.get("items", [])) > 0:  # check if tariffs exist for the address and customer type
                    if active_only:
                        for item in data["items"]:
                            end_ts = item.get("enddate")
                            if end_ts:
                                # Item has an end date, skip it
                                continue
                            else:
                                final_response.append(item)
                    else:
                        final_response = data["items"]
                else:
                    final_response = {"error": "Tariffs not found for this location at this time"}
            else:
                final_response = {"error": f"Status {response.status_code}: {response.text}"}

            return json.dumps(final_response, indent=4)

        except Exception as e:
            logger.error(f"Tariff lookup failed: {e}")
            return json.dumps({"error": str(e), "address": address})
