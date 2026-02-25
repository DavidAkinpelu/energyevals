import json
import os
import re
from typing import Any, Literal

import requests
from loguru import logger

from .base_tool import BaseTool, tool_method

# 12×24 schedule matrices that bloat responses without adding analytical value.
# The rate structures (energyratestructure / demandratestructure) are kept because
# they contain the actual prices; the schedules only map time slots to tier indices.
_SCHEDULE_FIELDS = {
    "energyweekdayschedule",
    "energyweekendschedule",
    "demandweekdayschedule",
    "demandweekendschedule",
}


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

    @tool_method()
    def get_utility_tariffs(
        self,
        sector: Literal["Residential", "Commercial", "Industrial", "Lighting"],
        address: str = "",
        state: str = "",
        eia_id: str | None = None,
        active_only: bool = True,
        include_schedules: bool = False,
        return_format: Literal["json"] = "json",
        detail: Literal["full"] = "full",
        version: Literal[7] = 7,
    ) -> str:
        """Look up utility electricity tariff records from the OpenEI IURDB for a given location and customer sector.

        At least one of `address`, `state`, or `eia_id` MUST be provided. Omitting all three
        causes the API to return every tariff in the country, which will time out.

        Args:
            sector: Customer type. One of "Residential", "Commercial", "Industrial", or "Lighting".
            address: Full street address including city, state, and ZIP code
                     (e.g., "123 Main St, Richmond, VA 23219"). Preferred over `state` alone
                     because it resolves the specific utility serving that location.
                     Leave empty if using `state` or `eia_id` instead.
            state: Two-letter US state abbreviation (e.g., "VA") to retrieve all tariffs for
                   utilities in that state. Use when you don't have a specific address.
                   Ignored if `address` is provided.
            eia_id: EIA utility ID to retrieve tariffs for a specific utility
                    (e.g., 13781 for Northern States Power Company - Wisconsin).
                    Can be combined with `state` or `address`.
            active_only: If True (default), returns only currently active tariffs (no end date).
                         Set to False to include retired tariffs.
            include_schedules: If False (default), omits the 12×24 time-of-use schedule matrices
                               (energyweekdayschedule, energyweekendschedule, demandweekdayschedule,
                               demandweekendschedule) to keep the response compact. Set to True only
                               when you need to know which rate tier applies at a specific hour.
            return_format: Fixed as "json".
            detail: Fixed as "full" to get complete rate structure details.
            version: Fixed at 7 (current API version).

        Returns:
            JSON string with a list of tariff records, or an error message.
        """
        if not self.api_key:
            return json.dumps({"error": "OPEN_EI_API_KEY not configured"})

        address = (address or "").strip()
        state = (state or "").strip()
        eia_id_clean = (eia_id or "").strip() if eia_id is not None else ""

        if not address and not state and not eia_id_clean:
            return json.dumps({
                "error": (
                    "At least one of 'address', 'state', or 'eia_id' must be provided. "
                    "Querying without any location filter returns every tariff in the country "
                    "and will time out. Example: state='VA' for all Virginia tariffs."
                )
            })

        params: dict[str, str | int] = {
            "version": version,
            "format": return_format,
            "api_key": self.api_key,
            "sector": sector,
            "detail": detail,
        }
        if address:
            params["address"] = address
        elif state:
            params["state"] = state
        if eia_id_clean:
            params["eia"] = eia_id_clean

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=60)

            if response.status_code == 200:
                raw = response.text
                cleaned_text = re.sub(r'[\x00-\x1F\x7F]', '', raw)
                data = json.loads(cleaned_text)
                final_response: Any = []

                if len(data.get("items", [])) > 0:
                    items = data["items"]
                    if active_only:
                        items = [item for item in items if not item.get("enddate")]
                    if not include_schedules:
                        items = [{k: v for k, v in item.items() if k not in _SCHEDULE_FIELDS} for item in items]
                    final_response = items
                else:
                    final_response = {"error": "Tariffs not found for this location at this time"}
            else:
                final_response = {"error": f"Status {response.status_code}: {response.text}"}

            return json.dumps(final_response, indent=4)

        except Exception as e:
            logger.error(f"Tariff lookup failed: {e}")
            return json.dumps({"error": str(e), "address": address, "state": state})
