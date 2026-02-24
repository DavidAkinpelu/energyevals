import json
import os
import re
from typing import Any, Literal

import requests
from loguru import logger

from .base_tool import BaseTool, tool_method


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
        address: str,
        sector: Literal["Residential", "Commercial", "Industrial", "Lighting"],
        return_format: Literal["json"] = "json",
        detail: Literal["full"] = "full",
        version: Literal[7] = 7,
        eia_id = None,
        active_only: bool = True,
    ) -> str:
        """Call the OpenEI utility rates API and return tariff records for a specific address and customer sector.

        Args:
            address: str representing detailed address, including state and zip code of the building. Set a empty string
                     if not used
            sector: str representing building type. Can be one of "Residential", "Commercial", "Industrial", or "Lighting"
            return_format: fixed as json
            detail: tag fixed as full to get detailed responses from the api
            version: fixed at 7 which is the current latest version of the api
            eia_id: Optional input to specify the eia id of the assocaited utility if known. Default is None
                    For example, 13781 is the EIA Utility ID for Northern States Power Company - Wisconsin (NSPW), 
                    which is the Xcel Energy entity that serves Michigan.

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
        if eia_id != None:
            params["eia"] = eia_id

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
