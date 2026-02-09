import json
from typing import Optional
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup
from loguru import logger

from energbench.agent.providers import ToolDefinition
from energbench.utils import generate_timestamp, get_system_ca_bundle

from ._base import DocketBaseTool


class TexasDocketTool(DocketBaseTool):
    """Search Texas Public Utility Commission (PUCT) filings or dockets."""

    def __init__(self) -> None:
        super().__init__(
            name="texas_dockets",
            description="Search Texas PUCT filings or dockets",
        )
        self.register_method("search_texas_dockets", self.search_texas)

    def get_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="search_texas_dockets",
                description="Search Texas Public Utility Commission (PUCT) filings or dockets.",
                parameters={
                    "type": "object",
                    "properties": {
                        "date_from": {
                            "type": "string",
                            "description": "Start date for the search in MM/DD/YYYY format.",
                        },
                        "date_to": {
                            "type": "string",
                            "description": "End date for the search in MM/DD/YYYY format.",
                        },
                        "utility_name": {
                            "type": "string",
                            "description": "Utility name to filter by",
                        },
                        "control_number": {
                            "type": "string",
                            "description": "Control number to filter by",
                        },
                        "description": {
                            "type": "string",
                            "description": "Description to filter by",
                        },
                        "filing_description": {
                            "type": "string",
                            "description": "Text to match in filing descriptions. DO NOT USE THIS!!!",
                        },
                        "utility_type": {
                            "type": "string",
                            "default": "A",
                            "description": "Utility type to filter by. Options: A, T, W.",
                        },
                        "document_type": {
                            "type": "string",
                            "default": "ALL",
                            "description": "Document type to filter by. Options: ALL (default)",
                        },
                        "item_match": {
                            "type": "string",
                            "default": "Equal",
                            "description": (
                                "Item match to filter by. Options: Equal (default), Contains, "
                                "Starts With, Ends With, Does Not Contain."
                            ),
                        },
                        "sort_order": {
                            "type": "string",
                            "default": "Descending",
                            "description": (
                                "Sort order to filter by. Options: "
                                "Descending (newest first), Ascending (oldest first)."
                            ),
                        },
                        "timeout": {
                            "type": "integer",
                            "default": 30,
                            "description": "Timeout in seconds. Defaults to 30.",
                        },
                    },
                    "required": ["date_from", "date_to"],
                },
            ),
        ]

    def search_texas(
        self,
        date_from: str,
        date_to: str,
        utility_name: Optional[str] = None,
        control_number: Optional[str] = None,
        description: Optional[str] = None,
        filing_description: Optional[str] = None,
        utility_type: str = "A",
        document_type: str = "ALL",
        item_match: str = "Equal",
        sort_order: str = "Descending",
        timeout: int = 30,
    ) -> str:
        """Query the Texas PUCT Interchange website for filings or dockets.

        Parameters:
            date_from: Start of filing date range in "MM/DD/YYYY" format.
            date_to: End of filing date range in "MM/DD/YYYY" format.
            utility_name: Name of the utility to search (for docket search).
            control_number: Docket control number (for filings search).
            description: Text to match in docket descriptions.
            filing_description: Text to match in filing descriptions. DO NOT USE THIS!!
            utility_type: Utility type. Default is "A" (ALL).
            document_type: Filter by document type. Default is "ALL".
            item_match: Match logic for string filters.
            sort_order: Sort by date filed.
            timeout: Timeout in seconds. Defaults to 30.

        Returns:
            JSON string with the search results.
        """
        try:
            timestamp = generate_timestamp()
            save_csv_path = f"texas_puc_filings_{timestamp}.csv"
            if not (date_from and date_to):
                raise ValueError("date_from and date_to must be provided in MM/DD/YYYY format")

            is_filing_search = bool(control_number)
            base_url = (
                "https://interchange.puc.texas.gov/search/filings/"
                if is_filing_search
                else "https://interchange.puc.texas.gov/search/dockets/"
            )

            query_params = {
                "UtilityType": utility_type,
                "ItemMatch": item_match,
                "DocumentType": document_type,
                "SortOrder": sort_order,
                "DateFiledFrom": f"{date_from} 00:00:00",
                "DateFiledTo": f"{date_to} 00:00:00",
            }

            if control_number:
                query_params["ControlNumber"] = control_number
            if utility_name:
                query_params["UtilityName"] = utility_name
            if description:
                query_params["Description"] = description
            if filing_description:
                query_params["FilingDescription"] = filing_description

            full_url = f"{base_url}?{urlencode(query_params)}"
            # Use the system CA bundle instead of certifi's — the Texas PUC
            # cert chains to a root CA that certifi has dropped.
            response = requests.get(
                full_url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=timeout,
                verify=get_system_ca_bundle(),
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            rows = soup.select("table tr")[1:]
            results = []
            for row in rows:
                cols = row.find_all("td")
                if len(cols) == 4:
                    link_tag = cols[0].find("a")
                    if not link_tag:
                        continue
                    control_number_text = link_tag.text.strip()
                    control_link = "https://interchange.puc.texas.gov" + str(link_tag.get("href", ""))
                    filings = cols[1].text.strip()
                    utility = cols[2].text.strip()
                    summary = cols[3].text.strip()
                    results.append(
                        {
                            "control_number": control_number_text,
                            "description": summary,
                            "utility": utility,
                            "filings": filings,
                            "link": control_link,
                        }
                    )

            saved_csv = self._save_csv(results, save_csv_path)
            return json.dumps(
                {
                    "source": "Texas PUC",
                    "search_url": full_url,
                    "num_results": len(results),
                    "results": results,
                    "saved_csv": saved_csv,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"Texas PUC search failed: {e}")
            return json.dumps({"error": str(e), "source": "Texas PUC"})
