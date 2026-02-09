import json
from typing import Optional

import requests
from loguru import logger

from energbench.agent.providers import ToolDefinition
from energbench.utils import generate_timestamp

from ._base import DocketBaseTool


class FERCDocketTool(DocketBaseTool):
    """Search the FERC eLibrary for filings and dockets."""

    def __init__(self) -> None:
        super().__init__(
            name="ferc_dockets",
            description="Search the FERC eLibrary for filings and dockets",
        )
        self.register_method("search_ferc_dockets", self.search_ferc)

    def get_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="search_ferc_dockets",
                description="Search the FERC eLibrary for filings and dockets.",
                parameters={
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date of the search window in 'YYYY-MM-DD' format.",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date of the search window in 'YYYY-MM-DD' format.",
                        },
                        "keyword": {
                            "type": "string",
                            "description": (
                                "Keyword or phrase to search within documents "
                                "(e.g. 'Capacity markets', 'data centers')."
                            ),
                        },
                        "docket_number": {
                            "type": "string",
                            "description": "Docket number to filter the search (e.g., 'ER25-1234').",
                        },
                        "sub_docket_numbers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of sub-docket numbers to include in the filter.",
                        },
                        "search_full_text": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to search within the full text of documents.",
                        },
                        "search_description": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to search within the document descriptions.",
                        },
                        "results_per_page": {
                            "type": "integer",
                            "default": 50,
                            "description": "Number of results to return per page.",
                        },
                        "page": {
                            "type": "integer",
                            "default": 0,
                            "description": (
                                "Zero-based index for pagination "
                                "(e.g., page=0 returns the first set of results)."
                            ),
                        },
                    },
                    "required": ["start_date", "end_date", "keyword"],
                },
            ),
        ]

    def search_ferc(
        self,
        start_date: str,
        end_date: str,
        keyword: str,
        docket_number: Optional[str] = None,
        sub_docket_numbers: Optional[list[str]] = None,
        search_full_text: bool = True,
        search_description: bool = True,
        results_per_page: int = 50,
        page: int = 0,
    ) -> str:
        """Search the FERC eLibrary using the AdvancedSearch API.

        Parameters:
            start_date: Start date of the search window in "YYYY-MM-DD" format.
            end_date: End date of the search window in "YYYY-MM-DD" format.
            keyword: Keyword or phrase to search within documents.
            docket_number: Docket number to filter the search (e.g., "ER25-1234").
            sub_docket_numbers: List of sub-docket numbers to include in the filter.
            search_full_text: Whether to search within the full text of documents.
            search_description: Whether to search within document descriptions.
            results_per_page: Number of results to return per page.
            page: Zero-based index for pagination.
        """
        try:
            url = "https://elibrary.ferc.gov/eLibraryWebAPI/api/Search/AdvancedSearch"
            payload = {
                "searchText": keyword,
                "searchFullText": search_full_text,
                "searchDescription": search_description,
                "dateSearches": [
                    {
                        "dateType": "filed_date",
                        "startDate": start_date,
                        "endDate": end_date,
                    }
                ],
                "availability": None,
                "affiliations": [],
                "categories": [],
                "libraries": [],
                "accessionNumber": None,
                "eFiling": False,
                "docketSearches": [
                    {
                        "docketNumber": docket_number or "",
                        "subDocketNumbers": sub_docket_numbers or [],
                    }
                ],
                "resultsPerPage": results_per_page,
                "curPage": page,
                "classTypes": [],
                "sortBy": "",
                "groupBy": "NONE",
                "idolResultID": "",
                "allDates": False,
            }

            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()

            timestamp = generate_timestamp()
            save_csv_path = f"ferc_search_{timestamp}.csv"

            results = []
            for hit in data.get("searchHits", []):
                result = {
                    "title": hit.get("description", ""),
                    "filed_date": hit.get("filedDate"),
                    "docket_numbers": hit.get("docketNumbers", []),
                    "category": hit.get("category"),
                    "libraries": hit.get("libraries", []),
                    "accession_number": hit.get("acesssionNumber"),
                    "pdf_files": [],
                    "affiliations": [
                        f"{a.get('afType')}: {a.get('affiliation')}"
                        for a in hit.get("affiliations", [])
                    ],
                }

                for file in hit.get("transmittals", []):
                    if file.get("fileType") == "PDF":
                        result["pdf_files"].append(
                            {
                                "file_name": file.get("fileName"),
                                "file_desc": file.get("fileDesc"),
                                "file_size": file.get("fileSize"),
                                "download_url": (
                                    "https://elibrary.ferc.gov/eLibrary/filedownload?fileid="
                                    f"{file.get('fileId')}"
                                ),
                            }
                        )
                results.append(result)

            saved_csv = self._save_csv(results, save_csv_path)
            return json.dumps(
                {
                    "source": "FERC",
                    "keyword": keyword,
                    "date_range": f"{start_date} to {end_date}",
                    "num_results": len(results),
                    "results": results,
                    "saved_csv": saved_csv,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"FERC search failed: {e}")
            return json.dumps({"error": str(e), "source": "FERC"})
