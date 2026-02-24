import json
import os
import re
from datetime import datetime, timezone
from html import unescape
from typing import Literal
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup, Tag
from loguru import logger

from energbench.utils import generate_timestamp

from ..base_tool import tool_method
from ._base import DocketBaseTool


class NewYorkDocketTool(DocketBaseTool):
    """Search New York DPS cases or documents between dates."""

    def __init__(self) -> None:
        super().__init__(
            name="new_york_dockets",
            description="Search New York DPS cases or documents",
        )

    @tool_method(name="search_new_york_dockets")
    def search_new_york(
        self,
        start_date: str,
        end_date: str,
        keyword: str | None = None,
        case_number: str | None = None,
        mode: Literal["cases", "documents"] = "cases",
        timeout: int = 30,
    ) -> str:
        """Search New York DPS cases or documents between dates.

        Args:
            start_date: Start date (MM/DD/YYYY)
            end_date: End date (MM/DD/YYYY)
            keyword: Keyword to search for
            case_number: Case number to filter by
            mode: Search mode - "cases" or "documents"
            timeout: Timeout in seconds

        Returns:
            JSON string with the search results.
        """
        try:
            base = "https://documents.dps.ny.gov/public"
            token = os.environ.get("NY_DPS_TOKEN")
            if not token:
                return json.dumps({
                    "error": "NY_DPS_TOKEN environment variable is not set. "
                    "Please set it to use the New York DPS search.",
                    "source": "New York PSC",
                })
            timestamp = generate_timestamp()
            save_csv_path = f"new_york_dps_{mode}_{timestamp}.csv"

            ms_date_re = re.compile(r"/Date\((\d+)\)/")

            def ms_date_to_str(s: str | None) -> str | None:
                if not s or not isinstance(s, str):
                    return None
                m = ms_date_re.fullmatch(s)
                if not m:
                    return None
                millis = int(m.group(1))
                dt = datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc)
                return dt.strftime("%Y-%m-%d")

            def extract_anchor(html_snippet: str | None) -> tuple[str | None, str | None]:
                if not html_snippet:
                    return None, None
                try:
                    soup = BeautifulSoup(unescape(html_snippet), "html.parser")
                    a = soup.find("a")
                    if not a or not isinstance(a, Tag):
                        return soup.get_text(strip=True) or None, None
                    text = a.get_text(strip=True) or None
                    href = str(a.get("href", ""))
                    if href:
                        href_clean = href.lstrip("./")
                        if href_clean.startswith("../"):
                            href_clean = href_clean[3:]
                        abs_url = f"{base}/{href_clean.lstrip('/')}"
                    else:
                        abs_url = None
                    return text, abs_url
                except Exception:
                    return unescape(str(html_snippet)), None

            def build_search_url(search_mode: str) -> str:
                mc = "1" if search_mode.lower().startswith("case") else "0"
                params = {
                    "MC": mc,
                    "IA": "",
                    "MT": "",
                    "MST": "",
                    "CN": case_number or "",
                    "MCT": keyword or "",
                    "SDF": start_date,
                    "SDT": end_date,
                    "C": "",
                    "M": "",
                    "CO": "",
                }
                return f"{base}/Common/SearchResults.aspx?{urlencode(params)}"

            session = requests.Session()
            search_url = build_search_url(mode)
            resp = session.get(search_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            qstring = soup.find("input", id="GridPlaceHolder_hdnQueryString")
            is_matter = soup.find("input", id="GridPlaceHolder_hdnbIsMatter")
            if not qstring or not is_matter or not isinstance(qstring, Tag) or not isinstance(is_matter, Tag):
                raise RuntimeError("Could not locate required hidden fields on results page.")

            query_string = unescape(str(qstring.get("value", "")))
            is_cases_mode = str(is_matter.get("value", "")).lower() == "true"
            data_url = (
                f"{base}/CaseMaster/MatterExternal/{token}?{query_string}"
                if is_cases_mode
                else f"{base}/CaseMaster/DocumentExternal/{token}?{query_string}"
            )

            data_resp = session.get(
                data_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout
            )
            data_resp.raise_for_status()
            raw_text = data_resp.text.strip()

            records = []
            try:
                data = json.loads(raw_text)
                if isinstance(data, list):
                    for item in data:
                        record = {
                            "MatterID": item.get("MatterID"),
                            "MatterType": item.get("MatterType"),
                            "MatterSubType": item.get("MatterSubType"),
                            "MatterTitle": item.get("MatterTitle"),
                            "Company": item.get("MatterCompanies"),
                            "SubmitDate": item.get("strSubmitDate"),
                            "TotalRecords": item.get("TotalRecords"),
                        }

                        start_date_str = ms_date_to_str(item.get("StartDate"))
                        if start_date_str:
                            record["StartDate"] = start_date_str

                        txt, url = extract_anchor(item.get("CaseOrMatterNumber"))
                        record["CaseOrMatterNumber"] = txt
                        record["CaseOrMatterNumber_url"] = url

                        records.append(record)
            except json.JSONDecodeError:
                records = []

            saved_csv = self._save_csv(records, save_csv_path)
            return json.dumps(
                {
                    "search_url": search_url,
                    "data_url": data_url,
                    "mode": "cases" if is_cases_mode else "documents",
                    "records": records,
                    "saved_csv": saved_csv,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"New York PSC search failed: {e}")
            return json.dumps({"error": str(e), "source": "New York PSC"})
