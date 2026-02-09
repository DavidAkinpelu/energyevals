import json
import re
from typing import Any, Literal, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag
from loguru import logger

from energbench.agent.providers import ToolDefinition
from energbench.utils import generate_timestamp

from ._base import DocketBaseTool


class MarylandDocketTool(DocketBaseTool):
    """Search Maryland PSC cases, rulemakings, public conferences, and official filings."""

    def __init__(self) -> None:
        super().__init__(
            name="maryland_dockets",
            description="Search Maryland PSC cases and official filings",
        )
        self.register_method("get_maryland_psc_item", self.get_maryland_psc_item)
        self.register_method("get_maryland_official_filings", self.get_maryland_official_filings)

    def get_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="get_maryland_psc_item",
                description="Fetch a Maryland PSC case/rulemaking/public conference by number.",
                parameters={
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "description": (
                                "Type of item to retrieve. "
                                "Options: 'rulemaking', 'public_conference', 'case'."
                            ),
                        },
                        "number": {
                            "type": "string",
                            "description": (
                                "Unique identifier for the item "
                                "(e.g., 'RM123456', 'PC123456')."
                            ),
                        },
                        "timeout": {
                            "type": "integer",
                            "default": 30,
                            "description": "Timeout in seconds. Defaults to 30.",
                        },
                    },
                    "required": ["kind", "number"],
                },
            ),
            ToolDefinition(
                name="get_maryland_official_filings",
                description="Search Maryland PSC official filings by date range.",
                parameters={
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in 'MM/DD/YYYY' format.",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in 'MM/DD/YYYY' format.",
                        },
                        "company_name": {
                            "type": "string",
                            "description": "Optional filter for the 'Company Name' field.",
                        },
                        "maillog_number": {
                            "type": "string",
                            "description": "Optional filter for the 'Maillog #' field.",
                        },
                        "timeout": {
                            "type": "integer",
                            "default": 30,
                            "description": "Timeout in seconds. Defaults to 30.",
                        },
                    },
                    "required": ["start_date", "end_date"],
                },
            ),
        ]

    def get_maryland_psc_item(
        self,
        kind: Literal["rulemaking", "public_conference", "case"],
        number: str,
        timeout: int = 30,
    ) -> str:
        """Fetch a Maryland PSC item (case/rulemaking/public conference) by number.

        For rulemaking, number should have a prefix RM and PC for public conference.
        No prefix needed for case.

        Parameters:
            kind: Type of item to retrieve.
            number: Unique identifier for the item.
            timeout: Timeout in seconds. Defaults to 30.

        Returns:
            JSON string with the item details.
        """
        try:
            base = "https://webpscxb.psc.state.md.us"
            timestamp = generate_timestamp()
            save_csv_path = f"maryland_psc_item_{kind}_{number}_{timestamp}.csv"

            def normalize_number(kind_value: str, number_value: str) -> str:
                value = number_value.strip()
                if kind_value == "rulemaking":
                    return value if value.upper().startswith("RM") else f"RM{value}"
                if kind_value == "public_conference":
                    return value if value.upper().startswith("PC") else f"PC{value}"
                return value

            def endpoint(kind_value: str) -> str:
                if kind_value == "rulemaking":
                    return "/DMS/rm/"
                if kind_value == "public_conference":
                    return "/DMS/pc/"
                return "/DMS/case/"

            url = base + endpoint(kind) + normalize_number(kind, number)
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            num_el = soup.find(id=lambda x: x and x.endswith("_hCaseNum"))
            date_el = soup.find(id=lambda x: x and x.endswith("_hFiledDate"))
            cap_el = soup.find(id=lambda x: x and x.endswith("_hCaseCaption"))

            case_number_text = num_el.get_text(strip=True) if num_el else ""
            filed_date_text = date_el.get_text(strip=True) if date_el else ""
            caption_text = cap_el.get_text(strip=True) if cap_el else ""

            case_number = (
                case_number_text.split(":", 1)[-1].strip()
                if ":" in case_number_text
                else case_number_text
            )
            filed_date = (
                filed_date_text.split(":", 1)[-1].strip()
                if ":" in filed_date_text
                else filed_date_text
            )

            table = soup.find("table", {"id": "caserulepublicdata"})
            entries: list[dict[str, Any]] = []
            if table and isinstance(table, Tag):
                tbody = table.find("tbody")
                if tbody and isinstance(tbody, Tag):
                    for tr in tbody.find_all("tr"):
                        tds = tr.find_all("td")
                        if len(tds) < 3:
                            continue
                        idx_span = tds[0].find("span")
                        index_str = idx_span.get_text(strip=True) if idx_span else ""
                        pdf_rel = str(idx_span.get("data-pdf", "")) if idx_span else None
                        pdf_url = (base + pdf_rel) if pdf_rel else None
                        subject = " ".join(tds[1].get_text(" ", strip=True).split())
                        date = tds[2].get_text(strip=True)
                        entries.append(
                            {
                                "index": index_str,
                                "subject": subject,
                                "date": date,
                                "pdf_url": pdf_url,
                            }
                        )

            saved_csv = self._save_csv(entries, save_csv_path)
            return json.dumps(
                {
                    "case_number": case_number or None,
                    "filed_date": filed_date or None,
                    "caption": caption_text or None,
                    "entries": entries,
                    "saved_csv": saved_csv,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"Maryland PSC item fetch failed: {e}")
            return json.dumps({"error": str(e), "source": "Maryland PSC"})

    def get_maryland_official_filings(
        self,
        start_date: str,
        end_date: str,
        company_name: Optional[str] = None,
        maillog_number: Optional[str] = None,
        timeout: int = 30,
    ) -> str:
        """Search Maryland PSC 'Official Filings' by date range.

        Parameters:
            start_date: Start date in "MM/DD/YYYY" format.
            end_date: End date in "MM/DD/YYYY" format.
            company_name: Optional filter for the 'Company Name' field.
            maillog_number: Optional filter for the 'Maillog #' field.
            timeout: Timeout in seconds. Defaults to 30.

        Returns:
            JSON string with the search results.
        """
        try:
            psc_base = "https://webpscxb.psc.state.md.us"
            url = f"{psc_base}/DMS/official-filings"
            session = requests.Session()
            session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; filings-scraper/1.0)"})

            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            def get_hidden(element_id: str) -> str:
                el = soup.find(id=element_id)
                return str(el.get("value", "")) if el and isinstance(el, Tag) else ""

            viewstate = get_hidden("__VIEWSTATE")
            eventvalidation = get_hidden("__EVENTVALIDATION")
            viewstategen = get_hidden("__VIEWSTATEGENERATOR")

            payload = {
                "__EVENTTARGET": "",
                "__EVENTARGUMENT": "",
                "__VIEWSTATE": viewstate,
                "__VIEWSTATEGENERATOR": viewstategen,
                "__EVENTVALIDATION": eventvalidation,
                "ctl00$ContentPlaceHolder1$txtStartDate": start_date,
                "ctl00$ContentPlaceHolder1$txtEndDate": end_date,
                "ctl00$ContentPlaceHolder1$txtCompanyName": company_name or "",
                "ctl00$ContentPlaceHolder1$txtMailLogNum": maillog_number or "",
                "ctl00$ContentPlaceHolder1$btnSubmit": "Submit",
            }

            post = session.post(url, data=payload, timeout=timeout)
            post.raise_for_status()
            soup = BeautifulSoup(post.text, "html.parser")

            table = soup.select_one("#maillogdata tbody")
            if not table:
                return json.dumps({"results": [], "saved_csv": None}, indent=2)

            timestamp = generate_timestamp()
            save_csv_path = f"maryland_official_filings_{timestamp}.csv"

            results = []
            for tr in table.select("tr"):
                tds = tr.find_all("td")
                if len(tds) < 2:
                    continue

                span = tds[0].select_one("span.btnOpenPdf")
                maillog_raw = span.get_text(strip=True) if span else tds[0].get_text(strip=True)
                pdf_rel = str(span.get("data-pdf", "")) if span and span.has_attr("data-pdf") else None
                pdf_url = urljoin(psc_base, pdf_rel) if pdf_rel else None

                match = re.search(r"(\d{3,})", maillog_raw or "")
                maillog_num_clean = match.group(1) if match else None
                description = re.sub(r"\s+", " ", tds[1].get_text(" ", strip=True))

                results.append(
                    {
                        "maillog_raw": maillog_raw,
                        "maillog_number": maillog_num_clean,
                        "description": description,
                        "pdf_url": pdf_url,
                    }
                )

            saved_csv = self._save_csv(results, save_csv_path)
            return json.dumps({"results": results, "saved_csv": saved_csv}, indent=2)
        except Exception as e:
            logger.error(f"Maryland official filings search failed: {e}")
            return json.dumps({"error": str(e), "source": "Maryland PSC"})
