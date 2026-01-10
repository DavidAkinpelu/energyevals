"""Regulatory docket tools for FERC and state PUCs."""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
import os
from urllib.parse import urlencode, urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger

from energbench.agent.providers import ToolDefinition

from .base_tool import BaseTool


class DocketTools(BaseTool):
    """Tools for searching regulatory dockets from FERC and state PUCs."""

    def __init__(self):
        super().__init__(
            name="dockets",
            description="Search regulatory dockets from FERC and state commissions",
        )

        self.register_method("search_ferc_dockets", self.search_ferc)
        self.register_method("get_maryland_psc_item", self.get_maryland_psc_item)
        self.register_method("get_maryland_official_filings", self.get_maryland_official_filings)
        self.register_method("search_texas_dockets", self.search_texas)
        self.register_method("search_new_york_dockets", self.search_new_york)
        self.register_method("search_north_carolina_dockets", self.search_north_carolina)
        self.register_method("search_south_carolina_dockets", self.search_south_carolina)
        self.register_method("search_virginia_dockets", self.search_virginia)
        self.register_method("search_dc_dockets", self.search_dc)

    def get_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="search_ferc_dockets",
                description="Search the FERC eLibrary for filings and dockets.",
                parameters={
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "Start date of the search window in 'YYYY-MM-DD' format."},
                        "end_date": {"type": "string", "description": "End date of the search window in 'YYYY-MM-DD' format."},
                        "keyword": {"type": "string", "description": "Keyword or phrase to search within documents (e.g. 'Capacity markets', 'data centers')."},
                        "docket_number": {"type": "string", "description": "Docket number to filter the search (e.g., 'ER25-1234')."},
                        "sub_docket_numbers": {"type": "array", "items": {"type": "string"}, "description": "List of sub-docket numbers to include in the filter."},
                        "search_full_text": {"type": "boolean", "default": True, "description": "Whether to search within the full text of documents."},
                        "search_description": {"type": "boolean", "default": True, "description": "Whether to search within the document descriptions."},
                        "results_per_page": {"type": "integer", "default": 50, "description": "Number of results to return per page."},
                        "page": {"type": "integer", "default": 0, "description": "Zero-based index for pagination (e.g., page=0 returns the first set of results)."},
                    },
                    "required": ["start_date", "end_date", "keyword"],
                },
            ),
            ToolDefinition(
                name="get_maryland_psc_item",
                description="Fetch a Maryland PSC case/rulemaking/public conference by number.",
                parameters={
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "description": "Type of item to retrieve. Options: 'rulemaking', 'public_conference', 'case'."   },
                        "number": {"type": "string", "description": "Unique identifier for the item (e.g., 'RM123456', 'PC123456')."},
                        "timeout": {"type": "integer", "default": 30, "description": "Timeout in seconds. Defaults to 30."},
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
                        "start_date": {"type": "string", "description": "Start date in 'MM/DD/YYYY' format."},
                        "end_date": {"type": "string", "description": "End date in 'MM/DD/YYYY' format."},
                        "company_name": {"type": "string", "description": "Optional filter for the 'Company Name' field."},
                        "maillog_number": {"type": "string", "description": "Optional filter for the 'Maillog #' field."},
                        "timeout": {"type": "integer", "default": 30, "description": "Timeout in seconds. Defaults to 30."},
                    },
                    "required": ["start_date", "end_date"],
                },
            ),
            ToolDefinition(
                name="search_texas_dockets",
                description="Search Texas Public Utility Commission (PUCT) filings or dockets.",
                parameters={
                    "type": "object",
                    "properties": {
                        "date_from": {"type": "string", "description": "Start date for the search in MM/DD/YYYY format."},
                        "date_to": {"type": "string", "description": "End date for the search in MM/DD/YYYY format."},
                        "utility_name": {"type": "string", "description": "Utility name to filter by"},
                        "control_number": {"type": "string", "description": "Control number to filter by"},
                        "description": {"type": "string", "description": "Description to filter by"},
                        "filing_description": {"type": "string", "description": "Text to match in filing descriptions. DO NOT USE THIS!!!"},
                        "utility_type": {"type": "string", "default": "A", "description": "Utility type to filter by. Options: A, T, W."},
                        "document_type": {"type": "string", "default": "ALL", "description": "Document type to filter by. Options: ALL (default)"},
                        "item_match": {"type": "string", "default": "Equal", "description": "Item match to filter by. Options: Equal (default), Contains, Starts With, Ends With, Does Not Contain."},
                        "sort_order": {"type": "string", "default": "Descending", "description": "Sort order to filter by. Options: Descending (newest first), Ascending (oldest first)."},
                        "timeout": {"type": "integer", "default": 30, "description": "Timeout in seconds. Defaults to 30."},
                    },
                    "required": ["date_from", "date_to"],
                },
            ),
            ToolDefinition(
                name="search_new_york_dockets",
                description="Search New York DPS cases or documents between dates.",
                parameters={
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "Start date (MM/DD/YYYY)"},
                        "end_date": {"type": "string", "description": "End date (MM/DD/YYYY)"},
                        "keyword": {"type": "string", "description": "Keyword to search for"},
                        "case_number": {"type": "string", "description": "Case number to filter by"},
                        "mode": {"type": "string", "enum": ["cases", "documents"], "default": "cases", "description": "Search mode: 'cases' or 'documents'"},
                        "timeout": {"type": "integer", "default": 30, "description": "Timeout in seconds. Defaults to 30."}
                    },
                    "required": ["start_date", "end_date"],
                },
            ),
            ToolDefinition(
                name="search_north_carolina_dockets",
                description="Search North Carolina Utilities Commission dockets.",
                parameters={
                    "type": "object",
                    "properties": {
                        "date_from": {"type": "string", "description": "Start date for the search in MM/DD/YYYY format."},
                        "date_to": {"type": "string", "description": "End date for the search in MM/DD/YYYY format."},
                        "docket_number": {"type": "string", "description": "Docket number to filter by"},
                        "company_name": {"type": "string", "description": "Company name to filter by"},
                        "exclude_closed": {"type": "boolean", "default": False, "description": "Whether to exclude closed dockets from results. Defaults to False."},
                        "limit_to_filing_type_labels": {"type": "array", "items": {"type": "string"}, "description": "List of filing type labels to restrict search results (case-insensitive exact matches)."},
                        "storage_state_path": {"type": "string", "description": "Optional Playwright storage state path to reuse Cloudflare session."},
                        "max_pages": {"type": "integer", "default": 5, "description": "Maximum number of pages to fetch. Defaults to 5."},
                        "timeout": {"type": "integer", "default": 30, "description": "Timeout in seconds. Defaults to 30."}
                    },
                    "required": ["date_from", "date_to"],
                },
            ),
            ToolDefinition(
                name="search_south_carolina_dockets",
                description="Search South Carolina PSC dockets by date range.",
                parameters={
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                        "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                        "organization": {"type": "string", "description": "Organization name to filter by"},
                        "individual": {"type": "string", "description": "Individual name to filter by"},
                        "summary": {"type": "string", "description": "Keyword(s) to search within filing summaries"},
                        "number_year": {"type": "string", "description": "Docket number year component"},
                        "number_sequence": {"type": "string", "description": "Docket number sequence component"},
                        "number_type": {"type": "string", "description": "Docket number typpe code"},
                        "timeout": {"type": "integer", "default": 30, "description": "Timeout in seconds"},
                    },
                    "required": ["start_date", "end_date"],
                },
            ),
            ToolDefinition(
                name="search_virginia_dockets",
                description="Search Virginia SCC daily filings by date range.",
                parameters={
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "Start date for the search in YYYY-MM-DD format (inclusive)."},
                        "end_date": {"type": "string", "description": "End date for the search in YYYY-MM-DD format (inclusive)."},
                        "docname_contains": {"type": "string", "description": "Keyword to search within document names"},
                        "case_contains": {"type": "string", "description": "Keyword to search within case numbers"},
                        "timeout": {"type": "integer", "default": 30, "description": "Timeout in seconds. Defaults to 30."},
                    },
                    "required": ["start_date", "end_date"],
                },
            ),
            ToolDefinition(
                name="search_dc_dockets",
                description="Search DC PSC filings by date range.",
                parameters={
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "Start date for the search in MM/DD/YYYY format."},
                        "end_date": {"type": "string", "description": "End date for the search in MM/DD/YYYY format."},
                        "keywords": {"type": "string", "description": "Keyword to search for in filings"},
                        "company_individual": {"type": "string", "description": "Company or individual to filter by"},
                        "case_type_id": {"type": "string", "description": "Case type identifier to filter by"},
                        "docket_number": {"type": "string", "description": "Docket/order number to filter by"},
                        "filing_type_id": {"type": "string", "description": "Filing type identifier to filter by"},
                        "sub_filing_type_id": {"type": "string", "description": "Sub-filing type identifier to filter by"},
                        "industry_type": {"type": "string", "description": "Industry type identifier to filter by"},
                        "records_to_show": {"type": "integer", "default": 50, "description": "Number of records to show. Defaults to 50."},
                        "records_to_skip": {"type": "integer", "default": 0, "description": "Number of records to skip. Defaults to 0."},
                        "timeout": {"type": "integer", "default": 30, "description": "Timeout in seconds. Defaults to 30."}
                    },
                    "required": ["start_date", "end_date"],
                },
            ),
        ]

    @staticmethod
    def _save_csv(rows: List[Dict[str, Any]], save_csv_path: Optional[str]) -> Optional[str]:
        if not save_csv_path:
            return None
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            df.to_csv(save_csv_path, index=False)
            return save_csv_path
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            return None

    def search_ferc(
        self,
        start_date: str,
        end_date: str,
        keyword: str,
        docket_number: Optional[str] = None,
        sub_docket_numbers: Optional[List[str]] = None,
        search_full_text: bool = True,
        search_description: bool = True,
        results_per_page: int = 50,
        page: int = 0
    ) -> str:
        """
        Search the FERC eLibrary using the AdvancedSearch API.

        This function sends a POST request to the FERC eLibrary API and returns structured results,
        including titles, docket numbers, filing dates, categories, affiliations, and downloadable PDF links.

        Parameters:
            start_date (str): Start date of the search window in "YYYY-MM-DD" format.
            end_date (str): End date of the search window in "YYYY-MM-DD" format.
            keyword (str): Keyword or phrase to search within documents (e.g. "Capacity markets", "data centers").
            docket_number (str, optional): Docket number to filter the search (e.g., "ER25-1234").
            sub_docket_numbers (List[str], optional): List of sub-docket numbers to include in the filter.
            search_full_text (bool): Whether to search within the full text of documents (default: True).
            search_description (bool): Whether to search within document descriptions (default: True).
            results_per_page (int): Number of results to return per page (default: 100).
            page (int): Zero-based index for pagination (e.g., page=0 returns the first set of results).
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

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
                        f"{a.get('afType')}: {a.get('affiliation')}" for a in hit.get("affiliations", [])
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

    def get_maryland_psc_item(
        self,
        kind: Literal["rulemaking", "public_conference", "case"],
        number: str,
        timeout: int = 30,
    ) -> str:
        """
        Fetch a Maryland PSC item (case/rulemaking/public conference) by number. 
        For rulemaking, number should have a prefix RM and PC for public conference. No prefix needed for case

        Parameters:
            kind (str): Type of item to retrieve. Options: "rulemaking", "public_conference".
            number (str): Unique identifier for the item (e.g., "RM123456", "PC123456").
            timeout (int): Timeout in seconds. Defaults to 30.
        Returns:
            JSON string with the item details.
        """
        try:
            base = "https://webpscxb.psc.state.md.us"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

            case_number_text = (num_el.get_text(strip=True) if num_el else "")
            filed_date_text = (date_el.get_text(strip=True) if date_el else "")
            caption_text = (cap_el.get_text(strip=True) if cap_el else "")

            case_number = case_number_text.split(":", 1)[-1].strip() if ":" in case_number_text else case_number_text
            filed_date = filed_date_text.split(":", 1)[-1].strip() if ":" in filed_date_text else filed_date_text

            table = soup.find("table", {"id": "caserulepublicdata"})
            entries: List[Dict[str, Any]] = []
            if table:
                tbody = table.find("tbody")
                if tbody:
                    for tr in tbody.find_all("tr"):
                        tds = tr.find_all("td")
                        if len(tds) < 3:
                            continue
                        idx_span = tds[0].find("span")
                        index_str = idx_span.get_text(strip=True) if idx_span else ""
                        pdf_rel = idx_span.get("data-pdf") if idx_span else None
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
        timeout: int = 30
    ) -> str:
        """
        Search Maryland PSC 'Official Filings' by date range, optionally filtering by company name or maillog number.
        
        Parameters:
            start_date (str): Start date in "MM/DD/YYYY" format (required).
            end_date (str): End date in "MM/DD/YYYY" format (required).
            company_name (str, optional): Optional filter for the 'Company Name' field.
            maillog_number (str, optional): Optional filter for the 'Maillog #' field.
            timeout (int): Timeout in seconds. Defaults to 30.
        
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
                return el.get("value", "") if el else ""

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

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_csv_path = f"maryland_official_filings_{timestamp}.csv"

            results = []
            for tr in table.select("tr"):
                tds = tr.find_all("td")
                if len(tds) < 2:
                    continue

                span = tds[0].select_one("span.btnOpenPdf")
                maillog_raw = span.get_text(strip=True) if span else tds[0].get_text(strip=True)
                pdf_rel = span.get("data-pdf") if span and span.has_attr("data-pdf") else None
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
        """
        Query the Texas PUCT Interchange website for filings or dockets using flexible search inputs.

        This function supports:
          - Filings search (when `control_number` is provided)
          - Dockets search (when `utility_name` is provided)
          - Keyword-only search using `description`
          - All of the above can be combined with a required date range and other filters

        Parameters:
            date_from (str): Start of filing date range in "MM/DD/YYYY" format (required).
            date_to (str): End of filing date range in "MM/DD/YYYY" format (required).
            utility_name (str, optional): Name of the utility to search (for docket search).
            control_number (str, optional): Docket control number (for filings search).
            description (str, optional): Text to match in docket descriptions.
            filing_description (str, optional): Text to match in filing descriptions. DO NOT USE THIS!!
            utility_type (str): Utility type. Default is "A" (ALL). Options: "A", "T", "W".
            document_type (str): Filter by document type. Default is "ALL".
            item_match (str): Match logic for string filters. Can only be "Equal".
            sort_order (str): Sort by date filed. "Descending" shows newest first, "Ascending" shows oldest first.
            timeout (int): Timeout in seconds. Defaults to 30
        Returns:
            JSON string with the search results.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            response = requests.get(full_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
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
                    control_link = "https://interchange.puc.texas.gov" + link_tag.get("href", "")
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

    def search_new_york(
        self,
        start_date: str,
        end_date: str,
        keyword: Optional[str] = None,
        case_number: Optional[str] = None,
        mode: str = "cases",
        timeout: int = 30
    ) -> str:
        """Search New York DPS cases or documents between dates.
        Args:
            start_date: Start date (MM/DD/YYYY)
            end_date: End date (MM/DD/YYYY)
            keyword: Keyword to search for
            case_number: Case number to filter by
            mode: Search mode - "cases" or "documents"
            timeout: Timeout in seconds
            save_csv_path: Optional path to save results as CSV
        Returns:
            JSON string with the search results.
        """
        try:
            from html import unescape
            from datetime import timezone

            base = "https://documents.dps.ny.gov/public"
            token = "17-02256"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_csv_path = f"new_york_dps_{mode}_{timestamp}.csv"

            ms_date_re = re.compile(r"/Date\((\d+)\)/")

            def ms_date_to_str(s: Optional[str]) -> Optional[str]:
                if not s or not isinstance(s, str):
                    return None
                m = ms_date_re.fullmatch(s)
                if not m:
                    return None
                millis = int(m.group(1))
                dt = datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc)
                return dt.strftime("%Y-%m-%d")

            def extract_anchor(html_snippet: Optional[str]) -> tuple[Optional[str], Optional[str]]:
                if not html_snippet:
                    return None, None
                try:
                    soup = BeautifulSoup(unescape(html_snippet), "html.parser")
                    a = soup.find("a")
                    if not a:
                        return soup.get_text(strip=True) or None, None
                    text = a.get_text(strip=True) or None
                    href = a.get("href")
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
            if not qstring or not is_matter:
                raise RuntimeError("Could not locate required hidden fields on results page.")

            query_string = unescape(qstring.get("value", ""))
            is_cases_mode = (is_matter.get("value", "").lower() == "true")
            data_url = (
                f"{base}/CaseMaster/MatterExternal/{token}?{query_string}"
                if is_cases_mode
                else f"{base}/CaseMaster/DocumentExternal/{token}?{query_string}"
            )

            data_resp = session.get(data_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
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

    def search_north_carolina(
        self,
        date_from: str,
        date_to: str,
        docket_number: Optional[str] = None,
        company_name: Optional[str] = None,
        exclude_closed: bool = False,
        limit_to_filing_type_labels: Optional[List[str]] = None,
        storage_state_path: Optional[str] = None,
        max_pages: int = 5,
        timeout: int = 30
    ) -> str:
        """Search North Carolina Utilities Commission dockets by date range, docket number, company name,
         exclude closed, limit to filing type labels, and max pages.
        Args:
            date_from: Start date for the search in MM/DD/YYYY format.
            date_to: End date for the search in MM/DD/YYYY format.
            docket_number: Docket number to filter by
            company_name: Company name to filter by
            exclude_closed: Whether to exclude closed dockets from results. Defaults to False.
            limit_to_filing_type_labels: List of filing type
                labels to restrict search results (case-insensitive exact matches). Defaults to None.
            storage_state_path: Optional Playwright storage state path to reuse Cloudflare session.
            max_pages: Maximum number of pages to fetch. Defaults to 5.
            timeout: Timeout in seconds. Defaults to 30.
        Returns:
            JSON string with the search results.
        """
        try:
            portal_url = "https://starw1.ncuc.gov/NCUC/page/Dockets/portal.aspx"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_csv_path = f"north_carolina_dockets_{timestamp}.csv"

            def collect_hidden_fields(soup: BeautifulSoup) -> Dict[str, str]:
                data: Dict[str, str] = {}
                for inp in soup.select("input[type=hidden]"):
                    name = inp.get("name")
                    if name:
                        data[name] = inp.get("value", "")
                return data

            def resolve_filing_type_values(soup: BeautifulSoup, desired_labels: List[str]) -> List[str]:
                desired_set = {lbl.strip().lower() for lbl in desired_labels}
                values: List[str] = []
                sel = soup.find("select", id=re.compile(r"_filingTypesList$"))
                if not sel:
                    return values
                for opt in sel.find_all("option"):
                    label = (opt.text or "").strip().lower()
                    if label in desired_set:
                        values.append(opt.get("value", ""))
                return values

            def parse_results_page(html: str) -> Dict[str, Any]:
                soup = BeautifulSoup(html, "html.parser")
                results: List[Dict[str, Any]] = []

                item_count = None
                count_span = soup.find(id=re.compile(r"_itemCountLabel$"))
                if count_span and count_span.text:
                    match = re.search(r"Items Count:(\d+)", count_span.text)
                    if match:
                        item_count = int(match.group(1))

                rss_link = None
                rss_a = soup.find("a", id=re.compile(r"RssButtonControl1_rssButtonHyperLink$"))
                if rss_a and rss_a.has_attr("href"):
                    rss_link = rss_a["href"]

                for row in soup.select("tr.SearchResultsItem, tr.SearchResultsAlternatingItem"):
                    a = row.select_one("a[href]")
                    if not a:
                        continue
                    docket_number_value = (a.get_text() or "").strip()
                    docket_link = a["href"]

                    tds = row.select("td.width-full")
                    description = None
                    if len(tds) >= 2:
                        description = (tds[1].get_text() or "").strip()

                    date_td = row.find("td", class_="text-left width-full")
                    date_filed = None
                    if date_td:
                        txt = " ".join(date_td.stripped_strings)
                        match = re.search(r"Date Filed:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{4})", txt)
                        if match:
                            date_filed = match.group(1)

                    results.append(
                        {
                            "docket_number": docket_number_value,
                            "date_filed": date_filed,
                            "description": description,
                            "docket_link": docket_link,
                        }
                    )

                pager_targets: List[str] = []
                for a in soup.select("tr.SearchResultsFooter a[href^='javascript:__doPostBack(']"):
                    href = a.get("href", "")
                    match = re.search(r"__doPostBack\('([^']+)'", href)
                    if match:
                        pager_targets.append(match.group(1))

                return {
                    "items": results,
                    "item_count": item_count,
                    "rss_link": rss_link,
                    "pager_targets": pager_targets,
                }

            def run_playwright_search(state_path: str) -> Dict[str, Any]:
                try:
                    from playwright.sync_api import sync_playwright
                except ImportError as exc:
                    raise RuntimeError(
                        "Playwright is required for North Carolina UC scraping. "
                        "Install it with `pip install playwright` and run "
                        "`python -m playwright install`."
                    ) from exc

                with sync_playwright() as p:
                    has_state = os.path.exists(state_path)
                    browser = p.chromium.launch(headless=has_state)
                    context = (
                        browser.new_context(storage_state=state_path)
                        if has_state
                        else browser.new_context()
                    )
                    page = context.new_page()
                    page.set_default_timeout(timeout * 1000)
                    try:
                        page.goto(portal_url, wait_until="networkidle", timeout=timeout * 1000)
                    except Exception:
                        page.goto(portal_url, wait_until="domcontentloaded", timeout=timeout * 1000)

                    if not has_state:
                        logger.info(
                            "Complete the Cloudflare challenge in the opened browser, "
                            "then press Enter here to continue."
                        )
                        try:
                            input()
                        except EOFError:
                            raise RuntimeError(
                                "Playwright requires manual confirmation for the Cloudflare challenge."
                            ) from None

                        os.makedirs(os.path.dirname(state_path), exist_ok=True)
                        context.storage_state(path=state_path)

                    html = page.content()
                    landing = BeautifulSoup(html, "html.parser")

                    if date_from:
                        page.fill(f'input[name="{fld_from}"]', date_from)
                    if date_to:
                        page.fill(f'input[name="{fld_to}"]', date_to)
                    if docket_number:
                        page.fill(f'input[name="{fld_dkt}"]', docket_number)
                    if company_name:
                        page.fill(f'input[name="{fld_co}"]', company_name)

                    if exclude_closed:
                        page.check(f'input[name="{fld_excl}"]')

                    if limit_to_filing_type_labels:
                        values = resolve_filing_type_values(landing, limit_to_filing_type_labels)
                        if values:
                            page.check(f'input[name="{fld_chk}"]')
                            page.select_option(f'select[name="{fld_types}"]', values)

                    page.click(f'input[name="{btn_search}"]')
                    try:
                        page.wait_for_load_state("networkidle")
                    except Exception:
                        page.wait_for_load_state("domcontentloaded")

                    page_html = page.content()
                    parsed = parse_results_page(page_html)
                    all_items = parsed["items"]
                    pages_fetched = 1

                    pager_targets = parsed["pager_targets"]
                    while pages_fetched < max_pages and pager_targets:
                        target = pager_targets.pop(0)
                        page.evaluate(f"__doPostBack('{target}', '')")
                        try:
                            page.wait_for_load_state("networkidle")
                        except Exception:
                            page.wait_for_load_state("domcontentloaded")
                        htmlN = page.content()
                        parsedN = parse_results_page(htmlN)
                        all_items.extend(parsedN["items"])
                        pages_fetched += 1
                        pager_targets = parsedN["pager_targets"]

                    browser.close()

                return {
                    "items": all_items,
                    "item_count": parsed["item_count"],
                    "rss_link": parsed["rss_link"],
                    "pages_fetched": pages_fetched,
                }

            session = requests.Session()
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,image/apng,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": portal_url,
                "Origin": "https://starw1.ncuc.gov",
                "Upgrade-Insecure-Requests": "1",
                "Connection": "keep-alive",
            }
            try:
                r = session.get(portal_url, headers=headers, timeout=timeout)
                r.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 403:
                    state_path = storage_state_path or os.path.expanduser(
                        "~/.config/energbench/ncuc_storage_state.json"
                    )
                    payload = run_playwright_search(state_path)
                    saved_csv = self._save_csv(payload["items"], save_csv_path)
                    payload["saved_csv"] = saved_csv
                    return json.dumps(payload, indent=2)
                raise

            landing = BeautifulSoup(r.text, "html.parser")
            data = collect_hidden_fields(landing)

            base = "ctl00$ContentPlaceHolder1$PortalPageControl1$ctl86$DocketSearchControlNCUC1$"
            fld_from = base + "filedOnOrAfterTextBox"
            fld_to = base + "filedOnOrBeforeTextBox"
            fld_dkt = base + "docketNumberTextBox"
            fld_co = base + "companyNameTextBox"
            fld_excl = base + "filterByDocketTypeOpenClosed"
            fld_chk = base + "filterByDocketTypeCheckBox"
            fld_types = base + "filingTypesList"
            btn_search = base + "searchButton"

            if date_from:
                data[fld_from] = date_from
            if date_to:
                data[fld_to] = date_to
            if docket_number:
                data[fld_dkt] = docket_number
            if company_name:
                data[fld_co] = company_name

            if exclude_closed:
                data[fld_excl] = "on"
            else:
                data.pop(fld_excl, None)

            if limit_to_filing_type_labels:
                values = resolve_filing_type_values(landing, limit_to_filing_type_labels)
                if values:
                    data[fld_chk] = "on"
                    data[fld_types] = values

            data[btn_search] = "Search"

            r2 = session.post(portal_url, headers=headers, data=data, timeout=timeout)
            r2.raise_for_status()
            page_html = r2.text
            parsed = parse_results_page(page_html)

            all_items = parsed["items"]
            raw_pages = [page_html]
            pages_fetched = 1

            pager_targets = parsed["pager_targets"]
            while pages_fetched < max_pages and pager_targets:
                target = pager_targets.pop(0)
                soup_prev = BeautifulSoup(raw_pages[-1], "html.parser")
                post_data = collect_hidden_fields(soup_prev)
                post_data["__EVENTTARGET"] = target
                post_data["__EVENTARGUMENT"] = ""

                rN = session.post(portal_url, headers=headers, data=post_data, timeout=timeout)
                rN.raise_for_status()
                htmlN = rN.text
                raw_pages.append(htmlN)
                parsedN = parse_results_page(htmlN)
                all_items.extend(parsedN["items"])
                pages_fetched += 1
                pager_targets = parsedN["pager_targets"]

            saved_csv = self._save_csv(all_items, save_csv_path)
            return json.dumps(
                {
                    "items": all_items,
                    "item_count": parsed["item_count"],
                    "rss_link": parsed["rss_link"],
                    "pages_fetched": pages_fetched,
                    "saved_csv": saved_csv,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"North Carolina UC search failed: {e}")
            return json.dumps({"error": str(e), "source": "North Carolina UC"})

    def search_south_carolina(
        self,
        start_date: str,
        end_date: str,
        organization: Optional[str] = None,
        individual: Optional[str] = None,
        summary: Optional[str] = None,
        number_year: Optional[str] = None,
        number_sequence: Optional[str] = None,
        number_type: Optional[str] = None,
        timeout: int = 30,
    ) -> str:
        """Search South Carolina PSC dockets by date range, organization, individual,
         summary, number year, number sequence, and number type.
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            organization: Organization name to filter by
            individual: Individual name to filter by
            summary: Keyword(s) to search within filing summaries
            number_year: Docket number year component
            number_sequence: Docket number sequence component
            number_type: Docket number type code
            timeout: Timeout in seconds
        Returns:
            JSON string with the search results.
        """
        try:
            base_url = "https://dms.psc.sc.gov"
            search_path = "/Web/Dockets/Search"

            params = {
                "NumberYear": number_year,
                "NumberSequence": number_sequence,
                "NumberType": number_type,
                "IndividualName": individual or "",
                "OrganizationName": organization or "",
                "Summary": summary or "",
                "StartDate": start_date or "",
                "EndDate": end_date or "",
            }

            resp = requests.get(urljoin(base_url, search_path), params=params, timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.select_one("table.datatable-standard-savestate")
            items: List[Dict[str, Any]] = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_csv_path = f"south_carolina_dockets_{timestamp}.csv"

            if table:
                tbody = table.find("tbody") or table
                for tr in tbody.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) < 2:
                        continue
                    link_a = tds[0].find("a", class_="detailNumber")
                    docket_number_value = (
                        link_a.get_text(strip=True) if link_a else tds[0].get_text(strip=True)
                    )
                    docket_link = (
                        urljoin(base_url, link_a["href"]) if link_a and link_a.has_attr("href") else None
                    )
                    summary_el = tds[1].find("span")
                    strong = summary_el.find("strong") if summary_el else None
                    summary_text = (
                        strong.get_text(" ", strip=True)
                        if strong
                        else (summary_el.get_text(" ", strip=True) if summary_el else tds[1].get_text(" ", strip=True))
                    )
                    parties_div = tds[1].find("div", class_="parties")
                    parties = []
                    if parties_div:
                        for a in parties_div.find_all("a"):
                            txt = a.get_text(" ", strip=True)
                            if txt:
                                parties.append(txt)
                    items.append(
                        {
                            "docket_number": docket_number_value,
                            "summary": summary_text,
                            "docket_link": docket_link,
                            "parties": parties,
                        }
                    )

            saved_csv = self._save_csv(items, save_csv_path)
            return json.dumps(
                {
                    "items": items,
                    "source_url": resp.url,
                    "saved_csv": saved_csv,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"South Carolina PSC search failed: {e}")
            return json.dumps({"error": str(e), "source": "South Carolina PSC"})

    def search_virginia(
        self,
        start_date: str,
        end_date: str,
        docname_contains: Optional[str] = None,
        case_contains: Optional[str] = None,
        timeout: int = 30
    ) -> str:
        """Search Virginia SCC daily filings by date range.
        Args:
            start_date: Start date for the search in YYYY-MM-DD format (inclusive).
            end_date: End date for the search in YYYY-MM-DD format (inclusive).
            docname_contains: Keyword to search within document names
            case_contains: Keyword to search within case numbers
            timeout: Timeout in seconds. Defaults to 30.
        Returns:
            JSON string with the search results.
        """
        try:
            base = "https://www.scc.virginia.gov/docketsearchapi/breeze/dailyfilings/getalldailyfilings"
            doc_base = "https://www.scc.virginia.gov/docketsearch/DOCS/"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_csv_path = f"virginia_dockets_{timestamp}.csv"

            start = datetime.fromisoformat(start_date).date()
            end = datetime.fromisoformat(end_date).date()
            end_plus = end.toordinal() + 1
            end = datetime.fromordinal(end_plus).date()

            start_utc = f"{start.isoformat()}T05:00:00.000Z"
            end_utc = f"{end.isoformat()}T05:00:00.000Z"

            filt = (
                f"(DateFiled ge datetime'{start_utc}') and "
                f"(DateFiled lt datetime'{end_utc}')"
            )
            params = {
                "$filter": filt,
                "$orderby": "Month,Day",
                "$select": "CaseNumber,DocName,Month,Day,Year,DocID,FileName",
            }
            url = f"{base}?{urlencode(params)}"

            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json() or []

            results = []
            for row in data:
                if docname_contains and docname_contains.lower() not in (row.get("DocName") or "").lower():
                    continue
                if case_contains and case_contains.lower() not in (row.get("CaseNumber") or "").lower():
                    continue
                y, m, d = row.get("Year"), row.get("Month"), row.get("Day")
                filed_date = None
                try:
                    filed_date = datetime(int(y), int(m), int(d)).date().isoformat()
                except Exception:
                    filed_date = None
                filename = row.get("FileName")
                doc_url = f"{doc_base}{filename}" if filename else None
                results.append(
                    {
                        "case_number": row.get("CaseNumber"),
                        "doc_name": row.get("DocName"),
                        "year": y,
                        "month": m,
                        "day": d,
                        "filed_date": filed_date,
                        "doc_id": row.get("DocID"),
                        "document_url": doc_url,
                    }
                )

            saved_csv = self._save_csv(results, save_csv_path)
            return json.dumps(
                {
                    "results": results,
                    "num_results": len(results),
                    "saved_csv": saved_csv,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"Virginia SCC search failed: {e}")
            return json.dumps({"error": str(e), "source": "Virginia SCC"})

    def search_dc(
        self,
        start_date: str,
        end_date: str,
        keywords: str = "",
        company_individual: str = "",
        case_type_id: str = "",
        docket_number: str = "",
        filing_type_id: str = "",
        sub_filing_type_id: str = "",
        industry_type: str = "",
        records_to_show: int = 50,
        records_to_skip: int = 0,
        timeout: int = 30,
        save_csv_path: Optional[str] = None,
    ) -> str:
        """Retrieve docket filings from the District of Columbia Public Service Commission (DC PSC) eDocket system.
        Args:
            start_date: Start date for the search in MM/DD/YYYY format.
            end_date: End date for the search in MM/DD/YYYY format.
            keywords: Keyword to search for in filinfs
            company_individual: Company or individual to filter by
            case_type_id: Case type to filter by
            docket_number: Docket/order number to filter by
            filing_type_id: Filing type identifier to filter by
            sub_filing_type_id: Sub-filing type identifier to filter by
            industry_type: Industry type identifier to filter by
            "1"= Electric,
            "2"= Natural Gas,
            "3" = Telecommunications,
            "4" = Multi-utility,
            records_to_show: Number of records to show. Defaults to 50.
            records_to_skip: Number of records to skip. Defaults to 0.
            timeout: Timeout in seconds. Defaults to 30
        Returns:
            JSON string with the search results.
        """
        try:
            url = "https://edocket.dcpsc.org/apis/api/Filing/GetFilings"
            params = {
                "isAdmin": "false",
                "orderByColumn": "receivedDate",
                "sortBy": "desc",
                "recordsToSkip": str(records_to_skip),
                "recordsToShow": str(records_to_show),
                "keywords": keywords,
                "isExactMatch": "false",
                "searchThruPDF": "false",
                "companyIndividual": company_individual,
                "caseTypeId": case_type_id,
                "caseNumber": "",
                "itemNumber": "",
                "orderNumber": docket_number,
                "filingTypeId": filing_type_id,
                "filingTypeOther": "",
                "subFilingTypeId": sub_filing_type_id,
                "subFilingTypeOther": "",
                "startDate": start_date,
                "endDate": end_date,
                "industryType": industry_type,
            }
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_csv_path = f"dc_psc_filings_{timestamp}.csv"

            filings_list = []
            for filing in data.get("resultsSet", []):
                download_url = (
                    f"https://edocket.dcpsc.org/public_filesearch/filing/{filing['attachmentId']}"
                    if filing.get("attachmentId") and not filing.get("isConfidential")
                    else None
                )
                description = BeautifulSoup(filing.get("description", ""), "html.parser").get_text()

                filings_list.append(
                    {
                        "filing_id": filing.get("filingId"),
                        "docket_number": filing.get("docketNumber"),
                        "company_or_individual": filing.get("companyOrIndividual"),
                        "filing_type": filing.get("filingType"),
                        "received_date": filing.get("receivedDate"),
                        "description": description,
                        "attachment_file_name": filing.get("attachmentFileName"),
                        "download_url": download_url,
                        "is_confidential": filing.get("isConfidential"),
                    }
                )

            saved_csv = self._save_csv(filings_list, save_csv_path)
            return json.dumps(
                {
                    "results": filings_list,
                    "num_results": len(filings_list),
                    "saved_csv": saved_csv,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"DC PSC search failed: {e}")
            return json.dumps({"error": str(e), "source": "DC PSC"})
