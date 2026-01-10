"""Exa search tool for web search capabilities."""

import json
import os
from typing import Optional, Union

from loguru import logger

from energbench.agent.providers import ToolDefinition

from .base_tool import BaseTool


class SearchTool(BaseTool):
    """Web search tool using Exa API.

    Provides semantic web search capabilities for finding relevant
    information about energy markets, regulations, and technical topics.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        text_length_limit: int = 1000,
        default_num_results: int = 5,
        text: bool = True,
        highlights: bool = True,
        summary: bool = False,
        livecrawl: str = "always",
        livecrawl_timeout: Optional[int] = None,
        subpages: Optional[int] = None,
        subpage_target: Optional[list[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        content_type: Optional[str] = None,
        category: Optional[str] = None,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None
    ):
        """Initialize the search tool.

        Args:
            api_key: Exa API key. Defaults to EXA_API_KEY env var.
            text_length_limit: Maximum length of text content per result.
            default_num_results: Default number of results to return.
        """
        super().__init__(
            name="search",
            description="Web search using Exa for finding energy-related information",
        )

        self.api_key = api_key or os.getenv("EXA_API_KEY")
        self.text_length_limit = text_length_limit
        self.default_num_results = default_num_results
        self.text = text
        self.highlights = highlights
        self.summary = summary
        self.livecrawl = livecrawl
        self.livecrawl_timeout = livecrawl_timeout
        self.subpages = subpages
        self.subpage_target = subpage_target
        self.start_crawl_date = start_crawl_date
        self.end_crawl_date = end_crawl_date
        self.start_published_date = start_published_date
        self.end_published_date = end_published_date
        self.content_type = content_type
        self.category = category
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains

        if not self.api_key:
            logger.warning("EXA_API_KEY not set. Search functionality will be limited.")

        # Register methods
        self.register_method("search_web", self.search)
        self.register_method("get_page_contents", self.get_contents)

    def get_tools(self) -> list[ToolDefinition]:
        """Return tool definitions for the search tool."""
        return [
            ToolDefinition(
                name="search_web",
                description=(
                    "Search the web using Exa for relevant information about energy markets, "
                    "regulations, companies, and technical topics. Returns titles, URLs, "
                    "and text snippets from matching pages."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query describing what you're looking for",
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5, max: 10)",
                            "default": 5,
                        },
                        "text": {
                            "type": "boolean",
                            "description": "Include text content in results (default: true)",
                        },
                        "highlights": {
                            "type": "boolean",
                            "description": "Include highlights (default: true)",
                        },
                        "summary": {
                            "type": "boolean",
                            "description": "Include Exa summaries when available (default: false)",
                        },
                        "livecrawl": {
                            "type": "string",
                            "description": "Live crawl behavior (default: always)",
                        },
                        "start_crawl_date": {
                            "type": "string",
                            "description": "Filter by crawl date start (YYYY-MM-DD)",
                        },
                        "end_crawl_date": {
                            "type": "string",
                            "description": "Filter by crawl date end (YYYY-MM-DD)",
                        },
                        "start_published_date": {
                            "type": "string",
                            "description": "Filter by published date start (YYYY-MM-DD)",
                        },
                        "end_published_date": {
                            "type": "string",
                            "description": "Filter by published date end (YYYY-MM-DD)",
                        },
                        "type": {
                            "type": "string",
                            "description": "Content type filter (e.g., article, blog, video)",
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional category to filter results",
                            "enum": [
                                "company",
                                "research paper",
                                "news",
                                "pdf",
                                "github",
                                "financial report",
                            ],
                        },
                        "include_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Restrict results to these domains",
                        },
                        "exclude_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude results from these domains",
                        },
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="get_page_contents",
                description=(
                    "Get the full page contents, summaries, and metadata for a list of URLs. "
                    "Returns instant results from cache, with automatic live crawling as fallback for uncached pages. "
                    "Use this after search_web to get more details from promising results."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of URLs to crawl (backwards compatible with 'ids' parameter)",
                        },
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Deprecated - use 'urls' instead. Array of document IDs obtained from searches",
                        },
                        "text": {
                            "oneOf": [
                                {"type": "boolean"},
                                {"type": "object"}
                            ],
                            "description": "If true, returns full page text with default settings. If false, disables text return. Can also be an object with custom settings.",
                        },
                        "highlights": {
                            "oneOf": [
                                {"type": "boolean"},
                                {"type": "object"}
                            ],
                            "description": "Text snippets the LLM identifies as most relevant from each page. Can be a boolean or an object with configuration.",
                        },
                        "summary": {
                            "oneOf": [
                                {"type": "boolean"},
                                {"type": "object"}
                            ],
                            "description": "Summary object for page summaries. Can be a boolean or an object with configuration.",
                        },
                        "livecrawl": {
                            "type": "string",
                            "enum": ["never", "fallback", "preferred", "always"],
                            "description": "Live crawl behavior: 'never' (cached only), 'fallback' (cache first, live if unavailable), 'preferred' (live first, cache fallback), 'always' (always live crawl). Defaults to 'fallback'.",
                        },
                        "livecrawlTimeout": {
                            "type": "integer",
                            "description": "Maximum duration in milliseconds for a live crawl attempt (e.g., 10000 for 10 seconds). Defaults to 10000.",
                        },
                        "subpages": {
                            "type": "integer",
                            "description": "Number of subpages to crawl from the provided URLs. Defaults to 5.",
                        },
                        "subpageTarget": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keywords to target specific subpages (e.g., ['about', 'products'])",
                        },
                    },
                    "required": ["urls"],
                },
            ),
        ]

    def search(
        self,
        query: str,
        num_results: int = 5,
        category: Optional[str] = None,
        text: Optional[bool] = None,
        highlights: Optional[bool] = None,
        summary: Optional[bool] = None,
        livecrawl: Optional[str] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        type: Optional[str] = None,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None
    ) -> str:
        """Search the web using Exa.

        Args:
            query: The search query.
            num_results: Number of results to return.
            category: Optional category filter.

        Returns:
            JSON string with search results.
        """
        try:
            from exa_py import Exa

            if not self.api_key:
                return json.dumps({"error": "EXA_API_KEY not configured"})

            exa = Exa(self.api_key)

            # Build search parameters
            search_kwargs = {
                "text": self.text if text is None else text,
                "highlights": self.highlights if highlights is None else highlights,
                "summary": self.summary if summary is None else summary,
                "num_results": self.default_num_results if num_results is None else num_results,
                "livecrawl": livecrawl or self.livecrawl,
                "start_crawl_date": start_crawl_date or self.start_crawl_date,
                "end_crawl_date": end_crawl_date or self.end_crawl_date,
                "start_published_date": start_published_date or self.start_published_date,
                "end_published_date": end_published_date or self.end_published_date,
                "type": type or self.content_type,
                "include_domains": include_domains or self.include_domains,
                "exclude_domains": exclude_domains or self.exclude_domains
            }

            if category or self.category:
                search_kwargs["category"] = category or self.category

            search_kwargs = {k: v for k, v in search_kwargs.items() if v is not None}

            # Perform search
            results = exa.search_and_contents(query, **search_kwargs)

            # Parse results
            parsed_results = []
            for result in results.results:
                result_dict = {"url": result.url}

                if result.title:
                    result_dict["title"] = result.title

                if result.author:
                    result_dict["author"] = result.author

                if result.published_date:
                    result_dict["published_date"] = result.published_date

                if result.text:
                    text = result.text[: self.text_length_limit]
                    result_dict["text"] = text

                if hasattr(result, "highlights") and result.highlights:
                    result_dict["highlights"] = result.highlights

                parsed_results.append(result_dict)

            return json.dumps(
                {
                    "query": query,
                    "num_results": len(parsed_results),
                    "results": parsed_results,
                },
                indent=2,
                ensure_ascii=False,
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return json.dumps({"error": str(e), "query": query})

    def get_contents(
        self,
        urls: list[str],
        ids: Optional[list[str]] = None,
        text: Optional[Union[bool, dict]] = None,
        highlights: Optional[Union[bool, dict]] = None,
        summary: Optional[Union[bool, dict]] = None,
        livecrawl: Optional[str] = 'fallback',
        livecrawlTimeout: Optional[int] = 10000,
        subpages: Optional[int] = 5,
        subpageTarget: Optional[list[str]] = None,
    ) -> str:
        """Retrieve content from specific URLs.

        Get the full page contents, summaries, and metadata for a list of URLs.
        Returns instant results from cache, with automatic live crawling as fallback
        for uncached pages.

        Args:
            urls: Array of URLs to crawl (backwards compatible with 'ids' parameter).
            ids: Deprecated - use 'urls' instead. Array of document IDs obtained from searches.
            text: If true, returns full page text with default settings. If false, disables
                text return. Can also be an object with custom settings.
            highlights: Text snippets the LLM identifies as most relevant from each page.
                Can be a boolean or an object with configuration.
            summary: Summary object for page summaries. Can be a boolean or an object
                with configuration.
            livecrawl: Live crawl behavior: 'never' (cached only), 'fallback' (cache first,
                live if unavailable), 'preferred' (live first, cache fallback), 'always' (always live crawl). Defaults to 'fallback'.
            livecrawlTimeout: Maximum duration in milliseconds for a live crawl attempt
                (e.g., 10000 for 10 seconds). Defaults to 10000.
            subpages: Number of subpages to crawl from the provided URLs. Defaults to 5.
            subpageTarget: Keywords to target specific subpages (e.g., ['about', 'products']). Defaults to None.

        Returns:
            JSON string with page contents.
        """
        try:
            from exa_py import Exa

            if not self.api_key:
                return json.dumps({"error": "EXA_API_KEY not configured"})

            exa = Exa(self.api_key)

            url_list = urls if urls is not None else (ids if ids else [])

            params: dict = {
                "urls": url_list,
            }

            if text is not None:
                params["text"] = text
            else:
                params["text"] = self.text

            if highlights is not None:
                params["highlights"] = highlights
            else:
                params["highlights"] = self.highlights

            if summary is not None:
                params["summary"] = summary
            else:
                params["summary"] = self.summary

            if livecrawl is not None:
                params["livecrawl"] = livecrawl
            elif self.livecrawl:
                params["livecrawl"] = self.livecrawl

            #NOTE: Seems livecrawlTimeout has been deprecated but it's still in the API documentation

            # if livecrawlTimeout is not None:
            #     params["livecrawlTimeout"] = livecrawlTimeout
            # elif self.livecrawl_timeout is not None:
            #     params["livecrawlTimeout"] = self.livecrawl_timeout

            if subpages is not None:
                params["subpages"] = subpages
            elif self.subpages is not None:
                params["subpages"] = self.subpages

            if subpageTarget is not None:
                params["subpageTarget"] = subpageTarget
            elif self.subpage_target is not None:
                params["subpageTarget"] = self.subpage_target

            params = {k: v for k, v in params.items() if v is not None}

            results = exa.get_contents(**params)

            parsed_results = []
            for result in results.results:
                result_dict = {"url": result.url}

                if result.title:
                    result_dict["title"] = result.title

                if result.text:
                    text_content = result.text
                    if self.text_length_limit:
                        text_content = text_content[: self.text_length_limit]
                    result_dict["text"] = text_content

                if hasattr(result, "highlights") and result.highlights:
                    result_dict["highlights"] = result.highlights

                parsed_results.append(result_dict)

            return json.dumps(
                {
                    "num_results": len(parsed_results),
                    "contents": parsed_results,
                },
                indent=2,
                ensure_ascii=False,
            )

        except Exception as e:
            logger.error(f"Content retrieval failed: {e}")
            return json.dumps({"error": str(e), "urls": urls})
