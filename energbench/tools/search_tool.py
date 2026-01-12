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
                            "enum": ["never", "fallback", "preferred", "always"],
                            "description": "Live crawl behavior (default: always)",
                        },
                        "search_type": {
                            "type": "string",
                            "enum": ["neural", "fast", "auto", "deep"],
                            "description": "Search method: neural (embeddings-based), fast (streamlined), auto (default, combines methods), deep (comprehensive with query expansion)",
                        },
                        "category": {
                            "type": "string",
                            "description": "Category to filter results",
                            "enum": [
                                "company",
                                "research paper",
                                "news",
                                "pdf",
                                "github",
                                "financial report",
                                "tweet",
                                "people",
                                "personal site",
                            ],
                        },
                        "start_crawl_date": {
                            "type": "string",
                            "description": "Filter by crawl date start (ISO 8601 format, e.g., '2023-01-01T00:00:00.000Z')",
                        },
                        "end_crawl_date": {
                            "type": "string",
                            "description": "Filter by crawl date end (ISO 8601 format, e.g., '2023-12-31T00:00:00.000Z')",
                        },
                        "start_published_date": {
                            "type": "string",
                            "description": "Filter by published date start (ISO 8601 format, e.g., '2023-01-01T00:00:00.000Z')",
                        },
                        "end_published_date": {
                            "type": "string",
                            "description": "Filter by published date end (ISO 8601 format, e.g., '2023-12-31T00:00:00.000Z')",
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
                            "description": "Array of URLs to crawl",
                        },
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Deprecated - use 'urls' instead",
                        },
                        "text": {
                            "oneOf": [
                                {"type": "boolean"},
                                {"type": "object"}
                            ],
                            "description": "Include full page text (default: true). Can be object with custom settings.",
                        },
                        "highlights": {
                            "oneOf": [
                                {"type": "boolean"},
                                {"type": "object"}
                            ],
                            "description": "Include relevant text snippets (default: true). Can be object with config.",
                        },
                        "summary": {
                            "oneOf": [
                                {"type": "boolean"},
                                {"type": "object"}
                            ],
                            "description": "Include page summaries (default: false). Can be object with config.",
                        },
                        "livecrawl": {
                            "type": "string",
                            "enum": ["never", "fallback", "preferred", "always"],
                            "description": "Live crawl behavior (default: fallback)",
                        },
                        "subpages": {
                            "type": "integer",
                            "description": "Number of subpages to crawl from the provided URLs",
                        },
                        "subpage_target": {
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
        num_results: Optional[int] = None,
        text: bool = True,
        highlights: bool = True,
        summary: bool = False,
        livecrawl: str = "always",
        search_type: str = "auto",
        category: Optional[str] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
    ) -> str:
        """Search the web using Exa.

        Args:
            query: The search query.
            num_results: Number of results to return (default: uses default_num_results).
            text: Include text content in results.
            highlights: Include highlights in results.
            summary: Include Exa summaries when available.
            livecrawl: Live crawl behavior ("never", "fallback", "preferred", "always").
            search_type: Search method ("neural", "fast", "auto", "deep"). Default: "auto".
            category: Category filter (e.g., "company", "news", "research paper", "pdf").
            start_crawl_date: Filter by crawl date start (ISO 8601, e.g., "2023-01-01T00:00:00.000Z").
            end_crawl_date: Filter by crawl date end (ISO 8601, e.g., "2023-12-31T00:00:00.000Z").
            start_published_date: Filter by published date start (ISO 8601, e.g., "2023-01-01T00:00:00.000Z").
            end_published_date: Filter by published date end (ISO 8601, e.g., "2023-12-31T00:00:00.000Z").
            include_domains: Restrict results to these domains.
            exclude_domains: Exclude results from these domains.

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
                "text": text,
                "highlights": highlights,
                "summary": summary,
                "num_results": num_results if num_results is not None else self.default_num_results,
                "livecrawl": livecrawl,
                "type": search_type,
                "category": category,
                "start_crawl_date": start_crawl_date,
                "end_crawl_date": end_crawl_date,
                "start_published_date": start_published_date,
                "end_published_date": end_published_date,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
            }

            # Remove None, empty strings, and empty lists
            search_kwargs = {
                k: v for k, v in search_kwargs.items()
                if v is not None and v != "" and v != []
            }

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
                    text_content = result.text[:self.text_length_limit]
                    result_dict["text"] = text_content

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
        urls: Optional[list[str]] = None,
        ids: Optional[list[str]] = None,
        text: Union[bool, dict] = True,
        highlights: Union[bool, dict] = True,
        summary: Union[bool, dict] = False,
        livecrawl: str = "fallback",
        subpages: Optional[int] = None,
        subpage_target: Optional[list[str]] = None,
    ) -> str:
        """Retrieve content from specific URLs.

        Get the full page contents, summaries, and metadata for a list of URLs.
        Returns instant results from cache, with automatic live crawling as fallback
        for uncached pages.

        Args:
            urls: Array of URLs to crawl.
            ids: Deprecated - use 'urls' instead. Array of document IDs obtained from searches.
            text: If true, returns full page text. Can also be an object with custom settings.
            highlights: Include text snippets identified as most relevant.
                Can be a boolean or an object with configuration.
            summary: Include page summaries. Can be a boolean or an object with configuration.
            livecrawl: Live crawl behavior: 'never' (cached only), 'fallback' (cache first,
                live if unavailable), 'preferred' (live first, cache fallback), 'always'.
            subpages: Number of subpages to crawl from the provided URLs.
            subpage_target: Keywords to target specific subpages (e.g., ['about', 'products']).

        Returns:
            JSON string with page contents.
        """
        try:
            from exa_py import Exa

            if not self.api_key:
                return json.dumps({"error": "EXA_API_KEY not configured"})

            exa = Exa(self.api_key)

            # Support both urls and deprecated ids parameter
            url_list = urls if urls is not None else (ids if ids else [])

            params: dict = {
                "urls": url_list,
                "text": text,
                "highlights": highlights,
                "summary": summary,
                "livecrawl": livecrawl,
                "subpages": subpages,
                "subpageTarget": subpage_target,
            }

            # Remove None, empty strings, and empty lists
            params = {
                k: v for k, v in params.items()
                if v is not None and v != "" and v != []
            }

            results = exa.get_contents(**params)

            parsed_results = []
            for result in results.results:
                result_dict = {"url": result.url}

                if result.title:
                    result_dict["title"] = result.title

                if result.text:
                    text_content = result.text[:self.text_length_limit]
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
