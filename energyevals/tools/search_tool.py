import json
import os
from typing import Literal

from exa_py import Exa
from loguru import logger

from .base_tool import BaseTool, tool_method
from .constants import SEARCH_MAX_RESULTS, SEARCH_TEXT_LIMIT


class SearchTool(BaseTool):
    """Web search tool using Exa API.

    Provides semantic web search capabilities for finding relevant
    information about energy markets, regulations, and technical topics.
    """

    def __init__(
        self,
        api_key: str | None = None,
        text_length_limit: int = SEARCH_TEXT_LIMIT,
        default_num_results: int = SEARCH_MAX_RESULTS,
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

    @tool_method(name="search_web")
    def search(
        self,
        query: str,
        num_results: int | None = None,
        text: bool = True,
        highlights: bool = True,
        summary: bool = False,
        livecrawl: Literal["never", "fallback", "preferred", "always"] = "fallback",
        search_type: Literal["neural", "fast", "auto", "deep"] = "auto",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> str:
        """Search the web using Exa for relevant information about energy markets, regulations, companies, and technical topics.
        Returns titles, URLs, and text snippets from matching pages.

        Args:
            query: The search query describing what you're looking for.
            num_results: Number of results to return.
            text: Include text content in results.
            highlights: Include highlights in results.
            summary: Include Exa summaries when available.
            livecrawl: Live crawl behavior ("never", "fallback", "preferred", "always").
            search_type: Search method. "neural" uses embeddings for semantic similarity, "fast" is
                keyword-based, "auto" (default) combines both methods, "deep" performs comprehensive
                search with query expansion.
            include_domains: Restrict results to these domains.
            exclude_domains: Exclude results from these domains.

        Returns:
            JSON string with search results including query, num_results, and a list of result
            objects each containing url, title, author, published_date, text, and highlights.
        """
        try:
            if not self.api_key:
                return json.dumps({"error": "EXA_API_KEY not configured"})

            exa = Exa(self.api_key)

            search_kwargs = {
                "text": text,
                "highlights": highlights,
                "summary": summary,
                "num_results": num_results if num_results is not None else self.default_num_results,
                "livecrawl": livecrawl,
                "type": search_type,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
            }

            search_kwargs = {
                k: v for k, v in search_kwargs.items()
                if v is not None and v != "" and v != []
            }

            results = exa.search_and_contents(query, **search_kwargs)

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

    @tool_method(name="get_page_contents")
    def get_contents(
        self,
        urls: list[str],
        text: bool | dict = True,
        highlights: bool | dict = True,
        summary: bool | dict = False,
        livecrawl: Literal["never", "fallback", "preferred", "always"] = "fallback",
        subpages: int | None = None,
        subpage_target: list[str] | None = None,
    ) -> str:
        """Get the full page contents, summaries, and metadata for a list of URLs.
        Returns instant results from cache, with automatic live crawling as fallback for uncached pages.
        Use this after search_web to get more details from promising results.

        Args:
            urls: Array of URLs to crawl.
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
            if not self.api_key:
                return json.dumps({"error": "EXA_API_KEY not configured"})

            exa = Exa(self.api_key)

            params: dict = {
                "urls": urls or [],
                "text": text,
                "highlights": highlights,
                "summary": summary,
                "livecrawl": livecrawl,
                "subpages": subpages,
                "subpageTarget": subpage_target,
            }

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
