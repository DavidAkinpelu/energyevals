#!/usr/bin/env python
"""Exercise SearchTool (Exa) methods."""
from __future__ import annotations

import json

from dotenv import load_dotenv

from energbench.tools.search_tool import SearchTool


def _print_result(label: str, result: str) -> None:
    print(f"\n=== {label} ===")
    print(result)


def _extract_first_url(result: str) -> str | None:
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    if not isinstance(first, dict):
        return None
    return first.get("url")


def main() -> None:
    load_dotenv()
    tool = SearchTool()

    search_result = tool.search(query="energy storage market trends", num_results=3)
    _print_result("search_web", search_result)

    url = _extract_first_url(search_result) or "https://www.example.com"
    contents_result = tool.get_contents(urls=[url], text=True, highlights=True, summary=False)
    _print_result("get_page_contents", contents_result)


if __name__ == "__main__":
    main()
