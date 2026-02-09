import json
import os
from unittest.mock import Mock

import pytest

from energbench.tools.search_tool import SearchTool


class TestSearchToolUnit:
    """Unit tests for SearchTool."""

    def test_init_with_api_key(self, monkeypatch):
        """Test initialization with API key."""
        monkeypatch.setenv("EXA_API_KEY", "test_key")
        tool = SearchTool()
        assert tool.name == "search"
        assert tool.api_key == "test_key"

    def test_init_without_api_key(self, monkeypatch):
        """Test initialization without API key."""
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        tool = SearchTool()
        assert tool.api_key is None

    def test_get_tools_definition(self):
        """Test tool definitions are properly structured."""
        tool = SearchTool(api_key="test_key")
        tools = tool.get_tools()

        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert "search_web" in tool_names
        assert "get_page_contents" in tool_names

    def test_search_success(self, mocker):
        """Test successful web search with mocked API."""
        mock_result = Mock()
        mock_result.url = "https://example.com/article1"
        mock_result.title = "Energy Storage Trends"
        mock_result.text = "Battery storage is growing rapidly in energy markets."
        mock_result.highlights = ["battery storage", "energy markets"]
        mock_result.published_date = "2024-01-15"
        mock_result.author = None

        mock_exa_instance = Mock()
        mock_exa_instance.search_and_contents.return_value = Mock(
            results=[mock_result]
        )

        mocker.patch("energbench.tools.search_tool.Exa", return_value=mock_exa_instance)

        tool = SearchTool(api_key="test_key")
        result = tool.search(query="energy storage", num_results=3)

        result_data = json.loads(result)
        assert "results" in result_data
        assert len(result_data["results"]) > 0
        assert result_data["results"][0]["url"] == "https://example.com/article1"
        assert "title" in result_data["results"][0]

    def test_search_with_params(self, mocker):
        """Test search passes parameters correctly."""
        mock_exa_instance = Mock()
        mock_exa_instance.search_and_contents.return_value = Mock(results=[])

        mocker.patch("energbench.tools.search_tool.Exa", return_value=mock_exa_instance)

        tool = SearchTool(api_key="test_key")
        tool.search(query="solar energy", num_results=5)

        mock_exa_instance.search_and_contents.assert_called_once()
        call_args = mock_exa_instance.search_and_contents.call_args
        assert call_args[0][0] == "solar energy"

    def test_search_error_handling(self, mocker):
        """Test search error handling."""
        mocker.patch(
            "energbench.tools.search_tool.Exa",
            side_effect=Exception("API connection failed"),
        )

        tool = SearchTool(api_key="test_key")
        result = tool.search(query="test", num_results=3)

        result_data = json.loads(result)
        assert "error" in result_data
        assert "API connection failed" in result_data["error"]

    def test_get_contents_success(self, mocker):
        """Test getting page contents with mocked API."""
        mock_result = Mock()
        mock_result.url = "https://example.com/page"
        mock_result.title = "Test Page"
        mock_result.text = "Full page content with detailed information."
        mock_result.highlights = ["detailed information"]

        mock_exa_instance = Mock()
        mock_exa_instance.get_contents.return_value = Mock(
            results=[mock_result]
        )

        mocker.patch("energbench.tools.search_tool.Exa", return_value=mock_exa_instance)

        tool = SearchTool(api_key="test_key")
        result = tool.get_contents(
            urls=["https://example.com/page"],
            text=True,
            highlights=True,
        )

        result_data = json.loads(result)
        assert "contents" in result_data
        assert len(result_data["contents"]) > 0
        assert "text" in result_data["contents"][0]

    def test_get_contents_text_only(self, mocker):
        """Test get_contents with text only (no highlights/summary)."""
        mock_result = Mock()
        mock_result.url = "https://example.com"
        mock_result.title = "Test"
        mock_result.text = "Content"
        mock_result.highlights = None

        mock_exa_instance = Mock()
        mock_exa_instance.get_contents.return_value = Mock(
            results=[mock_result]
        )

        mocker.patch("energbench.tools.search_tool.Exa", return_value=mock_exa_instance)

        tool = SearchTool(api_key="test_key")
        tool.get_contents(
            urls=["https://example.com"],
            text=True,
            highlights=False,
            summary=False,
        )

        mock_exa_instance.get_contents.assert_called_once()

    def test_get_contents_empty_urls(self):
        """Test get_contents with empty URL list."""
        tool = SearchTool(api_key="test_key")
        result = tool.get_contents(urls=[], text=True)

        result_data = json.loads(result)
        assert isinstance(result_data, dict)


@pytest.mark.integration
@pytest.mark.requires_api_key
class TestSearchToolIntegration:
    """Integration tests with real Exa API."""

    @pytest.mark.asyncio
    async def test_search_real_api(self):
        """Test search with real Exa API."""
        if not os.getenv("EXA_API_KEY"):
            pytest.skip("EXA_API_KEY not set")

        tool = SearchTool()
        result = tool.search(
            query="energy storage market trends",
            num_results=3,
        )

        result_data = json.loads(result)
        assert "results" in result_data or "error" in result_data

        if "results" in result_data:
            for item in result_data["results"]:
                assert "url" in item

    @pytest.mark.asyncio
    async def test_get_contents_real_api(self):
        """Test get_contents with real API."""
        if not os.getenv("EXA_API_KEY"):
            pytest.skip("EXA_API_KEY not set")

        tool = SearchTool()

        search_result = tool.search(query="energy market", num_results=1)
        search_data = json.loads(search_result)

        if "results" in search_data and search_data["results"]:
            url = search_data["results"][0]["url"]

            result = tool.get_contents(
                urls=[url],
                text=True,
                highlights=True,
                summary=False,
            )

            result_data = json.loads(result)
            assert isinstance(result_data, dict)
