import json
import logging
from typing import Any

from energbench.agent.schema import ToolDefinition, ToolExecutor
from energbench.mcp.client import MCPClient
from energbench.tools.base_tool import ToolRegistry

from .config import ToolsConfig
from .constants import TOOL_DESCRIPTION_PREVIEW_LENGTH
from .display import print_header

logger = logging.getLogger(__name__)


def filter_tools(all_tools: list[ToolDefinition], config: ToolsConfig) -> list[ToolDefinition]:
    """Filter tools based on include/exclude configuration.

    Args:
        all_tools: List of all available tools
        config: Tools configuration with include/exclude lists

    Returns:
        Filtered list of tools
    """
    if not config.enabled:
        return []

    if config.include:
        included_names = set(config.include)
        tools = [t for t in all_tools if t.name in included_names]
        logger.info(f"Including only specified tools: {config.include}")
    else:
        if config.exclude:
            excluded_names = set(config.exclude)
            tools = [t for t in all_tools if t.name not in excluded_names]
            logger.info(f"Excluding tools: {config.exclude}")
        else:
            tools = all_tools

    return tools


def list_tools(std_registry: ToolRegistry, mcp_client: MCPClient | None = None) -> int:
    """List all available tools.

    Args:
        std_registry: Standard tool registry
        mcp_client: Optional MCP client

    Returns:
        Exit code (0 for success)
    """
    print_header("Available Tools")

    print("\n  Standard Tools:")
    for tool in std_registry.get_all_tools():
        desc = (
            tool.description[:TOOL_DESCRIPTION_PREVIEW_LENGTH] + "..."
            if len(tool.description) > TOOL_DESCRIPTION_PREVIEW_LENGTH
            else tool.description
        )
        print(f"    - {tool.name}: {desc}")

    if mcp_client:
        print("\n  MCP Tools:")
        for tool in mcp_client.list_tools():
            desc = (
                tool.description[:TOOL_DESCRIPTION_PREVIEW_LENGTH] + "..."
                if len(tool.description) > TOOL_DESCRIPTION_PREVIEW_LENGTH
                else tool.description
            )
            print(f"    - {tool.name}: {desc}")

    return 0


def build_tool_executor(
    std_registry: ToolRegistry,
    mcp_client: MCPClient | None = None,
) -> ToolExecutor:
    """Build a tool executor that combines standard tools and MCP tools."""
    std_tools = {tool.name for tool in std_registry.get_all_tools()}
    mcp_tools = {tool.name for tool in mcp_client.list_tools()} if mcp_client else set()

    async def executor(tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name in std_tools:
            result = await std_registry.execute(tool_name, **arguments)
            return result.to_json()
        if mcp_client and tool_name in mcp_tools:
            return await mcp_client.call_tool(tool_name, arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    return executor
