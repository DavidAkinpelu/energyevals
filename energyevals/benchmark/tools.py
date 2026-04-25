import json
import logging
from typing import Any

from energyevals.agent.exceptions import ToolExecutionError
from energyevals.agent.schema import ToolDefinition, ToolExecutor
from energyevals.agent.schema.tools import ToolResult
from energyevals.mcp.client import MCPClient
from energyevals.tools.base_tool import ToolRegistry

from .config import ToolsConfig
from .constants import TOOL_DESCRIPTION_PREVIEW_LENGTH
from .display import print_header

logger = logging.getLogger(__name__)


def _build_tool_groups(registry: ToolRegistry) -> dict[str, set[str]]:
    """Build a mapping of parent tool names to their method names from the registry."""
    groups: dict[str, set[str]] = {}
    for method_name, parent_name in registry._method_to_tool.items():
        groups.setdefault(parent_name, set()).add(method_name)
    return groups


def _expand_names(names: list[str], groups: dict[str, set[str]]) -> set[str]:
    """Expand a mix of group names and individual tool names into a flat set."""
    expanded: set[str] = set()
    for name in names:
        if name in groups:
            expanded |= groups[name]
        else:
            expanded.add(name)
    return expanded


def filter_tools(
    all_tools: list[ToolDefinition],
    config: ToolsConfig,
    registry: ToolRegistry | None = None,
) -> list[ToolDefinition]:
    """Filter tools based on include/exclude configuration.

    Both parent tool names (e.g. ``"search"``, ``"openweather"``) and individual
    tool method names (e.g. ``"search_web"``) are accepted in include/exclude
    lists.  Parent tool names are resolved dynamically from the *registry*.

    Args:
        all_tools: List of all available tools
        config: Tools configuration with include/exclude lists
        registry: Optional tool registry used to resolve group names.
            When ``None``, names are matched against individual method names only.

    Returns:
        Filtered list of tools
    """
    if not config.enabled:
        return []

    groups = _build_tool_groups(registry) if registry else {}

    if config.include:
        included_names = _expand_names(config.include, groups)
        tools = [t for t in all_tools if t.name in included_names]
        logger.info(f"Including only specified tools: {config.include} -> {included_names}")
    else:
        if config.exclude:
            excluded_names = _expand_names(config.exclude, groups)
            tools = [t for t in all_tools if t.name not in excluded_names]
            logger.info(f"Excluding tools: {config.exclude} -> {excluded_names}")
        else:
            tools = all_tools

    return tools


def merge_tools(
    std_tools: list[ToolDefinition],
    mcp_tools: list[ToolDefinition],
) -> list[ToolDefinition]:
    """Merge standard and MCP tools with MCP-first ordering and std-wins deduplication.

    Result ordering: [MCP-unique tools...] + [all standard tools...]

    On a name collision the standard tool is kept and the MCP version is dropped.
    A warning is logged for each collision so operators can identify unexpected overlaps.

    Args:
        std_tools: Standard tools (retained on name collision).
        mcp_tools: MCP tools (listed first; dropped on name collision with std).

    Returns:
        Deduplicated list with MCP-unique tools first, followed by all standard tools.
    """
    std_names = {t.name for t in std_tools}
    mcp_unique: list[ToolDefinition] = []
    for tool in mcp_tools:
        if tool.name in std_names:
            logger.warning(
                f"Tool name conflict: '{tool.name}' exists in both MCP and standard tools. "
                "Standard tool takes priority."
            )
        else:
            mcp_unique.append(tool)
    return mcp_unique + std_tools


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


def _wrap_mcp_result(raw: str) -> str:
    """Normalise raw MCP server output to the standard ToolResult JSON schema.

    MCP servers return arbitrary text or JSON.  This wrapper converts the
    raw output into the same ``{"success": ..., "data": ..., "error": ...,
    "metadata": {...}}`` envelope that standard tools emit so the agent
    always sees a consistent format regardless of tool origin.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Plain-text response — wrap as successful string data
        return ToolResult(success=True, data=raw).to_json()

    # Bare MCP error envelope: {"error": "...", "tool": "..."}
    if isinstance(parsed, dict) and "error" in parsed and len(parsed) <= 2:
        return ToolResult(success=False, data=None, error=str(parsed["error"])).to_json()

    return ToolResult(success=True, data=parsed).to_json()


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
            raw = await mcp_client.call_tool(tool_name, arguments)
            return _wrap_mcp_result(raw)
        raise ToolExecutionError(f"Unknown tool: {tool_name}", tool_name=tool_name)

    return executor
