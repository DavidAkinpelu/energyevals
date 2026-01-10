"""Base tool class for energBench standard tools."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loguru import logger

from energbench.agent.providers import ToolDefinition


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    data: Any
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(
            {
                "success": self.success,
                "data": self.data,
                "error": self.error,
                "metadata": self.metadata,
            },
            indent=2,
            default=str,
        )


class BaseTool(ABC):
    """Abstract base class for standard tools.

    All tools should inherit from this class and implement the required methods.
    """

    def __init__(self, name: str, description: str):
        """Initialize the tool.

        Args:
            name: Unique name for the tool.
            description: Description of what the tool does.
        """
        self.name = name
        self.description = description
        self._methods: dict[str, Callable] = {}

    @abstractmethod
    def get_tools(self) -> list[ToolDefinition]:
        """Return list of tool definitions for this tool.

        Each tool can expose multiple methods as separate tools.
        """
        pass

    def register_method(self, name: str, method: Callable):
        """Register a method as a callable tool.

        Args:
            name: Name of the tool method.
            method: The callable method.
        """
        self._methods[name] = method
        logger.debug(f"Registered tool method: {name}")

    async def execute(self, method_name: str, **kwargs) -> ToolResult:
        """Execute a tool method.

        Args:
            method_name: Name of the method to execute.
            **kwargs: Arguments for the method.

        Returns:
            ToolResult with the execution outcome.
        """
        if method_name not in self._methods:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown method: {method_name}",
            )

        method = self._methods[method_name]

        try:
            result = method(**kwargs)

            # Handle async methods
            if hasattr(result, "__await__"):
                result = await result

            return ToolResult(
                success=True,
                data=result,
                metadata={"method": method_name},
            )

        except Exception as e:
            logger.error(f"Tool execution failed: {method_name} - {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                metadata={"method": method_name},
            )


class ToolRegistry:
    """Registry for managing multiple tools."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._method_to_tool: dict[str, str] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool with the registry."""
        self._tools[tool.name] = tool

        for tool_def in tool.get_tools():
            self._method_to_tool[tool_def.name] = tool.name

        logger.info(f"Registered tool: {tool.name}")

    def get_all_tools(self) -> list[ToolDefinition]:
        """Get all tool definitions from all registered tools."""
        all_tools = []
        for tool in self._tools.values():
            all_tools.extend(tool.get_tools())
        return all_tools

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool method to execute.
            **kwargs: Arguments for the tool.

        Returns:
            ToolResult with the execution outcome.
        """
        if tool_name not in self._method_to_tool:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}",
            )

        parent_tool_name = self._method_to_tool[tool_name]
        tool = self._tools[parent_tool_name]

        return await tool.execute(tool_name, **kwargs)

    def get_executor(self) -> Callable:
        """Get a tool executor function for use with ReActAgent."""

        async def executor(tool_name: str, arguments: dict[str, Any]) -> str:
            result = await self.execute(tool_name, **arguments)
            return result.to_json()

        return executor
