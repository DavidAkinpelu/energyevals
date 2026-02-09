import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Optional

from loguru import logger

from energbench.agent.providers import ToolDefinition
from energbench.core.errors import APIError, ToolError


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    data: Any
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
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
        self._methods: dict[str, Callable[..., Any]] = {}

    @abstractmethod
    def get_tools(self) -> list[ToolDefinition]:
        """Return list of tool definitions for this tool.

        Each tool can expose multiple methods as separate tools.
        """
        pass

    def register_method(self, name: str, method: Callable[..., Any]) -> None:
        """Register a method as a callable tool.

        Args:
            name: Name of the tool method.
            method: The callable method.
        """
        self._methods[name] = method
        logger.debug(f"Registered tool method: {name}")

    async def execute(self, method_name: str, **kwargs: Any) -> ToolResult:
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

            if hasattr(result, "__await__"):
                result = await result

            return ToolResult(
                success=True,
                data=result,
                metadata={"method": method_name},
            )

        except ToolError as e:
            logger.warning(f"Tool error in {method_name}: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                metadata={"method": method_name, "tool_name": e.tool_name, "recoverable": e.recoverable},
            )

        except APIError as e:
            logger.warning(f"API error in {method_name}: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                metadata={
                    "method": method_name,
                    "tool_name": e.tool_name,
                    "status_code": e.status_code,
                    "recoverable": e.recoverable,
                },
            )

        except Exception as e:
            logger.error(f"Unexpected error in {method_name}: {type(e).__name__}: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Unexpected error: {type(e).__name__}: {str(e)}",
                metadata={"method": method_name, "error_type": type(e).__name__},
            )


class ToolRegistry:
    """Registry for managing multiple tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._method_to_tool: dict[str, str] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

        for tool_def in tool.get_tools():
            self._method_to_tool[tool_def.name] = tool.name

        logger.debug(f"Registered tool: {tool.name}")

    @classmethod
    def discover_tools(cls) -> "ToolRegistry":
        """Auto-discover and register tools from entry points.

        Discovers tools registered via setuptools entry points under the
        'energbench.tools' group.

        Returns:
            ToolRegistry with all discovered tools registered
        """


        registry = cls()

        try:
            discovered: Any = entry_points(group="energbench.tools")  # type: ignore[call-arg]
        except TypeError:
            all_eps: Any = entry_points()
            discovered = all_eps.get("energbench.tools", []) if isinstance(all_eps, dict) else []

        for ep in discovered:
            try:
                tool_cls = ep.load()
                tool = tool_cls()
                registry.register(tool)
                logger.info(f"Discovered and registered tool: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed to load tool '{ep.name}': {e}")

        return registry

    def get_all_tools(self) -> list[ToolDefinition]:
        all_tools = []
        for tool in self._tools.values():
            all_tools.extend(tool.get_tools())
        return all_tools

    async def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
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

    def get_executor(self) -> Callable[..., Any]:
        """Get a tool executor function for use with ReActAgent."""

        async def executor(tool_name: str, arguments: dict[str, Any]) -> str:
            result = await self.execute(tool_name, **arguments)
            return result.to_json()

        return executor
