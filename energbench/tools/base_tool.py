import inspect
import re as _re
import types
from abc import ABC
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from loguru import logger

from energbench.agent.providers import ToolDefinition
from energbench.agent.schema.tools import ToolResult
from energbench.core.errors import APIError, ToolError

# ---------------------------------------------------------------------------
# @tool_method decorator
# ---------------------------------------------------------------------------

def tool_method(
    name: str | None = None,
    *,
    parameters: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Mark a method as an exposed tool.

    The method's docstring (summary paragraph before ``Args:`` / ``Returns:``)
    is used as the LLM-facing tool description so there is a single source of
    truth.  When *parameters* is omitted the JSON Schema is auto-generated from
    the method's type hints and docstring ``Args:``/``Parameters:`` section.

    Args:
        name: Tool name exposed to the LLM. Defaults to the method name.
        parameters: Explicit JSON Schema for the tool's accepted parameters.
            When ``None`` (the default) the schema is built automatically from
            the decorated method's signature and docstring.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func._tool_metadata = {  # type: ignore[attr-defined]
            "name": name or func.__name__,
            "parameters": parameters,
        }
        return func

    return decorator


# ---------------------------------------------------------------------------
# BaseTool
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    """Abstract base class for standard tools.

    Subclasses decorate callable methods with :func:`tool_method` to expose
    them as LLM tools.  ``get_tools()`` and method registration are handled
    automatically; subclasses only need ``__init__`` and the decorated methods.
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
        self._auto_register_tool_methods()

    # -- auto-discovery of @tool_method decorated methods -------------------

    def _auto_register_tool_methods(self) -> None:
        """Scan for ``@tool_method``-decorated methods and register them."""
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr = getattr(self, attr_name, None)
            if callable(attr) and hasattr(attr, "_tool_metadata"):
                tool_name: str = attr._tool_metadata["name"]
                self._methods[tool_name] = attr
                logger.debug(f"Auto-registered tool method: {tool_name}")

    # -- docstring → description helper -------------------------------------

    @staticmethod
    def _get_method_description(method: Callable[..., Any]) -> str:
        """Extract the summary paragraph from *method*'s docstring.

        Everything before the first ``Args:``, ``Returns:``, ``Parameters:``,
        ``Raises:``, or ``**`` heading is treated as the summary.
        """
        doc = method.__doc__
        if not doc:
            raise ValueError(
                f"Method {getattr(method, '__name__', method)} has no docstring; "
                "a docstring is required for @tool_method methods."
            )
        lines = doc.strip().splitlines()
        summary_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith(
                ("args:", "returns:", "parameters:", "raises:", "**")
            ):
                break
            if not stripped and summary_lines:
                break
            if stripped:
                summary_lines.append(stripped)
        return " ".join(summary_lines)

    # -- docstring Args → per-parameter descriptions -------------------------

    @staticmethod
    def _parse_docstring_args(method: Callable[..., Any]) -> dict[str, str]:
        """Extract per-parameter descriptions from the ``Args:``/``Parameters:`` section."""
        doc = getattr(method, "__doc__", None) or ""
        lines = doc.strip().splitlines()

        in_args = False
        args_dict: dict[str, str] = {}
        current_param: str | None = None
        current_desc: list[str] = []

        for line in lines:
            stripped = line.strip()

            if stripped.lower().startswith(("args:", "parameters:")):
                in_args = True
                continue

            if in_args and stripped.lower().startswith(("returns:", "raises:", "**")):
                if current_param:
                    args_dict[current_param] = " ".join(current_desc).strip()
                break

            if not in_args:
                continue

            if not stripped:
                if current_param:
                    args_dict[current_param] = " ".join(current_desc).strip()
                    current_param = None
                    current_desc = []
                continue

            param_match = _re.match(r"^(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)", stripped)
            if param_match:
                if current_param:
                    args_dict[current_param] = " ".join(current_desc).strip()
                current_param = param_match.group(1)
                current_desc = [param_match.group(2)] if param_match.group(2) else []
            elif current_param:
                current_desc.append(stripped)

        if current_param:
            args_dict[current_param] = " ".join(current_desc).strip()

        return args_dict

    # -- Python type → JSON Schema mapping -----------------------------------

    @staticmethod
    def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
        """Map a Python type annotation to a JSON Schema fragment."""
        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is Literal:
            values = list(args)
            if all(isinstance(v, bool) for v in values):
                return {"type": "boolean", "enum": values}
            if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
                return {"type": "integer", "enum": values}
            if all(isinstance(v, str) for v in values):
                return {"type": "string", "enum": values}
            if all(isinstance(v, (int, float)) for v in values):
                return {"type": "number", "enum": values}
            return {"enum": values}

        if origin is Union or isinstance(annotation, types.UnionType):
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return BaseTool._python_type_to_json_schema(non_none[0])
            return {"oneOf": [BaseTool._python_type_to_json_schema(a) for a in non_none]}

        if origin is list:
            if args:
                return {"type": "array", "items": BaseTool._python_type_to_json_schema(args[0])}
            return {"type": "array"}

        _type_map: dict[type, dict[str, str]] = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            dict: {"type": "object"},
        }
        return dict(_type_map.get(annotation, {"type": "string"}))

    # -- auto-generate full parameters schema --------------------------------

    @staticmethod
    def _build_parameters_schema(method: Callable[..., Any]) -> dict[str, Any]:
        """Build a JSON Schema ``parameters`` object from *method*'s signature and docstring.

        * ``inspect.signature``  — parameter names, defaults, required-ness
        * ``get_type_hints``     — Python types → JSON Schema types
        * docstring ``Args:``    — per-parameter descriptions
        """
        func = getattr(method, "__func__", method)

        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}
        hints.pop("self", None)
        hints.pop("return", None)

        sig = inspect.signature(method)
        doc_args = BaseTool._parse_docstring_args(method)

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            annotation = hints.get(param_name, param.annotation)
            has_default = param.default is not inspect.Parameter.empty

            if annotation is not inspect.Parameter.empty:
                prop_schema = BaseTool._python_type_to_json_schema(annotation)
            else:
                prop_schema = {"type": "string"}

            if param_name in doc_args:
                prop_schema["description"] = doc_args[param_name]

            if has_default and param.default is not None:
                prop_schema["default"] = param.default

            properties[param_name] = prop_schema

            if not has_default:
                required.append(param_name)

        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    # -- build ToolDefinition list -------------------------------------------

    def get_tools(self) -> list[ToolDefinition]:
        """Build tool definitions from ``@tool_method``-decorated methods."""
        tools: list[ToolDefinition] = []
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr = getattr(self, attr_name, None)
            if callable(attr) and hasattr(attr, "_tool_metadata"):
                metadata: dict[str, Any] = attr._tool_metadata
                params = metadata["parameters"]
                if params is None:
                    params = self._build_parameters_schema(attr)
                tools.append(
                    ToolDefinition(
                        name=metadata["name"],
                        description=self._get_method_description(attr),
                        parameters=params,
                    )
                )
        return tools

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
