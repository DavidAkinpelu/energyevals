"""Standard tools for energBench."""

from .base_tool import BaseTool, ToolRegistry, ToolResult
from .battery_tool import BatteryOptimizationTool
from .docket_tools import DocketTools
from .gridstatus_tool import GridStatusAPITool
from .openweather_tool import OpenWeatherTool
from .renewables_tool import RenewablesTool
from .search_tool import SearchTool
from .system_tool import SystemTool
from .tariffs_tool import TariffsTool

__all__ = [
    # Base
    "BaseTool",
    "ToolRegistry",
    "ToolResult",
    # Tools
    "SearchTool",
    "GridStatusAPITool",
    "TariffsTool",
    "RenewablesTool",
    "BatteryOptimizationTool",
    "DocketTools",
    "OpenWeatherTool",
    "SystemTool",
]


def create_default_registry() -> ToolRegistry:
    """Create a tool registry with all default tools registered.

    Returns:
        ToolRegistry with standard energy analytics tools.
    """
    registry = ToolRegistry()

    # Register all tools
    registry.register(SearchTool())
    registry.register(GridStatusAPITool())
    registry.register(TariffsTool())
    registry.register(RenewablesTool())
    registry.register(BatteryOptimizationTool())
    registry.register(DocketTools())
    registry.register(OpenWeatherTool())
    registry.register(SystemTool())

    return registry
