from .base_tool import BaseTool, ToolRegistry, ToolResult
from .battery_tool import BatteryOptimizationTool
from .dockets import (
    DCDocketTool,
    FERCDocketTool,
    MarylandDocketTool,
    NewYorkDocketTool,
    NorthCarolinaDocketTool,
    SouthCarolinaDocketTool,
    TexasDocketTool,
    VirginiaDocketTool,
)
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
    "SearchTool",
    "GridStatusAPITool",
    "TariffsTool",
    "RenewablesTool",
    "BatteryOptimizationTool",
    "OpenWeatherTool",
    "SystemTool",
    "DCDocketTool",
    "FERCDocketTool",
    "MarylandDocketTool",
    "NewYorkDocketTool",
    "NorthCarolinaDocketTool",
    "SouthCarolinaDocketTool",
    "TexasDocketTool",
    "VirginiaDocketTool",
]


def create_default_registry() -> ToolRegistry:
    """Create a tool registry with all default tools registered.

    Returns:
        ToolRegistry with standard energy analytics tools.
    """
    registry = ToolRegistry()

    registry.register(SearchTool())
    registry.register(GridStatusAPITool())
    registry.register(TariffsTool())
    registry.register(RenewablesTool())
    registry.register(BatteryOptimizationTool())
    registry.register(OpenWeatherTool())
    registry.register(SystemTool())

    registry.register(FERCDocketTool())
    registry.register(MarylandDocketTool())
    registry.register(TexasDocketTool())
    registry.register(NewYorkDocketTool())
    registry.register(NorthCarolinaDocketTool())
    registry.register(SouthCarolinaDocketTool())
    registry.register(VirginiaDocketTool())
    registry.register(DCDocketTool())

    return registry
