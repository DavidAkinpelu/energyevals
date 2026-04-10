from typing import Any  # noqa: E402

from .agent import (
    AgentRun,
    AnthropicProvider,
    BaseProvider,
    DeepInfraProvider,
    Message,
    OpenAIProvider,
    ReActAgent,
    ToolDefinition,
    get_provider,
)
from .core import (
    APIError,
    ConfigurationError,
    EnergyEvalsError,
    PathLike,
    ProviderError,
    ToolError,
    ensure_path,
)
from .tools import (
    BatteryOptimizationTool,
    DCDocketTool,
    FERCDocketTool,
    GridStatusAPITool,
    MarylandDocketTool,
    NewYorkDocketTool,
    NorthCarolinaDocketTool,
    RenewablesTool,
    SearchTool,
    SouthCarolinaDocketTool,
    TariffsTool,
    TexasDocketTool,
    ToolRegistry,
    VirginiaDocketTool,
    create_default_registry,
)


def __getattr__(name: str) -> Any:
    """Lazy import for optional modules."""
    if name == "mcp":
        from . import mcp
        return mcp
    if name == "observability":
        from . import observability
        return observability
    if name == "utils":
        from . import utils
        return utils
    if name == "benchmark":
        from . import benchmark
        return benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EnergyEvalsError",
    "ToolError",
    "APIError",
    "ProviderError",
    "ConfigurationError",
    "PathLike",
    "ensure_path",
    "ReActAgent",
    "AgentRun",
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "DeepInfraProvider",
    "get_provider",
    "Message",
    "ToolDefinition",
    "ToolRegistry",
    "create_default_registry",
    "SearchTool",
    "GridStatusAPITool",
    "TariffsTool",
    "RenewablesTool",
    "BatteryOptimizationTool",
    "DCDocketTool",
    "FERCDocketTool",
    "MarylandDocketTool",
    "NewYorkDocketTool",
    "NorthCarolinaDocketTool",
    "SouthCarolinaDocketTool",
    "TexasDocketTool",
    "VirginiaDocketTool",
    "mcp",
    "observability",
    "utils",
    "benchmark",
]
