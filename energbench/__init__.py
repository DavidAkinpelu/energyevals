"""energBench - AI agent evaluation framework for energy analytics.

This package provides:
- Custom ReAct agent with multi-provider support (OpenAI, Anthropic, DeepInfra)
- MCP servers for RAG and Database access
- Standard tools for energy analytics (search, grid status, tariffs, etc.)
- Evaluation framework with benchmarks and metrics
- Langfuse observability integration
"""

__version__ = "0.1.0"

from .agent import (
    AgentBuilder,
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
from .tools import (
    BatteryOptimizationTool,
    DocketTools,
    GridStatusAPITool,
    RenewablesTool,
    SearchTool,
    TariffsTool,
    ToolRegistry,
    create_default_registry,
)

# Lazy imports for optional modules
def __getattr__(name):
    """Lazy import for optional modules."""
    if name == "mcp":
        from . import mcp
        return mcp
    if name == "observability":
        from . import observability
        return observability
    if name == "LangfuseObserver":
        from .observability import LangfuseObserver
        return LangfuseObserver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Version
    "__version__",
    # Agent
    "ReActAgent",
    "AgentBuilder",
    "AgentRun",
    # Providers
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "DeepInfraProvider",
    "get_provider",
    # Types
    "Message",
    "ToolDefinition",
    # Tools
    "ToolRegistry",
    "create_default_registry",
    "SearchTool",
    "GridStatusAPITool",
    "TariffsTool",
    "RenewablesTool",
    "BatteryOptimizationTool",
    "DocketTools",
    # Submodules (lazy loaded)
    "mcp",
    "observability",
    "LangfuseObserver",
]
