"""MCP module for energBench.

Provides MCP servers for RAG and Database access, plus a client
for connecting to MCP servers from the ReAct agent.
"""

from .client import (
    MCPClient,
    MCPServerConfig,
    MCPToolAdapter,
    create_mcp_client,
    get_default_mcp_servers,
)

__all__ = [
    "MCPClient",
    "MCPServerConfig",
    "MCPToolAdapter",
    "create_mcp_client",
    "get_default_mcp_servers",
]
