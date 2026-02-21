import json
import os
from contextlib import AsyncExitStack
from typing import Any

from loguru import logger
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

from energbench.agent.providers import ToolDefinition
from energbench.agent.schema import MCPServerConfig


class MCPClient:
    """Client for connecting to multiple MCP servers.

    Manages connections to MCP servers and provides a unified interface
    for tool discovery and execution.
    """

    def __init__(self, servers: list[MCPServerConfig] | None = None):
        """Initialize the MCP client.

        Args:
            servers: List of MCP server configurations.
        """
        self.servers = servers or []
        self._sessions: dict[str, Any] = {}
        self._tools: dict[str, dict[str, Any]] = {}
        self._connected = False
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Connect to all configured MCP servers.

        Supports both stdio (local) and SSE (remote) transports.
        """
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for server in self.servers:
            try:
                if server.url:
                    await self._connect_sse(server)
                else:
                    await self._connect_stdio(server)

                session = self._sessions.get(server.name)
                if session:
                    tools_result = await session.list_tools()
                    for tool in tools_result.tools:
                        self._tools[tool.name] = {
                            "server": server.name,
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        }

                    logger.info(
                        f"Connected to MCP server '{server.name}' "
                        f"({'SSE' if server.url else 'stdio'}) "
                        f"with {len(tools_result.tools)} tools"
                    )

            except Exception as e:
                logger.error(f"Failed to connect to MCP server '{server.name}': {e}")
                raise

        self._connected = True

    async def _connect_stdio(self, server: MCPServerConfig) -> None:
        """Connect to a local MCP server via stdio.

        Args:
            server: Server configuration with command.
        """
        params = StdioServerParameters(
            command=server.command,
            args=server.args,
            env=server.env,
        )

        assert self._exit_stack is not None
        read, write = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )

        session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()

        self._sessions[server.name] = session

    async def _connect_sse(self, server: MCPServerConfig) -> None:
        """Connect to a remote MCP server via SSE.

        Args:
            server: Server configuration with URL.
        """
        assert self._exit_stack is not None
        read, write = await self._exit_stack.enter_async_context(
            sse_client(server.url)
        )

        session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()

        self._sessions[server.name] = session

    async def disconnect(self) -> None:
        """Disconnect from all MCP servers."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
                logger.info("Disconnected from all MCP servers")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
            finally:
                self._exit_stack = None

        self._sessions.clear()
        self._tools.clear()
        self._connected = False

    def list_tools(self) -> list[ToolDefinition]:
        """Get all available tools from connected servers.

        Returns:
            List of tool definitions.
        """
        return [
            ToolDefinition(
                name=info["name"],
                description=info["description"],
                parameters=info["parameters"],
            )
            for info in self._tools.values()
        ]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the appropriate MCP server.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments for the tool.

        Returns:
            JSON string with the tool result.
        """
        if tool_name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        tool_info = self._tools[tool_name]
        server_name = tool_info["server"]
        session = self._sessions.get(server_name)

        if not session:
            return json.dumps({"error": f"Server not connected: {server_name}"})

        try:
            result = await session.call_tool(tool_name, arguments)

            if result.content:
                for content in result.content:
                    if hasattr(content, "text"):
                        return str(content.text)

            return json.dumps({"result": "Tool executed successfully"})

        except Exception as e:
            logger.error(f"Tool call failed: {tool_name} - {e}")
            return json.dumps({"error": str(e), "tool": tool_name})

    def get_executor(self) -> Any:
        """Get a tool executor function for use with ReActAgent.

        Returns:
            Async function that executes tools by name.
        """

        async def executor(tool_name: str, arguments: dict[str, Any]) -> str:
            return await self.call_tool(tool_name, arguments)

        return executor

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected to any servers."""
        return self._connected and len(self._sessions) > 0


class MCPToolAdapter:
    """Adapter for using MCP tools with the standard tool interface.

    This allows MCP tools to be used alongside standard tools in the
    same agent configuration.
    """

    def __init__(self, client: MCPClient):
        """Initialize the adapter.

        Args:
            client: Connected MCP client.
        """
        self.client = client

    def get_tools(self) -> list[ToolDefinition]:
        """Get tool definitions from MCP servers."""
        return self.client.list_tools()

    async def execute(self, tool_name: str, **kwargs: Any) -> str:
        """Execute an MCP tool.

        Args:
            tool_name: Name of the tool.
            **kwargs: Tool arguments.

        Returns:
            Tool result as JSON string.
        """
        return await self.client.call_tool(tool_name, kwargs)


def get_default_mcp_servers() -> list[MCPServerConfig]:
    """Get default MCP server configurations.

    Checks for remote URLs in environment variables first, then falls back to local commands.

    Environment variables:
        - RAG_SERVER_URL: URL for remote RAG server (e.g., https://example.com/sse)
        - DATABASE_SERVER_URL: URL for remote Database server

    Returns:
        List of MCP server configs for RAG and Database servers.
    """
    rag_url = os.getenv("RAG_SERVER_URL")
    db_url = os.getenv("DATABASE_SERVER_URL")

    servers = []

    if rag_url:
        servers.append(
            MCPServerConfig(
                name="energy-rag",
                url=rag_url,
                description="Energy document retrieval and search (remote)",
            )
        )


    if db_url:
        servers.append(
            MCPServerConfig(
                name="energy-database",
                url=db_url,
                description="Energy market database queries (remote)",
            )
        )

    return servers


async def create_mcp_client(
    servers: list[MCPServerConfig] | None = None,
) -> MCPClient:
    """Create and connect an MCP client.

    Args:
        servers: Server configurations. Uses defaults if not provided.

    Returns:
        Connected MCP client.

    Note:
        The caller is responsible for calling client.disconnect() when done,
        or using the client as an async context manager.
    """
    if servers is None:
        servers = get_default_mcp_servers()

    client = MCPClient(servers)
    await client.connect()
    return client
