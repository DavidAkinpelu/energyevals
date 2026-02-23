import asyncio
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
    for tool discovery and execution.  Each server gets its own
    ``AsyncExitStack`` so that individual connections can be torn down
    and re-established without affecting other servers.
    """

    def __init__(
        self,
        servers: list[MCPServerConfig] | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ):
        """Initialize the MCP client.

        Args:
            servers: List of MCP server configurations.
            max_retries: Maximum reconnection attempts per tool call failure.
            retry_base_delay: Base delay in seconds between retries (doubles each attempt).
        """
        self.servers = servers or []
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

        self._sessions: dict[str, Any] = {}
        self._tools: dict[str, dict[str, Any]] = {}
        self._connected = False
        self._server_stacks: dict[str, AsyncExitStack] = {}
        self._server_configs: dict[str, MCPServerConfig] = {}

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
        for server in self.servers:
            try:
                await self._connect_server(server)
            except Exception as e:
                logger.error(f"Failed to connect to MCP server '{server.name}': {e}")
                raise

        self._connected = True

    async def _connect_server(self, server: MCPServerConfig) -> None:
        """Connect to a single MCP server and register its tools.

        Args:
            server: Server configuration.
        """
        stack = AsyncExitStack()
        await stack.__aenter__()

        try:
            if server.url:
                session = await self._open_sse(stack, server)
            else:
                session = await self._open_stdio(stack, server)
        except Exception:
            await stack.aclose()
            raise

        self._server_stacks[server.name] = stack
        self._server_configs[server.name] = server
        self._sessions[server.name] = session

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

    async def _open_stdio(self, stack: AsyncExitStack, server: MCPServerConfig) -> ClientSession:
        """Open a stdio transport and return an initialised session.

        Args:
            stack: The exit stack that owns the connection lifetime.
            server: Server configuration with command.
        """
        params = StdioServerParameters(
            command=server.command,
            args=server.args,
            env=server.env,
        )

        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def _open_sse(self, stack: AsyncExitStack, server: MCPServerConfig) -> ClientSession:
        """Open an SSE transport and return an initialised session.

        Args:
            stack: The exit stack that owns the connection lifetime.
            server: Server configuration with URL.
        """
        read, write = await stack.enter_async_context(sse_client(server.url))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def _reconnect_server(self, server_name: str) -> None:
        """Tear down and re-establish the connection for a single server.

        Only SSE servers are reconnected; stdio failures are not retried.

        Args:
            server_name: Name of the server to reconnect.

        Raises:
            ValueError: If the server is unknown or uses stdio transport.
            Exception: If the reconnection itself fails.
        """
        config = self._server_configs.get(server_name)
        if not config:
            raise ValueError(f"No config stored for server '{server_name}'")
        if not config.url:
            raise ValueError(f"Reconnection is only supported for SSE servers ('{server_name}' uses stdio)")

        old_stack = self._server_stacks.pop(server_name, None)
        if old_stack:
            try:
                await old_stack.aclose()
            except Exception as e:
                logger.debug(f"Ignored error closing old stack for '{server_name}': {e}")

        self._sessions.pop(server_name, None)
        old_tool_names = [name for name, info in self._tools.items() if info["server"] == server_name]
        for name in old_tool_names:
            del self._tools[name]

        logger.warning(f"Reconnecting to MCP server '{server_name}' ...")
        await self._connect_server(config)

    async def disconnect(self) -> None:
        """Disconnect from all MCP servers."""
        for name, stack in reversed(list(self._server_stacks.items())):
            try:
                await stack.aclose()
            except Exception as e:
                logger.warning(f"Error disconnecting from '{name}': {e}")

        self._server_stacks.clear()
        self._server_configs.clear()
        self._sessions.clear()
        self._tools.clear()
        self._connected = False
        logger.info("Disconnected from all MCP servers")

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

        On transient failures the client will attempt to reconnect to the
        server and retry up to ``max_retries`` times with exponential backoff.

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
        server_config = self._server_configs.get(server_name)
        is_sse = server_config is not None and server_config.url is not None

        last_error: Exception | None = None
        attempts = 1 + (self.max_retries if is_sse else 0)

        for attempt in range(attempts):
            session = self._sessions.get(server_name)
            if not session:
                if attempt == 0:
                    return json.dumps({"error": f"Server not connected: {server_name}"})
                break

            try:
                result = await session.call_tool(tool_name, arguments)

                if result.content:
                    for content in result.content:
                        if hasattr(content, "text"):
                            return str(content.text)

                return json.dumps({"result": "Tool executed successfully"})

            except Exception as e:
                last_error = e
                if not is_sse or attempt >= self.max_retries:
                    break

                delay = self.retry_base_delay * (2 ** attempt)
                logger.warning(
                    f"Tool call '{tool_name}' failed (attempt {attempt + 1}/{attempts}), "
                    f"reconnecting to '{server_name}' in {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)

                try:
                    await self._reconnect_server(server_name)
                except Exception as re_err:
                    logger.error(f"Reconnection to '{server_name}' failed: {re_err}")
                    break

        logger.error(f"Tool call failed: {tool_name} - {last_error}")
        return json.dumps({"error": str(last_error), "tool": tool_name})

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
