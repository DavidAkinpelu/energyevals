# MCP Client Usage Guide

The `MCPClient` supports connecting to MCP servers using both local (stdio) and remote (SSE/URL) transports.

## Quick Start

### Local Servers (Default)

```python
from energbench.mcp import create_mcp_client

# Uses local servers by default
client = await create_mcp_client()

# List available tools
tools = client.list_tools()

# Call a tool
result = await client.call_tool("search_documents", {
    "query": "battery storage",
    "market": "ERCOT",
    "num_documents": 5
})

# Clean up
await client.disconnect()
```

### Remote Servers via Environment Variables

Set environment variables to use deployed servers:

```bash
# .env file
RAG_SERVER_URL=https://energy-rag-server-production.up.railway.app/sse
DATABASE_SERVER_URL=https://energy-database-server-production.up.railway.app/sse
```

Then use the same code as above - it will automatically detect and use the URLs.

### Explicit Configuration

#### Remote Servers Only

```python
from energbench.mcp import MCPServerConfig, MCPClient

servers = [
    MCPServerConfig(
        name="energy-rag",
        url="https://your-rag-server.com/sse",
        description="Remote RAG server",
    ),
    MCPServerConfig(
        name="energy-database",
        url="https://your-db-server.com/sse",
        description="Remote Database server",
    ),
]

client = MCPClient(servers)
await client.connect()
```

#### Local Servers Only

```python
from energbench.mcp import MCPServerConfig, MCPClient

servers = [
    MCPServerConfig(
        name="energy-rag",
        command="energy-rag-server",
        description="Local RAG server",
    ),
    MCPServerConfig(
        name="energy-database",
        command="energy-database-server",
        description="Local Database server",
    ),
]

client = MCPClient(servers)
await client.connect()
```

#### Mixed Local and Remote

```python
servers = [
    # Local RAG server
    MCPServerConfig(
        name="energy-rag",
        command="energy-rag-server",
        description="Local RAG server",
    ),
    # Remote Database server
    MCPServerConfig(
        name="energy-database",
        url="https://your-db-server.com/sse",
        description="Remote Database server",
    ),
]

client = MCPClient(servers)
await client.connect()
```

## Configuration Options

### MCPServerConfig

The `MCPServerConfig` dataclass supports the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier for the server |
| `command` | `str` | Conditional* | Command to run local server (e.g., `"energy-rag-server"`) |
| `url` | `str` | Conditional* | URL for remote server (e.g., `"https://example.com/sse"`) |
| `args` | `list[str]` | No | Command-line arguments (only for local servers) |
| `env` | `dict[str, str]` | No | Environment variables (only for local servers) |
| `description` | `str` | No | Human-readable description |

\* Either `command` or `url` must be provided (but not both).

### Environment Variables

The `get_default_mcp_servers()` function checks these environment variables:

- `RAG_SERVER_URL`: URL for remote RAG server
- `DATABASE_SERVER_URL`: URL for remote Database server

If not set, it falls back to local servers using commands:
- `energy-rag-server`
- `energy-database-server`

## Transport Types

### stdio (Local Servers)

- **When**: Server runs as a subprocess on the same machine
- **How**: Communicates via stdin/stdout
- **Requires**: Server command available in PATH or virtualenv
- **Config**: Provide `command` field in `MCPServerConfig`

### SSE (Remote Servers)

- **When**: Server is deployed remotely (Railway, Cloud, etc.)
- **How**: Communicates via HTTP/SSE
- **Requires**: Server URL accessible over network
- **Config**: Provide `url` field in `MCPServerConfig`

## Examples

See `examples/mcp_client_example.py` for complete working examples.

## Deployment

For instructions on deploying MCP servers to Railway or other platforms, see:
- `mcp-servers/DEPLOYMENT_GUIDE.md` - Comprehensive deployment guide
- `mcp-servers/rag-server/README.md` - RAG server specific docs
- `mcp-servers/database-server/README.md` - Database server specific docs

## Troubleshooting

### Connection Issues

**Local servers:**
- Ensure servers are installed: `pip install ./mcp-servers/rag-server ./mcp-servers/database-server`
- Verify commands are available: `which energy-rag-server`
- Check environment variables are set in `.env` file

**Remote servers:**
- Verify URL is accessible: `curl https://your-server.com/health`
- Check server logs for connection errors
- Ensure firewalls/security groups allow connections

### Tool Discovery

If tools are not appearing:
- Check connection logs for errors
- Verify server is running correctly
- For remote servers, check the `/health` endpoint

### Tool Execution

If tool calls fail:
- Check tool arguments match the expected schema
- Review server logs for execution errors
- Ensure required credentials (API keys, DB passwords) are configured on the server

## Integration with ReAct Agent

```python
from energbench.mcp import create_mcp_client
from energbench.agent import ReActAgent

# Create MCP client
mcp_client = await create_mcp_client()

# Get tools
tools = mcp_client.list_tools()

# Create agent with MCP tools
agent = ReActAgent(
    provider=your_provider,
    tools=tools,
    tool_executor=mcp_client.get_executor(),
)

# Use the agent
response = await agent.run("What's the battery storage revenue in ERCOT?")
```

## API Reference

### MCPClient

#### Methods

- `__init__(servers: Optional[list[MCPServerConfig]])` - Initialize client
- `async connect()` - Connect to all configured servers
- `async disconnect()` - Disconnect from all servers
- `list_tools() -> list[ToolDefinition]` - Get available tools
- `async call_tool(tool_name: str, arguments: dict) -> str` - Execute a tool
- `get_executor()` - Get executor function for ReActAgent

#### Properties

- `is_connected: bool` - Check if client is connected

### create_mcp_client

```python
async def create_mcp_client(
    servers: Optional[list[MCPServerConfig]] = None
) -> MCPClient
```

Convenience function that creates and connects an MCP client.

**Args:**
- `servers`: Server configurations. If None, uses `get_default_mcp_servers()`

**Returns:**
- Connected `MCPClient` instance

### get_default_mcp_servers

```python
def get_default_mcp_servers() -> list[MCPServerConfig]
```

Returns default server configurations, checking environment variables for URLs.

**Returns:**
- List of `MCPServerConfig` for RAG and Database servers

