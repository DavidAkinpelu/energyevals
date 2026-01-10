# MCP Servers Testing Plan

This document outlines how to test the two MCP servers (Energy RAG Server and Energy Database Server) both locally and remotely.

---

## Overview

| Server | Purpose | Local Command | Remote Command |
|--------|---------|---------------|----------------|
| Energy RAG Server | Semantic search over Qdrant vector database | `energy-rag-server` | `energy-rag-server-sse` |
| Energy Database Server | PostgreSQL querying for energy data | `energy-database-server` | `energy-database-server-sse` |

---

## Prerequisites

### 1. Install Both Servers

```bash
# From repository root
pip install ./mcp-servers/rag-server
pip install ./mcp-servers/database-server
```

### 2. Configure Environment Variables

**RAG Server** (`mcp-servers/rag-server/.env`):
```bash
cp mcp-servers/rag-server/.env.example mcp-servers/rag-server/.env
# Edit with your credentials:
# - QDRANT_URL
# - QDRANT_API_KEY
# - QDRANT_COLLECTION_NAME
# - JINAAI_API_KEY
# - GCS_SERVICE_ACCOUNT_FILE (optional, for images)
```

**Database Server** (`mcp-servers/database-server/.env`):
```bash
cp mcp-servers/database-server/.env.example mcp-servers/database-server/.env
# Edit with your credentials:
# - DB_HOST
# - DB_PORT
# - DB_USER
# - DB_PASSWORD
```

---

## Part 1: Local Testing (stdio transport)

Local testing uses subprocess stdin/stdout communication. The servers run as child processes.

### Method 1: MCP Inspector (Interactive)

The MCP Inspector provides a visual interface for testing tools.

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Test RAG Server
cd mcp-servers/rag-server
mcp-inspector energy-rag-server

# Test Database Server
cd mcp-servers/database-server
mcp-inspector energy-database-server
```

**What to test in Inspector:**
- List available tools
- Call `check_vector_database_schema` (RAG)
- Call `search_documents` with a sample query (RAG)
- Call `show_databases` (Database)
- Call `show_tables` with a database name (Database)
- Call `run_query` with a simple SELECT (Database)

### Method 2: Python Test Scripts

Create test scripts to programmatically verify server functionality.

**Test RAG Server (`test_rag_local.py`):**
```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_rag_server():
    server_params = StdioServerParameters(
        command="energy-rag-server",
        args=[],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Test 1: List tools
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")
            assert len(tools.tools) == 2, "Expected 2 tools"

            # Test 2: Check schema
            result = await session.call_tool("check_vector_database_schema", {})
            print(f"Schema check: {result.content[0].text[:200]}...")

            # Test 3: Search documents
            result = await session.call_tool("search_documents", {
                "query": "renewable energy capacity",
                "num_documents": 3
            })
            print(f"Search results: {result.content[0].text[:500]}...")

            print("All RAG server tests passed!")

if __name__ == "__main__":
    asyncio.run(test_rag_server())
```

**Test Database Server (`test_db_local.py`):**
```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_database_server():
    server_params = StdioServerParameters(
        command="energy-database-server",
        args=[],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Test 1: List tools
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")
            assert len(tools.tools) == 7, "Expected 7 tools"

            # Test 2: Show databases
            result = await session.call_tool("show_databases", {})
            print(f"Databases: {result.content[0].text}")

            # Test 3: Show tables
            result = await session.call_tool("show_tables", {"db_name": "ercot_db"})
            print(f"ERCOT tables: {result.content[0].text[:300]}...")

            # Test 4: Describe table
            result = await session.call_tool("describe_table", {
                "db_name": "ercot_db",
                "table": "ercot_zones"
            })
            print(f"Table schema: {result.content[0].text[:300]}...")

            # Test 5: Run query
            result = await session.call_tool("run_query", {
                "db_name": "ercot_db",
                "query": "SELECT * FROM ercot_zones LIMIT 5"
            })
            print(f"Query result: {result.content[0].text[:300]}...")

            # Test 6: Security - verify blocked operations
            try:
                result = await session.call_tool("run_query", {
                    "db_name": "ercot_db",
                    "query": "DELETE FROM ercot_zones"
                })
                print(f"Security test failed - DELETE should be blocked")
            except Exception as e:
                print(f"Security test passed - DELETE blocked: {e}")

            print("All Database server tests passed!")

if __name__ == "__main__":
    asyncio.run(test_database_server())
```

### Method 3: Using the MCPClient Class

Test using the project's built-in MCPClient:

```python
import asyncio
from energbench.mcp import MCPClient, MCPServerConfig

async def test_with_mcp_client():
    client = MCPClient([
        MCPServerConfig(name="rag", command="energy-rag-server"),
        MCPServerConfig(name="database", command="energy-database-server"),
    ])

    await client.connect()

    try:
        # List all tools from both servers
        tools = client.get_tools()
        print(f"Total tools available: {len(tools)}")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:50]}...")

        # Call RAG tool
        result = await client.call_tool("search_documents", {
            "query": "solar power trends",
            "num_documents": 2
        })
        print(f"RAG search result: {result[:200]}...")

        # Call Database tool
        result = await client.call_tool("show_databases", {})
        print(f"Databases: {result}")

    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(test_with_mcp_client())
```

---

## Part 2: Remote Testing (SSE transport)

Remote testing uses HTTP Server-Sent Events. Servers run as HTTP services.

### Step 1: Start Servers Locally with SSE

**Terminal 1 - RAG Server:**
```bash
cd mcp-servers/rag-server
source .env  # or set environment variables
energy-rag-server-sse
# Server starts on http://0.0.0.0:8000
```

**Terminal 2 - Database Server:**
```bash
cd mcp-servers/database-server
source .env
PORT=8001 energy-database-server-sse
# Server starts on http://0.0.0.0:8001
```

### Step 2: Verify Health Endpoints

```bash
# Check RAG server health
curl http://localhost:8000/health
# Expected: {"status": "ok"}

curl http://localhost:8000/
# Expected: {"name": "energy-rag-server", "version": "0.1.0"}

# Check Database server health
curl http://localhost:8001/health
# Expected: {"status": "ok"}

curl http://localhost:8001/
# Expected: {"name": "energy-database-server", "version": "0.1.0"}
```

### Step 3: Test with Python SSE Client

**Test Remote RAG Server (`test_rag_remote.py`):**
```python
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def test_rag_server_sse():
    url = "http://localhost:8000/sse"

    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Test 1: List tools
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")

            # Test 2: Check schema
            result = await session.call_tool("check_vector_database_schema", {})
            print(f"Schema: {result.content[0].text[:200]}...")

            # Test 3: Search
            result = await session.call_tool("search_documents", {
                "query": "battery storage",
                "num_documents": 2,
                "market": "ERCOT"
            })
            print(f"Search: {result.content[0].text[:300]}...")

            print("Remote RAG server tests passed!")

if __name__ == "__main__":
    asyncio.run(test_rag_server_sse())
```

**Test Remote Database Server (`test_db_remote.py`):**
```python
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def test_database_server_sse():
    url = "http://localhost:8001/sse"

    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Test 1: List tools
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")

            # Test 2: Show databases
            result = await session.call_tool("show_databases", {})
            print(f"Databases: {result.content[0].text}")

            # Test 3: Run query
            result = await session.call_tool("run_query", {
                "db_name": "ercot_db",
                "query": "SELECT COUNT(*) FROM ercot_zones"
            })
            print(f"Query result: {result.content[0].text}")

            print("Remote Database server tests passed!")

if __name__ == "__main__":
    asyncio.run(test_database_server_sse())
```

### Step 4: Test with MCPClient (Remote Mode)

```python
import asyncio
from energbench.mcp import MCPClient, MCPServerConfig

async def test_remote_with_mcp_client():
    client = MCPClient([
        MCPServerConfig(name="rag", url="http://localhost:8000/sse"),
        MCPServerConfig(name="database", url="http://localhost:8001/sse"),
    ])

    await client.connect()

    try:
        tools = client.get_tools()
        print(f"Tools from remote servers: {len(tools)}")

        # Test tools...
        result = await client.call_tool("show_databases", {})
        print(f"Databases: {result}")

    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(test_remote_with_mcp_client())
```

---

## Part 3: Docker Testing

Test servers in containerized environment before deployment.

### Build Images

```bash
# Build RAG server image
cd mcp-servers/rag-server
docker build -t energy-rag-server .

# Build Database server image
cd mcp-servers/database-server
docker build -t energy-database-server .
```

### Run Containers

```bash
# Run RAG server
docker run -d \
  --name rag-server \
  -p 8000:8000 \
  --env-file mcp-servers/rag-server/.env \
  energy-rag-server

# Run Database server
docker run -d \
  --name database-server \
  -p 8001:8000 \
  --env-file mcp-servers/database-server/.env \
  energy-database-server
```

### Verify and Test

```bash
# Check containers are running
docker ps

# Check logs
docker logs rag-server
docker logs database-server

# Test health endpoints
curl http://localhost:8000/health
curl http://localhost:8001/health

# Run SSE test scripts
python test_rag_remote.py
python test_db_remote.py
```

### Cleanup

```bash
docker stop rag-server database-server
docker rm rag-server database-server
```

---

## Part 4: Remote Deployment Testing (Railway)

After deploying to Railway or similar platform.

### Configure Environment

Set these environment variables locally to point to deployed servers:

```bash
export RAG_SERVER_URL="https://your-rag-server.railway.app/sse"
export DATABASE_SERVER_URL="https://your-database-server.railway.app/sse"
```

### Test Deployed Services

```python
import asyncio
import os
from energbench.mcp import create_mcp_client

async def test_deployed_servers():
    # Uses environment variables automatically
    client = await create_mcp_client()

    await client.connect()

    try:
        tools = client.get_tools()
        print(f"Connected to {len(tools)} tools")

        # Test RAG
        result = await client.call_tool("search_documents", {
            "query": "wind energy forecast",
            "num_documents": 3
        })
        print(f"RAG working: {len(result)} chars returned")

        # Test Database
        result = await client.call_tool("show_databases", {})
        print(f"Database working: {result}")

        print("All deployed server tests passed!")

    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(test_deployed_servers())
```

---

## Part 5: Integration Testing

Test servers as part of the full agent workflow.

### Test with ReAct Agent

```python
import asyncio
from energbench.mcp import create_mcp_client
from energbench.agents import ReActAgent  # adjust import as needed

async def test_agent_integration():
    client = await create_mcp_client()
    await client.connect()

    try:
        tools = client.get_tools()

        # Create agent with MCP tools
        agent = ReActAgent(tools=tools)

        # Test query that uses both servers
        response = await agent.run(
            "What renewable energy data is available in the ERCOT database, "
            "and find related documents about Texas energy markets."
        )

        print(f"Agent response: {response}")

    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(test_agent_integration())
```

---

## Test Checklist

### RAG Server Tests

| Test | Local | Remote | Docker |
|------|-------|--------|--------|
| Server starts without errors | [ ] | [ ] | [ ] |
| Health endpoint returns OK | N/A | [ ] | [ ] |
| `check_vector_database_schema` returns schema | [ ] | [ ] | [ ] |
| `search_documents` returns results | [ ] | [ ] | [ ] |
| `search_documents` with filters works | [ ] | [ ] | [ ] |
| `search_documents` with `include_images=true` works | [ ] | [ ] | [ ] |
| Invalid queries handled gracefully | [ ] | [ ] | [ ] |

### Database Server Tests

| Test | Local | Remote | Docker |
|------|-------|--------|--------|
| Server starts without errors | [ ] | [ ] | [ ] |
| Health endpoint returns OK | N/A | [ ] | [ ] |
| `show_databases` lists databases | [ ] | [ ] | [ ] |
| `show_tables` lists tables | [ ] | [ ] | [ ] |
| `describe_table` shows schema | [ ] | [ ] | [ ] |
| `preview_table` returns rows | [ ] | [ ] | [ ] |
| `inspect_query` returns plan | [ ] | [ ] | [ ] |
| `run_query` executes SELECT | [ ] | [ ] | [ ] |
| `get_table_description` returns docs | [ ] | [ ] | [ ] |
| DELETE/INSERT/UPDATE blocked | [ ] | [ ] | [ ] |
| SQL injection attempts blocked | [ ] | [ ] | [ ] |
| Year-suffix table handling works | [ ] | [ ] | [ ] |

### Integration Tests

| Test | Status |
|------|--------|
| MCPClient connects to both servers | [ ] |
| Tools from both servers available | [ ] |
| Agent can use MCP tools | [ ] |
| Connection cleanup works | [ ] |

---

## Troubleshooting

### Common Issues

**Server won't start:**
- Check `.env` file exists and has correct values
- Verify all required environment variables are set
- Check port is not already in use

**Connection refused:**
- Ensure server is running
- Check correct port (8000 for RAG, 8001 for Database if running both)
- Verify firewall settings

**SSE connection fails:**
- Use `/sse` endpoint not root
- Check CORS settings if accessing from browser
- Verify server is running in SSE mode (`-sse` suffix)

**Tool calls fail:**
- Check server logs for errors
- Verify database/Qdrant credentials are correct
- Check network connectivity to external services

**Docker issues:**
- Ensure `.env` file is properly formatted
- Check container logs: `docker logs <container>`
- Verify port mappings: `docker ps`
