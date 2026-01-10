# MCP Servers Deployment Guide

This guide covers local testing and deployment to Railway for the Energy MCP servers.

## Table of Contents

1. [Local Testing](#local-testing)
   - [RAG Server](#testing-rag-server-locally)
   - [Database Server](#testing-database-server-locally)
   - [Testing with MCP Inspector](#testing-with-mcp-inspector)
   - [Testing with Python Client](#testing-with-python-client)
2. [Railway Deployment](#railway-deployment)
   - [Prerequisites](#prerequisites)
   - [Deploying RAG Server](#deploying-rag-server)
   - [Deploying Database Server](#deploying-database-server)
   - [Using SSE Transport](#using-sse-transport-for-remote-servers)
3. [Connecting to Deployed Servers](#connecting-to-deployed-servers)

---

## Local Testing

### Testing RAG Server Locally

#### 1. Setup

```bash
cd mcp-servers/rag-server

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install in development mode
pip install -e .

# Copy and configure environment
cp .env.example .env
```

Edit `.env` with your credentials:
```bash
QDRANT_URL=https://your-qdrant-instance.cloud
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=energy_docs
JINAAI_API_KEY=your-jina-api-key
GCS_SERVICE_ACCOUNT_FILE=/path/to/service-account.json  # Optional
```

#### 2. Run the Server

```bash
# Run directly
energy-rag-server

# Or run as module
python -m energy_rag_server.server
```

The server will start and wait for MCP client connections via stdio.

#### 3. Quick Smoke Test

Create a test script `test_rag.py`:

```python
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_rag_server():
    server_params = StdioServerParameters(
        command="energy-rag-server",
        env={
            "QDRANT_URL": "https://your-instance.cloud",
            "QDRANT_API_KEY": "your-key",
            "QDRANT_COLLECTION_NAME": "energy_docs",
            "JINAAI_API_KEY": "your-jina-key",
        }
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description[:50]}...")

            # Test schema check
            print("\nTesting check_vector_database_schema...")
            result = await session.call_tool("check_vector_database_schema", arguments={})
            schema = json.loads(result.content[0].text)
            print(f"Collection: {schema.get('collection_name')}")
            print(f"Points count: {schema.get('points_count')}")
            print(f"Markets: {schema.get('market_values', [])}")

            # Test search
            print("\nTesting search_documents...")
            result = await session.call_tool(
                "search_documents",
                arguments={
                    "query": "ERCOT battery storage",
                    "num_documents": 3,
                    "include_images": False,
                }
            )
            search_results = json.loads(result.content[0].text)
            print(f"Found {search_results.get('num_results')} documents")
            for doc in search_results.get("documents", [])[:2]:
                print(f"  - Score: {doc['score']:.3f}, Text: {doc['text'][:100]}...")

if __name__ == "__main__":
    asyncio.run(test_rag_server())
```

Run: `python test_rag.py`

---

### Testing Database Server Locally

#### 1. Setup

```bash
cd mcp-servers/database-server

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .

# Copy and configure environment
cp .env.example .env
```

Edit `.env`:
```bash
DB_HOST=your-postgres-host
DB_PORT=5432
DB_USER=your-username
DB_PASSWORD=your-password
```

#### 2. Run the Server

```bash
energy-database-server
```

#### 3. Quick Smoke Test

Create `test_db.py`:

```python
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_database_server():
    server_params = StdioServerParameters(
        command="energy-database-server",
        env={
            "DB_HOST": "your-host",
            "DB_PORT": "5432",
            "DB_USER": "your-user",
            "DB_PASSWORD": "your-password",
        }
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}")

            # Test show_databases
            print("\nTesting show_databases...")
            result = await session.call_tool("show_databases", arguments={})
            dbs = json.loads(result.content[0].text)
            print(f"Databases: {dbs.get('databases', [])}")

            # Test show_tables (if ercot_db exists)
            if "ercot_db" in dbs.get("databases", []):
                print("\nTesting show_tables for ercot_db...")
                result = await session.call_tool(
                    "show_tables",
                    arguments={"db_name": "ercot_db"}
                )
                tables = json.loads(result.content[0].text)
                print(f"Table count: {tables.get('table_count')}")
                print(f"First 5 tables: {tables.get('tables', [])[:5]}")

            # Test a query
            print("\nTesting run_query...")
            result = await session.call_tool(
                "run_query",
                arguments={
                    "db_name": "ercot_db",
                    "query": "SELECT COUNT(*) as count FROM spp_2024 LIMIT 1"
                }
            )
            query_result = json.loads(result.content[0].text)
            print(f"Query result: {query_result.get('rows', [])}")

if __name__ == "__main__":
    asyncio.run(test_database_server())
```

Run: `python test_db.py`

---

### Testing with MCP Inspector

The MCP Inspector is a visual tool for testing MCP servers.

#### Install MCP Inspector

```bash
npm install -g @modelcontextprotocol/inspector
```

#### Test RAG Server

```bash
# From the rag-server directory with .env configured
mcp-inspector energy-rag-server
```

This opens a web UI where you can:
- See available tools
- Call tools with custom arguments
- View responses

#### Test Database Server

```bash
# From the database-server directory with .env configured
mcp-inspector energy-database-server
```

---

### Testing with Python Client

Create a reusable test client `mcp_test_client.py`:

```python
import asyncio
import json
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPTestClient:
    def __init__(self, command: str, env: dict[str, str]):
        self.server_params = StdioServerParameters(command=command, env=env)
        self.session = None

    async def __aenter__(self):
        self._stdio = stdio_client(self.server_params)
        self._streams = await self._stdio.__aenter__()
        self._session_ctx = ClientSession(*self._streams)
        self.session = await self._session_ctx.__aenter__()
        await self.session.initialize()
        return self

    async def __aexit__(self, *args):
        await self._session_ctx.__aexit__(*args)
        await self._stdio.__aexit__(*args)

    async def list_tools(self) -> list[str]:
        result = await self.session.list_tools()
        return [t.name for t in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any] = None) -> dict:
        result = await self.session.call_tool(name, arguments=arguments or {})
        return json.loads(result.content[0].text)


async def main():
    # Test RAG server
    print("=" * 50)
    print("Testing RAG Server")
    print("=" * 50)

    async with MCPTestClient(
        command="energy-rag-server",
        env={
            "QDRANT_URL": "https://your-instance.cloud",
            "QDRANT_API_KEY": "your-key",
            "QDRANT_COLLECTION_NAME": "energy_docs",
            "JINAAI_API_KEY": "your-jina-key",
        }
    ) as client:
        print(f"Tools: {await client.list_tools()}")
        schema = await client.call_tool("check_vector_database_schema")
        print(f"Schema: {json.dumps(schema, indent=2)}")

    # Test Database server
    print("\n" + "=" * 50)
    print("Testing Database Server")
    print("=" * 50)

    async with MCPTestClient(
        command="energy-database-server",
        env={
            "DB_HOST": "your-host",
            "DB_PORT": "5432",
            "DB_USER": "your-user",
            "DB_PASSWORD": "your-password",
        }
    ) as client:
        print(f"Tools: {await client.list_tools()}")
        dbs = await client.call_tool("show_databases")
        print(f"Databases: {dbs}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Railway Deployment

Railway is a cloud platform for deploying applications. MCP servers need to use SSE (Server-Sent Events) transport for remote deployment instead of stdio.

### Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Railway CLI** (optional but recommended):
   ```bash
   npm install -g @railway/cli
   railway login
   ```

### Deploying RAG Server

#### 1. Create SSE Wrapper

Create `mcp-servers/rag-server/energy_rag_server/sse_server.py`:

```python
"""SSE transport wrapper for Railway deployment."""

import os
from dotenv import load_dotenv
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route

load_dotenv()

# Import the MCP server
from energy_rag_server.server import mcp


def create_sse_app():
    """Create Starlette app with SSE transport."""
    sse = SseServerTransport("/messages")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp._mcp_server.run(
                streams[0], streams[1], mcp._mcp_server.create_initialization_options()
            )

    async def handle_messages(request):
        await sse.handle_post_message(request.scope, request.receive, request._send)

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/messages", endpoint=handle_messages, methods=["POST"]),
            Route("/health", endpoint=lambda r: JSONResponse({"status": "ok"})),
        ]
    )

    return app


# For Railway
app = create_sse_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

#### 2. Update Dependencies

Add to `mcp-servers/rag-server/pyproject.toml`:

```toml
dependencies = [
    "mcp>=1.0",
    "qdrant-client>=1.7",
    "llama-index-embeddings-jinaai>=0.3",
    "google-cloud-storage>=2.14",
    "pydantic>=2.0",
    "python-dotenv>=1.0",
    # SSE transport dependencies
    "starlette>=0.27",
    "uvicorn>=0.23",
    "sse-starlette>=1.6",
]

[project.scripts]
energy-rag-server = "energy_rag_server.server:main"
energy-rag-server-sse = "energy_rag_server.sse_server:main"
```

#### 3. Create Dockerfile

Create `mcp-servers/rag-server/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source
COPY energy_rag_server/ energy_rag_server/

# Expose port
EXPOSE 8000

# Run SSE server
CMD ["python", "-m", "energy_rag_server.sse_server"]
```

#### 4. Create railway.json

Create `mcp-servers/rag-server/railway.json`:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

#### 5. Deploy to Railway

**Option A: Via Railway Dashboard**

1. Go to [railway.app](https://railway.app) and create new project
2. Select "Deploy from GitHub repo"
3. Connect your repo and select the `mcp-servers/rag-server` directory
4. Add environment variables:
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `QDRANT_COLLECTION_NAME`
   - `JINAAI_API_KEY`
   - `GCS_SERVICE_ACCOUNT_FILE` (if using GCS)
5. Deploy

**Option B: Via Railway CLI**

```bash
cd mcp-servers/rag-server

# Initialize Railway project
railway init

# Set environment variables
railway variables set QDRANT_URL="https://your-instance.cloud"
railway variables set QDRANT_API_KEY="your-key"
railway variables set QDRANT_COLLECTION_NAME="energy_docs"
railway variables set JINAAI_API_KEY="your-jina-key"

# Deploy
railway up
```

#### 6. Get Deployment URL

After deployment, Railway provides a URL like:
```
https://energy-rag-server-production.up.railway.app
```

---

### Deploying Database Server

#### 1. Create SSE Wrapper

Create `mcp-servers/database-server/energy_database_server/sse_server.py`:

```python
"""SSE transport wrapper for Railway deployment."""

import os
from dotenv import load_dotenv
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse

load_dotenv()

from energy_database_server.server import mcp


def create_sse_app():
    """Create Starlette app with SSE transport."""
    sse = SseServerTransport("/messages")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp._mcp_server.run(
                streams[0], streams[1], mcp._mcp_server.create_initialization_options()
            )

    async def handle_messages(request):
        await sse.handle_post_message(request.scope, request.receive, request._send)

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/messages", endpoint=handle_messages, methods=["POST"]),
            Route("/health", endpoint=lambda r: JSONResponse({"status": "ok"})),
        ]
    )

    return app


app = create_sse_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

#### 2. Update Dependencies

Add to `mcp-servers/database-server/pyproject.toml`:

```toml
dependencies = [
    "mcp>=1.0",
    "psycopg2-binary>=2.9",
    "python-dotenv>=1.0",
    # SSE transport dependencies
    "starlette>=0.27",
    "uvicorn>=0.23",
    "sse-starlette>=1.6",
]
```

#### 3. Create Dockerfile

Create `mcp-servers/database-server/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY energy_database_server/ energy_database_server/

EXPOSE 8000

CMD ["python", "-m", "energy_database_server.sse_server"]
```

#### 4. Deploy

```bash
cd mcp-servers/database-server

railway init
railway variables set DB_HOST="your-host"
railway variables set DB_PORT="5432"
railway variables set DB_USER="your-user"
railway variables set DB_PASSWORD="your-password"
railway up
```

---

### Using SSE Transport for Remote Servers

Once deployed, connect using SSE transport instead of stdio.

#### Python Client for SSE

```python
import asyncio
import json
from mcp import ClientSession
from mcp.client.sse import sse_client

async def connect_to_railway_rag():
    """Connect to RAG server on Railway."""
    url = "https://energy-rag-server-production.up.railway.app/sse"

    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            print(f"Connected! Tools: {[t.name for t in tools.tools]}")

            # Search documents
            result = await session.call_tool(
                "search_documents",
                arguments={"query": "ERCOT battery", "num_documents": 3}
            )
            print(json.loads(result.content[0].text))

asyncio.run(connect_to_railway_rag())
```

#### Using with OpenAI

```python
import asyncio
import json
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client

client = OpenAI()

async def agent_with_remote_tools():
    rag_url = "https://energy-rag-server-production.up.railway.app/sse"
    db_url = "https://energy-database-server-production.up.railway.app/sse"

    async with sse_client(rag_url) as (rag_read, rag_write), \
               sse_client(db_url) as (db_read, db_write):

        async with ClientSession(rag_read, rag_write) as rag_session, \
                   ClientSession(db_read, db_write) as db_session:

            await rag_session.initialize()
            await db_session.initialize()

            # Get all tools
            rag_tools = await rag_session.list_tools()
            db_tools = await db_session.list_tools()

            # Convert to OpenAI format
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema
                    }
                }
                for t in [*rag_tools.tools, *db_tools.tools]
            ]

            # Chat with tools
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "What ERCOT data is available?"}],
                tools=openai_tools
            )

            # Handle tool calls...
            print(response.choices[0].message)

asyncio.run(agent_with_remote_tools())
```

---

## Connecting to Deployed Servers

### Environment Variables for Client

Create a `.env` for your client application:

```bash
# Remote MCP Servers (Railway)
RAG_SERVER_URL=https://energy-rag-server-production.up.railway.app/sse
DATABASE_SERVER_URL=https://energy-database-server-production.up.railway.app/sse
```

### Health Checks

Both servers expose a `/health` endpoint:

```bash
curl https://energy-rag-server-production.up.railway.app/health
# {"status": "ok"}
```

### Monitoring on Railway

Railway provides:
- **Logs**: View real-time logs in dashboard
- **Metrics**: CPU, memory, network usage
- **Deployments**: Rollback to previous versions

Access via Railway dashboard or CLI:
```bash
railway logs
railway status
```

---

## Troubleshooting

### Common Issues

**1. Connection refused**
- Check if server is running
- Verify environment variables are set
- Check Railway deployment logs

**2. Timeout errors**
- Increase client timeout
- Check network connectivity
- Verify Railway service is healthy

**3. Authentication errors**
- Verify API keys are correct
- Check environment variable names

**4. Database connection issues**
- Ensure database allows Railway IPs
- Check connection string format
- Verify SSL settings if required

### Debug Mode

For local debugging, run with verbose logging:

```bash
# Set debug environment
export MCP_DEBUG=1
energy-rag-server
```

Or in Python:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
