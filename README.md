# energBench

AI agent evaluation framework for energy analytics.

## Features

- **Custom ReAct Agent**: Multi-provider support (OpenAI, Anthropic, DeepInfra)
- **MCP Servers**: RAG and Database tools via Model Context Protocol
- **Standard Tools**: Search, GridStatus, Tariffs, Renewables, Battery optimization, Dockets
- **Evaluation Framework**: Benchmarks, metrics, model comparison
- **Observability**: Langfuse integration for tracing

## Installation

For Debian/Ubuntu, install the Ipopt solver and system deps first (builds Ipopt + required third-party solvers from source). The install script skips Ipopt's Java test harness to avoid JDK native library issues on some systems.

```bash
# Install system dependencies (Ipopt + build tooling)
sudo ./install.sh

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
```

## MCP Servers

The framework includes MCP (Model Context Protocol) servers for RAG and database access.

### Local Servers (Default)

```bash
# Install MCP servers
pip install ./mcp-servers/rag-server
pip install ./mcp-servers/database-server

# They'll be used automatically by the agent
```

### Remote Servers (Production)

To use deployed MCP servers, set environment variables in your `.env`:

```bash
RAG_SERVER_URL=https://your-rag-server.com/sse
DATABASE_SERVER_URL=https://your-db-server.com/sse
```

See [`docs/MCP_CLIENT_USAGE.md`](docs/MCP_CLIENT_USAGE.md) for detailed usage and [`mcp-servers/DEPLOYMENT_GUIDE.md`](mcp-servers/DEPLOYMENT_GUIDE.md) for deployment instructions.

## Usage

```bash
# Run a benchmark
python scripts/run_benchmark.py --benchmark benchmarks/datasets/energy_analysis/ercot.json --provider openai --model gpt-4o

# Compare models
python scripts/compare_models.py --benchmark benchmarks/datasets/energy_analysis/ercot.json
```

## Project Structure

```
energbench/
├── agent/          # ReAct agent and LLM providers
├── mcp/            # MCP servers (RAG, Database)
├── tools/          # Standard tools
├── evaluation/     # Benchmarks and metrics
├── observability/  # Langfuse integration
└── utils/          # Configuration and utilities
```
