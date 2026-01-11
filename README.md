# energBench

AI agent evaluation framework for energy analytics.

## Features

- **Custom ReAct Agent**: Multi-provider support (OpenAI, Anthropic, DeepInfra)
- **MCP Servers**: RAG and Database tools via Model Context Protocol
- **Standard Tools**: Search, GridStatus, Tariffs, Renewables, Battery optimization, Dockets
- **Evaluation Framework**: Benchmarks, metrics, model comparison
- **Observability**: Langfuse and JSON file tracing with full data capture

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

## Usage

```bash
# Run a benchmark
python scripts/run_benchmark.py --benchmark benchmarks/datasets/energy_analysis/ercot.json --provider openai --model gpt-4o

# Compare models
python scripts/compare_models.py --benchmark benchmarks/datasets/energy_analysis/ercot.json
```

## Observability

The framework supports multiple observability backends for tracing agent runs:

- **Langfuse**: Cloud-based observability platform
- **JSON**: Local JSON file logging
- **Both**: Use multiple backends simultaneously

### Configuration

For Langfuse, set environment variables in your `.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com  # optional, defaults to cloud
```

### Usage

```python
from energbench.observability import get_observer

# Choose backend: "langfuse", "json", "both", or "auto"
observer = get_observer("json", output_dir="./traces")

# After running your agent
run = await agent.run("What are ERCOT energy prices?")

# Trace the run
trace_id = observer.trace_agent_run(
    run=run,
    metadata={"experiment": "v1"},
    tags=["ercot", "prices"],
    user_id="analyst_1",
    session_id="session_123",
)

# Flush and cleanup
observer.flush()
observer.shutdown()
```

### Backend Options

| Backend | Description |
|---------|-------------|
| `langfuse` | Send traces to Langfuse cloud (requires credentials) |
| `json` | Write traces to local JSON files |
| `both` | Use both Langfuse and JSON simultaneously |
| `auto` | Use Langfuse if available, otherwise JSON |

### JSON Observer Features

The JSON observer captures complete trace data:

- All execution steps (thought, action, observation, answer, error)
- Full tool inputs and outputs (never truncated)
- Failed tool calls with error details
- Token usage and latency metrics
- Step-by-step timestamps

```python
from energbench.observability import JSONFileObserver

observer = JSONFileObserver(
    output_dir="./traces",      # Directory for trace files
    single_file=False,          # True for JSONL, False for individual files
    pretty_print=True,          # Format JSON with indentation
)

# Load a trace later
trace_data = observer.load_trace(trace_id)
print(trace_data["step_summary"])  # Summary of steps and failures
```

## Project Structure

```
energbench/
├── agent/          # ReAct agent and LLM providers
├── mcp/            # MCP client for remote servers
├── tools/          # Standard tools (search, gridstatus, tariffs, etc.)
├── evaluation/     # Benchmarks and metrics
├── observability/  # Tracing backends (Langfuse, JSON)
└── utils/          # Configuration and utilities

mcp-servers/
├── rag-server/     # RAG server with vector search
└── database-server/# Database query server

scripts/
└── agent_tests/    # Runnable benchmark scripts

tests/
├── provider_tests/     # LLM provider tests
├── agent_tests/        # Agent and metrics tests
├── observability_tests/# Observer tests
├── tool_tests/         # Individual tool tests
└── unit_tests/         # Unit tests
```

## Running Tests

```bash
# Run observability tests
python tests/observability_tests/test_observers.py

# Run provider tests (requires API keys)
python tests/provider_tests/test_providers.py

# Run agent metrics tests (requires MCP servers)
python tests/agent_tests/test_agent_metrics.py

# Run benchmark prompts with observability
python scripts/agent_tests/run_agent_prompts.py --observe json
```
