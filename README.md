# energBench

AI agent evaluation framework for energy analytics.

## Features

- **ReAct Agent**: Multi-provider LLM support (OpenAI, Anthropic, Google, DeepInfra)
- **Energy Tools**: GridStatus, Tariffs, Renewables, Battery optimization, Dockets, Weather, Search
- **MCP Integration**: External RAG and database tools via Model Context Protocol
- **Benchmark Framework**: Evaluate agents across questions with metrics and comparison
- **Observability**: Langfuse and JSON tracing with full execution data

## Quick Start

```bash
# Install system dependencies (Ipopt solver for battery optimization)
sudo ./install.sh

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys

# Run a benchmark
python scripts/run_benchmark.py
```

## Installation

### System Dependencies

For battery optimization tools, install Ipopt solver:

```bash
# Debian/Ubuntu
sudo ./install.sh
```

The install script builds Ipopt and required third-party solvers from source, skipping Java test harness to avoid JDK issues.

### Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For development (includes testing and linting tools):

```bash
pip install -r requirements-dev.txt
```

## Configuration

### API Keys

Create a `.env` file with your credentials:

```bash
# LLM Providers (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
DEEPINFRA_API_KEY=...

# Observability (optional)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Tools (optional - enables specific functionality)
EXA_API_KEY=...                      # SearchTool
GRIDSTATUS_API_KEY=...               # GridStatusAPITool
OPENWEATHER_API_KEY=...              # OpenWeatherTool
OPEN_EI_API_KEY=...                  # TariffsTool
RENEWABLES_NINJA_API_KEY=...         # RenewablesTool
```

Copy `.env.example` for a template.

### MCP Servers

MCP servers provide RAG and database access. They connect via remote URLs configured
in your `.env` file. Set the URL env vars below to enable them; if neither is set,
MCP is effectively disabled even with `mcp.enabled: true`.

```bash
RAG_SERVER_URL=https://your-rag-server.com/sse
DATABASE_SERVER_URL=https://your-db-server.com/sse
```

## Usage

### Ask a Question (Interactive)

The quickest way to use energBench is the interactive agent script. Type a question, get an answer:

```bash
# Start interactive mode (defaults to openai / gpt-4o-mini)
python scripts/run_agent.py

# Choose a provider
python scripts/run_agent.py -p anthropic
python scripts/run_agent.py -p google

# Pick a specific model
python scripts/run_agent.py -p openai -m gpt-4o

# Enable MCP tools (RAG + database)
python scripts/run_agent.py --mcp

# Run without tools (pure LLM)
python scripts/run_agent.py --no-tools

# Ask a single question (no interactive loop)
python scripts/run_agent.py -q "What are current ERCOT energy prices?"
```

Inside the interactive session, type your question at the `>` prompt. The agent will use its tools to research and answer. Type `quit` to exit.

### Running Benchmarks

For detailed benchmark configuration, custom questions, evaluation, and multi-model comparison, see the [Benchmark Guide](docs/BENCHMARK_GUIDE.md).
Benchmark runs require at least one explicit `models` entry in config; there is no provider/model fallback.

Multi-trial seed controls are configured in `agent`:

```yaml
agent:
  num_trials: 3
  shuffle: true
  seed_mode: rotate              # fixed | rotate | random_per_trial
  seed: 12345                    # optional base seed
  # seeds: [101, 202, 303]       # optional explicit per-trial seeds
```

## Architecture

### ReAct Agent Loop

The agent uses a Reasoning-Acting loop:

1. **Thought**: Analyze the question and plan next action
2. **Action**: Select and execute a tool
3. **Observation**: Process tool output
4. **Repeat**: Continue until answer is complete

Maximum iterations default to 25 (configurable).

### Provider Abstraction

A unified interface to run models from any major LLM provider:

- **OpenAI** — GPT, O1, O3, and more
- **Anthropic** — Claude models (Sonnet, Opus, Haiku)
- **Google** — Gemini models (Flash, Pro)
- **DeepInfra** — Open-source models (Llama, Mistral, and more)

Providers implement a common `BaseProvider` protocol with tool calling and streaming support.

### Tool System

Tools are registered via the default tool registry in `create_default_registry()`:
1. **Direct registration**: Tools instantiated and registered in code
2. **MCP servers**: External tools via Model Context Protocol

Each tool provides:
- JSON schema for LLM tool calling
- Async execution
- Error handling with structured results

### Observability

Traces capture full execution:
- All ReAct steps (thought, action, observation)
- Tool inputs/outputs
- Token usage and latency
- Failed calls with errors

Backends:
- **Langfuse**: Cloud platform with UI
- **JSON**: Local JSONL or individual files
- **Both**: Use multiple observers simultaneously


## Development

Run tests:

```bash
pytest
```

Lint and type check:

```bash
ruff check .
mypy energbench
```

## Documentation
