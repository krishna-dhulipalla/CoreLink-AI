# CoreLink AI — Generalist Reasoning Engine

A domain-agnostic A2A (Agent-to-Agent) reasoning engine built on **LangGraph** and **MCP (Model Context Protocol)**. The core brain is generic — domain-specific capabilities live entirely in external MCP servers that the agent discovers and uses automatically.

## Architecture

```
┌──────────────────────────────┐
│         CoreLink AI          │
│   LangGraph ReAct Engine     │
│  Plan → Act → Learn Loop     │
│                              │
│  ┌────────────────────────┐  │
│  │  MultiServerMCPClient  │  │
│  │  (auto-discovers tools)│  │
│  └────────┬───────────────┘  │
└───────────┼──────────────────┘
            │ stdio / http
    ┌───────┼──────────────────────────────┐
    ▼       ▼       ▼                      ▼
 finance  options  trading_sim        risk_metrics
 (3 tools) chain   (4 tools)          (5+ tools)
           (4 tools)
```

The agent's system prompt contains **no domain-specific logic**. Adding a new domain requires only:

1. Writing a new `src/mcp_servers/<domain>/server.py`
2. Appending its startup command to `MCP_SERVER_STDIO` in `.env`

## Finance MCP Servers

The agent ships with four finance-focused MCP servers (16 tools):

| Server          | Tools                                                                                                                |
| --------------- | -------------------------------------------------------------------------------------------------------------------- |
| `finance`       | `black_scholes_price`, `option_greeks`, `mispricing_analysis`                                                        |
| `options_chain` | `get_options_chain`, `get_expirations`, `get_iv_surface`, `analyze_strategy`                                         |
| `trading_sim`   | `create_portfolio`, `execute_options_trade`, `get_positions`, `get_pnl_report`                                       |
| `risk_metrics`  | `calculate_portfolio_greeks`, `calculate_var`, `calculate_risk_metrics`, `run_stress_test`, `calculate_max_drawdown` |

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/krishna-dhulipalla/CoreLink-AI.git
cd CoreLink-AI

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# Fill in your keys (see Configuration below)
```

## Configuration

Create a `.env` file with the following:

```env
# Required
OPENAI_API=sk-...

# Optional: Tavily for web search
TAVILY_API_KEY=tvly-...

# Optional: LangSmith for observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=corelink-ai

# MCP Servers (auto-configured for finance by default)
MCP_SERVER_STDIO="finance=uv run python src/mcp_servers/finance/server.py,options_chain=uv run python src/mcp_servers/options_chain/server.py,trading_sim=uv run python src/mcp_servers/trading_sim/server.py,risk_metrics=uv run python src/mcp_servers/risk_metrics/server.py"

# Connect to external MCP servers (e.g. from a benchmark/green agent)
MCP_SERVER_URLS=""
```

## Running the Agent

```bash
# Start the agent server (default port 9009)
uv run python src/server.py

# Or on a custom port
uv run python src/server.py --port 9010
```

The agent will automatically spawn all configured MCP server subprocesses and load their tools.

## Running the MCP Servers Standalone

Each MCP server can be run independently for testing:

```bash
# Finance pricing server
uv run python src/mcp_servers/finance/server.py

# Options chain data server
uv run python src/mcp_servers/options_chain/server.py

# Paper trading simulator
uv run python src/mcp_servers/trading_sim/server.py

# Portfolio risk metrics
uv run python src/mcp_servers/risk_metrics/server.py
```

## Testing

```bash
# Install test dependencies
uv sync --extra test

# Run all unit tests
uv run pytest tests/ -v

# Run A2A conformance tests (requires the agent server to be running)
uv run pytest tests/test_agent.py --agent-url http://localhost:9009

# Run finance tool unit tests (no server needed)
uv run pytest tests/test_finance_tools.py -v

# Run feature unit tests (no server needed)
uv run pytest tests/test_features.py -v
```

## Project Structure

```
src/
├── server.py               # A2A server entrypoint
├── executor.py             # A2A ↔ LangGraph bridge
├── agent.py                # Core reasoning engine (LangGraph)
├── mcp_client.py           # MultiServerMCPClient wrapper
├── context_manager.py      # Token counting & message windowing
├── conversation_store.py   # Multi-turn conversation state
├── finance_tools.py        # Legacy direct tool wrappers (kept for tests)
├── tools.py                # Built-in tools (calculator, search)
└── mcp_servers/
    ├── finance/server.py       # Black-Scholes pricing & Greeks
    ├── options_chain/server.py # Options chain & IV surface
    ├── trading_sim/server.py   # Paper trading engine
    └── risk_metrics/server.py  # VaR, Sharpe, stress testing

tests/
├── test_features.py        # Unit tests for windowing & reflection
├── test_finance_tools.py   # Unit tests for finance math tools
└── test_agent.py           # A2A conformance tests

docs/
├── DESIGN.md               # Architecture deep-dive
├── A2A_INTERFACE_SPEC.md   # A2A protocol interface spec
├── core_foundations.md     # Core system documentation
└── progress.md             # Agent handoff & chat log
```

## Docker

```bash
# Build
docker build -t corelink-ai .

# Run
docker run -p 9009:9009 --env-file .env corelink-ai
```
