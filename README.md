# CoreLink AI

General-purpose A2A reasoning engine built on LangGraph and MCP. The core runtime is domain-agnostic; domain capability comes from MCP servers discovered at startup.

## Current Architecture

- `Coordinator`: chooses a lightweight execution plan such as `direct_answer` or `react_reason -> verifier_check`.
- `Executor`: tool-enabled reasoning loop for multi-step tasks.
- `Verifier`: step-level gatekeeper that returns `PASS`, `REVISE`, or `BACKTRACK`.
- `Format Normalizer`: final formatting pass when the request explicitly needs JSON or XML.
- `Execution Memory`: compact router, executor, and verifier hints stored in SQLite and reused on later runs.
- `Context Windowing`: trims long histories while preserving tool-call adjacency.
- `Multi-turn Support`: conversation history is reused through A2A `context_id`.

## Repository Layout

```text
src/
  server.py                 A2A server entrypoint
  executor.py               A2A <-> LangGraph bridge
  context_manager.py        token counting and message windowing
  conversation_store.py     multi-turn conversation storage
  mcp_client.py             MCP tool loading
  tools.py                  built-in calculator/search/time tools
  agent/
    graph.py                graph construction
    runner.py               run wrapper and summaries
    state.py                shared state schema
    prompts.py              prompts and structured outputs
    cost.py                 token/cost accounting
    operators.py            operator registry and defaults
    memory/                 SQLite-backed execution memory
    nodes/                  coordinator, reasoner, verifier, formatter, context
  mcp_servers/              local MCP servers

tests/
docs/
```

## Setup

```bash
uv sync
```

Create `.env` with the keys and MCP configuration you want to use.

Common variables:

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=
MODEL_NAME=gpt-4o-mini
TAVILY_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=project-pulse
MCP_SERVER_STDIO=
MCP_SERVER_URLS=
```

## Run

Start the A2A server:

```bash
uv run python src/server.py
```

Custom port:

```bash
uv run python src/server.py --port 9010
```

## MCP Servers

You can run local MCP servers directly when testing:

```bash
uv run python src/mock_mcp_server.py
uv run python src/mcp_servers/finance/server.py
uv run python src/mcp_servers/options_chain/server.py
uv run python src/mcp_servers/trading_sim/server.py
uv run python src/mcp_servers/risk_metrics/server.py
```

If you use stdio MCP servers, list them in `MCP_SERVER_STDIO`. If you use HTTP MCP servers, list them in `MCP_SERVER_URLS`.

## Tests

Targeted examples:

```bash
uv run pytest tests/test_agent.py -v
uv run pytest tests/test_features.py -v
uv run pytest tests/test_coordinator.py -v
uv run pytest tests/test_verifier.py -v
uv run pytest tests/test_memory.py -v
uv run pytest tests/test_end_to_end_verifier.py -v
```

Run the full suite:

```bash
uv run pytest tests -v
```

## Notes

- Built-in tools and MCP tools are both available to the runtime.
- The verifier/backtrack loop is the default heavy-research path.
- Execution memory currently uses compact exact-match task signatures in SQLite.
- LangSmith tracing is optional; if outbound network access is blocked, local tests still run.
