# CoreLink AI

CoreLink AI is a finance-first A2A reasoning engine built on LangGraph and MCP.

It is designed for tasks where a plain chat workflow is not enough: quantitative finance, options strategy analysis, portfolio risk review, event-driven finance, document-backed reasoning, and policy-constrained recommendations.

## What It Does

- Runs a modular reasoning engine instead of a single prompt-heavy loop
- Uses structured evidence and normalized tool outputs for finance tasks
- Supports finance-specific templates for quant, options, research, portfolio risk, and event-driven analysis
- Applies risk and compliance gates on actionable finance paths
- Exposes an A2A-compatible server surface for local experimentation

## Architecture

![CoreLink AI architecture](docs/architecture.png)

At a high level, the system works in five steps:

1. `Intake` normalizes the request and detects output requirements.
2. `Planner + Capability Resolver` chooses the execution mode and safe tool families for the task.
3. `Context Curator` assembles compact evidence from the prompt, tools, and documents.
4. `Executor` reasons with the curated context and calls tools only when exact data or computation is needed.
5. `Reviewer`, `Self-Reflection`, `Output Adapter`, and `Reflect` finish the answer and persist run records.

## Supported Workloads

- Finance quant and market-data-backed analysis
- Options strategy analysis with risk scenarios
- Portfolio risk review and policy-aware recommendations
- Equity research and event-driven finance workflows
- Document-backed finance and general QA flows

## Handling Finance Complexity

Finance tasks are harder than general QA because they involve uncertainty, exact computation, risk tradeoffs, and policy constraints.

CoreLink AI handles that by:

- using structured market, document, and analytics tools instead of relying on free-form model recall
- separating evidence gathering from reasoning and final answer generation
- running risk checks on actionable finance paths
- applying compliance checks when the output moves from analysis toward recommendation
- keeping assumptions and sources explicit throughout the run

## Quickstart

Install dependencies:

```bash
uv sync
```

Create a `.env` file:

```env
OPENAI_API_KEY=...
MODEL_PROFILE=balanced
MODEL_NAME=Qwen/Qwen3-32B-fast
PROFILER_MODEL=Qwen/Qwen3-32B-fast
SOLVER_MODEL=Qwen/Qwen3-32B-fast
REVIEWER_MODEL=Qwen/Qwen3-32B-fast
STRUCTURED_OUTPUT_MODE=local_json
MCP_SERVER_STDIO=
MCP_SERVER_URLS=
```

Model role env vars:
- `PROFILER_MODEL` controls task profiling fallback
- `SOLVER_MODEL` controls the main engine reasoning path
- `REVIEWER_MODEL` controls ambiguous final review only

Legacy env names like `COORDINATOR_MODEL`, `EXECUTOR_MODEL`, and `VERIFIER_MODEL` still work, but the canonical names above are preferred.

Start the A2A server:

```bash
uv run python src/server.py --port 9010
```

Agent card:

```text
http://127.0.0.1:9010/.well-known/agent-card.json
```

## Local Validation

Deterministic smoke:

```bash
uv run pytest tests/test_engine_runtime.py -q
```

Live LLM smoke:

```bash
uv run python scripts/run_live_engine_smoke.py
```

Benchmark/stateless smoke:

```bash
BENCHMARK_STATELESS=1 uv run python scripts/run_benchmark_stateless_smoke.py
```

Test suite:

```bash
uv run pytest tests -q
```

## Benchmark Mode

Set `BENCHMARK_STATELESS=1` when you want each run to behave like a fresh benchmark item instead of an interactive conversation.

In this mode the runtime:
- ignores prior conversation history
- avoids writing prior turns back into the next benchmark item
- dedupes adjacent repeated messages before graph entry

Use it for offline benchmark slices and trace debugging. Leave it unset for normal multi-turn A2A usage.

## Project Layout

```text
src/
  server.py
  mcp_servers/
  agent/
    graph.py
    runner.py
    nodes/
    context/
    solver/
    tools/
    memory/

docs/
  architecture.svg
  v3_checkpoint.md
  finance_hands_checkpoint.md
```

## Status

CoreLink AI is actively evolving. The current runtime is stable enough for local use, live smoke validation, and continued finance-system development, but it should not be treated as finished production infrastructure.

Development progress is tracked internally in [docs/progress.md](docs/progress.md).
