# CoreLink AI

CoreLink AI is a reasoning engine for evidence-grounded analytical work. It plans tasks, retrieves supporting material, extracts structured evidence, performs deterministic computations when possible, and applies review before returning a final answer.

The project is built for workflows where correctness, provenance, and controlled tool use matter more than unconstrained generation.

## Why CoreLink AI

Most agent systems fail in one of two ways:

- they rely too heavily on model recall instead of evidence
- they use tools, but without enough policy around when to search, when to compute, when to retry, and when to stop

CoreLink AI is built to solve that gap. The runtime is designed to:

- prefer grounded evidence over unsupported synthesis
- prefer deterministic compute over free-form math when the task allows it
- change retrieval strategy when a current search regime stalls
- validate answerability before allowing a failure endpoint

## What It Can Do

CoreLink AI is intended for tasks such as:

- document-grounded financial and analytical questions
- table extraction and period-aware aggregation
- multi-step reasoning over structured and unstructured evidence
- tool-backed analysis with explicit review and retry policy
- server-side A2A agent execution

## How It Works

At a high level, the system follows this flow:

```text
Request
  -> plan the task
  -> choose tools and retrieval strategy
  -> gather and structure evidence
  -> compute or synthesize within policy
  -> review and repair if needed
  -> format and return the final answer
```

The key runtime behaviors are:

- Typed planning
  The system builds an explicit task and retrieval intent before entering the tool loop.

- Strategy-based retrieval
  Retrieval is not a single fixed path. The runtime can rotate between strategies such as table-first, hybrid, multi-table, or multi-document search depending on the question and prior failures.

- Structured evidence
  Retrieved material is normalized into tables, rows, cells, and grounded text fragments so downstream steps operate on typed evidence instead of raw strings.

- Deterministic compute
  When a question can be answered through explicit aggregation or transformation, the runtime prefers a deterministic compute path.

- Bounded LLM arbitration
  LLMs are used for planning, arbitration, repair, and capability acquisition at controlled boundaries rather than as a blanket fallback.

- Review before finalization
  Answers are checked for semantic alignment, evidence sufficiency, and endpoint policy before they are emitted.

## Design Principles

- Evidence first
  Answers should come from retrieved evidence and verified transformations.

- Deterministic where possible
  If the answer can be computed safely, the runtime should compute it rather than improvise it.

- Explicit retries
  When the system is wrong, it should mutate strategy or search regime, not repeat the same failed path.

- Policy-driven stopping
  The runtime should not quietly give up. It should either continue through a materially new path or stop with a clear reason.

- Operational transparency
  The system is designed to be observable and diagnosable in local development and evaluation runs.

## Quickstart

### Prerequisites

- Python 3.13+
- `uv`
- an OpenAI-compatible API key

### Install

```bash
git clone <your-repo-url>
cd Project-Pulse-Generalist-A2A-Reasoning-Engine
uv sync
```

Create your local environment file:

```bash
cp .env.example .env
```

Set at least:

```bash
OPENAI_API_KEY=your_key_here
```

Optional:

- `OPENAI_BASE_URL`
- role-specific model overrides such as `SOLVER_MODEL` or `REVIEWER_MODEL`

## Run The Server

Start the A2A server:

```bash
uv run python src/server.py --host 127.0.0.1 --port 9009
```

The server exposes CoreLink AI as an A2A-compatible agent with streaming support.

## Run Locally

### General runtime smoke

```bash
uv run python scripts/run_live_engine_smoke.py
```

### Test suite

```bash
uv run pytest tests/
```

Focused examples:

```bash
uv run pytest tests/test_engine_runtime.py -q
uv run pytest tests/test_retrieval_strategy_kernel.py -q
uv run pytest tests/test_llm_repair.py -q
```

## Configuration

The runtime is configured through environment variables in `.env`.

Common settings include:

- provider access
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`

- model overrides
  - `SOLVER_MODEL`
  - `REVIEWER_MODEL`
  - `DOCUMENT_SOLVER_MODEL`
  - `DOCUMENT_REVIEWER_MODEL`

- runtime controls
  - `MAX_TOOL_CALLS`
  - `MAX_REVISE_CYCLES`
  - `MAX_CONTEXT_TOKENS`
  - `STRUCTURED_OUTPUT_MODE`
  - `TOOL_CALL_MODE`

- optional persistence and diagnostics
  - `ENABLE_RUN_TRACER`
  - `TRACE_MAX_RECENT`
  - `ENABLE_AGENT_MEMORY`

For normal usage, only provider credentials are required. Benchmark- and corpus-specific settings are optional.

## Evaluation

CoreLink AI has been hardened with benchmark-driven testing, including document-heavy financial QA workloads. Evaluation support exists in the repo, but it is separate from the public runtime surface.

If you want to run the current local benchmark smoke path:

```powershell
$env:BENCHMARK_NAME="officeqa"
$env:BENCHMARK_STATELESS="1"
uv run python scripts/run_officeqa_regression.py --smoke
```

If the benchmark requires a local corpus index, build it first:

```powershell
$env:OFFICEQA_CORPUS_DIR="data/officeqa/source/treasury_bulletins_parsed/jsons"
uv run python scripts/build_officeqa_index.py --corpus-root "$env:OFFICEQA_CORPUS_DIR"
uv run python scripts/verify_officeqa_corpus.py --corpus-root "$env:OFFICEQA_CORPUS_DIR"
```

OfficeQA is used here as an evaluation harness, not as the product identity of CoreLink AI.

## Operating Notes

- The runtime supports strategy-aware retrieval rather than a single hardcoded search path.
- Deterministic compute remains the preferred path for structured numeric questions.
- Optional memory is available but disabled by default for most local and benchmark runs.
- The system is designed to run cleanly in stateless benchmark mode and normal server mode.

## Status

CoreLink AI is currently on the V6 architecture track, focused on:

- typed retrieval strategies
- bounded repair and regime mutation
- evidence arbitration
- compute capability acquisition
- answerability-aware stopping policy

This is the most stable architecture the project has had so far.
