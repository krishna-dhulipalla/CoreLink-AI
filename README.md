# OfficeQA Reasoning Engine

This repository is now an OfficeQA-only document reasoning runtime. The active path is built for Treasury Bulletin retrieval, structured evidence extraction, deterministic compute, validation, and exact benchmark answer formatting.

## Overview

The current goal is to solve OfficeQA reliably:

- retrieve from the Treasury Bulletin corpus
- extract the correct table, row, cell, or page evidence
- compute deterministically when the question is numeric
- emit the exact final answer contract expected by the benchmark

The canonical docs are:

- `docs/officeqa_integration_plan.md`: failure analysis and architecture rationale
- `docs/officeqa_execution_plan.md`: active backlog and delivery phases
- `docs/v5_runtime_walkthrough.md`: current V5 runtime walkthrough
- `docs/progress.md`: short execution log and handoff history

## Runtime Shape

The active runtime is:

`A2A request -> benchmark adapter -> corpus retrieval -> structured extraction -> deterministic compute -> validator/reviewer -> output adapter`

Core runtime layers:

- benchmark adapter
- corpus retrieval layer
- structured extraction layer
- deterministic compute layer
- validator and output adapter
- tracer, budgets, and run artifacts

## Model Stack

Role-based model selection lives in `src/agent/model_config.py`.

Current `MODEL_PROFILE=officeqa` defaults are:

- `profiler`: `Qwen/Qwen3-32B-fast`
- `direct`: `Qwen/Qwen3-32B-fast`
- `solver`: `deepseek-ai/DeepSeek-V3.2`
- `reviewer`: `Qwen/Qwen3-32B-fast`
- `adapter`: `Qwen/Qwen3-32B-fast`
- `reflection`: `Qwen/Qwen3-32B-fast`

Recommended strong OfficeQA setup:

- keep a strong document solver through `DOCUMENT_SOLVER_MODEL` or `LONG_CONTEXT_SOLVER_MODEL`
- use `DOCUMENT_REVIEWER_MODEL` when scope or ambiguity failures show up often
- keep adapter and reflection lightweight unless they become the bottleneck

The executor logs the active startup model map so teammates can confirm the role stack without reading source.

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) or Python 3.13+
- an OpenAI-compatible API key

### Install

```bash
git clone https://github.com/your-username/CoreLink-AI.git
cd CoreLink-AI
uv sync
cp .env.example .env
```

Set at least:

- `OPENAI_API_KEY`
- `BENCHMARK_NAME=officeqa`
- `MODEL_PROFILE=officeqa`

## Local OfficeQA Corpus

The canonical local layout is:

```text
data/
  officeqa/
    README.md
    treasury_bulletins_parsed/
    treasury_bulletin_pdfs/
    officeqa.csv
```

Local setup:

1. Download or clone the OfficeQA corpus locally.
2. Put the parsed corpus under `data/officeqa/treasury_bulletins_parsed/`.
3. Set `OFFICEQA_CORPUS_DIR=data/officeqa/treasury_bulletins_parsed`.
4. Build the index:

```bash
uv run python scripts/build_officeqa_index.py --corpus-root "$OFFICEQA_CORPUS_DIR"
```

5. Verify the bundle:

```bash
uv run python scripts/verify_officeqa_corpus.py --corpus-root "$OFFICEQA_CORPUS_DIR"
```

The script writes index artifacts under `OFFICEQA_INDEX_DIR` or, by default, under `OFFICEQA_CORPUS_DIR/.officeqa_index/`.

## Competition Deployment

Do not assume Judge or A2A exposes the Treasury corpus.

- keep the corpus out of git history
- package or mount the corpus for competition runs
- point `OFFICEQA_CORPUS_DIR` at that packaged path
- when `COMPETITION_MODE=1` and `BENCHMARK_NAME=officeqa`, startup fails fast if corpus or index is missing

Judge tools are optional auxiliary surfaces, not the primary data path.

## Running The Runtime

Start the A2A server:

```bash
uv run src/server.py --port 9009
```

Recommended local benchmark-testing defaults:

- `BENCHMARK_NAME=officeqa`
- `BENCHMARK_STATELESS=1`
- `ENABLE_RUN_TRACER=1`
- `TRACE_MAX_RECENT=5`
- `ENABLE_AGENT_MEMORY=0`

## Validation

Run the full test suite:

```bash
uv run pytest tests/
```

Useful smoke commands:

- `uv run pytest tests/test_engine_runtime.py -q`
- `BENCHMARK_STATELESS=1 uv run python scripts/run_benchmark_stateless_smoke.py`
- `BENCHMARK_NAME=officeqa BENCHMARK_STATELESS=1 uv run python scripts/run_officeqa_regression.py --smoke`

The OfficeQA regression runner writes JSON reports under `Results&traces/` with:

- subsystem classification: `routing`, `retrieval`, `extraction`, `compute`, `validation`, `formatting`, or `pass`
- chosen sources and extracted tables
- compute ledger
- final answer

## Project Structure

- `src/server.py`: A2A Starlette server entrypoint
- `src/executor.py`: bridge between A2A requests and the OfficeQA graph
- `src/agent/`: core runtime implementation
- `docs/`: analysis, plan, walkthrough, and progress docs
- `data/officeqa/`: local landing zone for the untracked OfficeQA corpus
