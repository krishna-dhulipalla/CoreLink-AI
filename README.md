# CoreLinkAI

CoreLinkAI is a document-grounded financial reasoning system. It is designed to answer financial questions from source documents by combining corpus retrieval, structured extraction, deterministic computation, validation, and strict output formatting.

## Overview

CoreLinkAI is built for financial reasoning tasks where the answer must come from documents, not from unsupported model memory. The system is intended for workloads such as:

- financial data extraction from reports and tables
- multi-period and inflation-adjusted calculations
- statistical analysis over document-derived series
- forecasting and trend questions grounded in historical records
- weighted averages, risk-style metrics, and similar finance computations

The current benchmark target is OfficeQA, which is used as a testing environment for this runtime.

## How It Works

The active runtime flow is:

`A2A request -> benchmark adapter -> corpus retrieval -> structured extraction -> deterministic compute -> validator/reviewer -> output adapter`

Core runtime layers:

- document retrieval over local or packaged corpora
- structured evidence extraction with provenance
- deterministic compute for supported financial operations
- validation before final answer formatting
- exact answer adaptation for benchmark or task contracts

## Current Benchmark

CoreLinkAI is currently evaluated against OfficeQA.

OfficeQA is a benchmark, not the product identity of the system. It is useful because it stresses the kinds of document-grounded finance tasks CoreLinkAI is meant to handle, including:

- simple extraction
- multi-year inflation-adjusted calculations
- statistical analysis such as regression, correlation, and standard deviation
- time-series forecasting
- weighted averages and risk-style financial metrics

## Model Configuration

The runtime uses role-based model selection from `src/agent/model_config.py`.

Recommended setup:

- use a strong document-grounded solver
- use a strong reviewer when ambiguity, scope, or long-context failures appear
- keep adapter and reflection models lighter unless they become bottlenecks

The system supports role-specific overrides through `.env.example`, including:

- `SOLVER_MODEL`
- `REVIEWER_MODEL`
- `DOCUMENT_SOLVER_MODEL`
- `LONG_CONTEXT_SOLVER_MODEL`
- `DOCUMENT_REVIEWER_MODEL`

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
- `MODEL_PROFILE=officeqa`

If you are running the current benchmark flow, also set:

- `BENCHMARK_NAME=officeqa`

## Local OfficeQA Data

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

## Competition And Deployment

Do not assume the evaluation environment exposes the corpus to the agent.

- keep the OfficeQA corpus out of git history
- package or mount the corpus for benchmark runs
- point `OFFICEQA_CORPUS_DIR` at that packaged path
- when `COMPETITION_MODE=1` and `BENCHMARK_NAME=officeqa`, startup fails fast if corpus or index is missing

Judge tools are optional auxiliary surfaces, not the primary data path.

## Running

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

The OfficeQA regression runner writes JSON reports under `Results&traces/` with subsystem classification, chosen sources, extracted tables, compute ledgers, and final answers.

## Project Structure

- `src/server.py`: A2A Starlette server entrypoint
- `src/executor.py`: bridge between A2A requests and the runtime graph
- `src/agent/`: core runtime implementation
- `data/officeqa/`: local landing zone for the untracked OfficeQA corpus
