# CoreLink AI: OfficeQA Reasoning Engine

CoreLink AI started as a broader finance-first reasoning engine, but the active project direction is now OfficeQA only. The repository is being reworked into a document-grounded Treasury Bulletin reasoning engine built on LangGraph and MCP-style tool execution.

## 🎯 Overview

The current goal is to solve OfficeQA reliably:

- retrieve from the Treasury Bulletin corpus
- extract the correct table or page evidence
- compute deterministically from extracted values
- emit the exact final answer contract expected by the benchmark

The canonical docs for the transition are:

- `docs/officeqa_integration_plan.md`: architecture analysis and failure report
- `docs/officeqa_execution_plan.md`: implementation backlog and phase tracker
- `docs/progress.md`: ongoing execution log

## 🏗️ Technical Architecture

The old finance-first architecture diagram is intentionally not treated as canonical anymore. Use `docs/officeqa_integration_plan.md` and `docs/officeqa_execution_plan.md` for the current OfficeQA direction and implementation flow.

### Core Components

- **Benchmark Adapter**: Activates OfficeQA-specific runtime rules and answer contracts.
- **Corpus Retrieval Layer**: Finds the right Treasury Bulletin pages, tables, and candidate evidence.
- **Structured Extraction Layer**: Converts retrieved document content into provenance-backed values.
- **Deterministic Compute Layer**: Performs exact benchmark calculations over extracted values.
- **Validator & Output Adapter**: Checks evidence and compute support before enforcing the final answer contract.

## 💎 Handling Finance Complexity

OfficeQA tasks involve deep document retrieval, table extraction, period alignment, and exact computation. CoreLink AI is being reworked to address these by:

- **Corpus-First Retrieval**: Using indexed Treasury Bulletin artifacts instead of broad web search.
- **Phase Separation**: Explicitly separating retrieval, extraction, validation, computation, and final answer formatting.
- **Deterministic Compute**: Preferring provenance-backed operators over prompt-only arithmetic.
- **Assumed Transparency**: Keeping source and compute provenance explicit throughout the execution trace.

## 💼 Supported Workloads

- **OfficeQA**: document-grounded financial reasoning over Treasury Bulletin artifacts

## 🚀 Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) (recommended) or Python 3.13+
- Your preferred OpenAI-compatible API key (e.g., Nebius, Groq, OpenAI)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/CoreLink-AI.git
    cd CoreLink-AI
    ```

2.  **Initialize the environment**:
    ```bash
    uv sync
    ```

3.  **Configure Environment Variables**:
    ```bash
    cp .env.example .env
    # Edit .env with your API keys and model preferences
    ```

### OfficeQA Corpus Connection

The runtime connects to the OfficeQA corpus through a filesystem root plus a generated manifest/index.

1.  Clone or download the OfficeQA corpus locally.
2.  Point `OFFICEQA_CORPUS_DIR` at the parsed/text corpus root you want to use.
    Example:
    ```bash
    OFFICEQA_CORPUS_DIR=/path/to/officeqa/treasury_bulletins_parsed
    ```
3.  Build the local index:
    ```bash
    uv run python scripts/build_officeqa_index.py --corpus-root "$OFFICEQA_CORPUS_DIR"
    ```
4.  The script writes a persistent manifest under `OFFICEQA_INDEX_DIR` or, by default, under `OFFICEQA_CORPUS_DIR/.officeqa_index/`.
5.  Verify the bundle before running the server:
    ```bash
    uv run python scripts/verify_officeqa_corpus.py --corpus-root "$OFFICEQA_CORPUS_DIR"
    ```

Current runtime flow:

- `search_reference_corpus` first looks for the OfficeQA manifest/index.
- If the index exists, search runs against indexed metadata such as years, section titles, table headers, row labels, unit hints, and preview text.
- Search hits return `document_id` plus the relative corpus path.
- `fetch_corpus_document` resolves that `document_id` back to the indexed artifact and reads the source file from the local corpus root.

### Competition Deployment

Do not assume Judge or A2A will expose the Treasury corpus to your agent.

- Keep the OfficeQA corpus out of git history.
- Package the corpus and `.officeqa_index/` with the deployment artifact, container image, or mounted read-only volume.
- Set `OFFICEQA_CORPUS_DIR` to that packaged or mounted path at startup.
- When `COMPETITION_MODE=1` and `BENCHMARK_NAME=officeqa`, the server now fails fast at startup if the corpus root or built index is missing.
- Judge MCP document tools remain optional auxiliary surfaces, not the primary data path.

4.  **Start the A2A Server**:
    ```bash
    uv run src/server.py --port 9009
    ```

## 🧪 Validation & Testing

Run the full test suite to verify A2A compliance:

```bash
uv run pytest tests/
```

### Smoke Tests

- **Deterministic Logic**: `uv run pytest tests/test_engine_runtime.py -q`
- **Live LLM Reasoning**: `uv run python scripts/run_live_engine_smoke.py`
- **Stateless Benchmark Mode**: `BENCHMARK_STATELESS=1 uv run python scripts/run_benchmark_stateless_smoke.py`

## 🛠️ Project Structure

- `src/server.py`: A2A Starlette server entrypoint.
- `src/executor.py`: Bridge between A2A requests and the Reasoning Brain.
- `src/agent/`: Core engine implementation (Graph, Nodes, Solver).
- `src/mcp_servers/`: Local Model Context Protocol server implementations.
- `docs/`: Technical specifications, milestone reports, and design docs.

---
*Autonomous Finance Reasoning Engine*
