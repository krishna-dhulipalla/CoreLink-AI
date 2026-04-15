# CoreLink AI

CoreLink AI is a robust, Generalist Agent-to-Agent (A2A) reasoning engine built for evidence-grounded analytical workloads. Unlike systems that rely on unconstrained generation or raw LLM recall, the engine emphasizes deterministic compute, dynamic search strategies, and strict policy-driven tool use to prioritize correctness, transparency, and provenance.

<p align="center">
  <img src="assets/corelink_diagram.png" alt="CoreLink AI System Architecture" width="100%">
</p>

## Overview

Most agent systems fail by either relying too heavily on model recall instead of evidence, or using tools without enough policy around when to search, compute, retry, and stop. 

CoreLink AI bridges this gap with a runtime designed to:
- Prefer grounded evidence over unsupported synthesis.
- Mutate retrieval strategies dynamically when a current search regime stalls.
- Validate answerability before allowing a failure endpoint.
- Opt for deterministic compute and fallback scripting over free-form math.

## Features

- **Evidence-First Retrieval:** Retrieval is not a single fixed path. The engine utilizes a dynamic retrieval retry policy that shifts seamlessly across different search regimes (e.g., table-first, text-first, hybrid, multi-document) based on execution feedback.
- **Deterministic Compute & Sandbox Execution:** When a question requires explicit aggregation or transformation, the runtime bypasses raw LLM math. It normalizes retrieved material into structured evidence (tables, rows, cells). If an existing tool is insufficient, the engine can autonomously generate and execute custom Python scripts within an isolated sandbox to compute the answer reliably.
- **Failure Recovery & Remediation:** LLMs act as arbiters and planners at explicit boundaries. Through a "Review before Finalization" cycle, any answer must meet rigorous sufficiency thresholds. If an answer lacks evidence or hits an exhaustion condition, the model will orchestrate a repair—rotating its search strategy or computing dynamically—rather than hallucinating an unsupported fact.
- **Stateful Graph Execution:** Built on a solid, stateful execution architecture, the agent ensures strict adherence to exhaustion proofs so it does not loop endlessly or silently fail. It produces computationally valid evidence or gracefully halts with a definitive, explainable halt condition.

## Architecture

CoreLink AI consists of four primary operational modules:

1. **Director/Orchestrator (Planning):** Acts as the central brain. It receives queries, breaks down analytical constraints, and issues structured instructions to the underlying mechanisms.
2. **Perception/Retrieval (Data Sources):** Responsible for interfacing with data APIs, documents, or vector indices. It adapts its search strategy (e.g., querying tables vs. raw text) based on signals from the orchestrator.
3. **Function/Tool Calling (Processing):** Takes retrieved segments and applies deterministic tools or sandboxed Python scripts. It converts abstract text questions into verifiable parameter execution.
4. **Validation/Review (Arbitration):** Cross-checks interim outputs against the original query to decide whether to finalize an answer or loop back to the Orchestrator with an exhaustion proof or failure context.

## Installation & Getting Started

### Prerequisites
- Python 3.13+
- `uv` (for fast Python package management)
- Git
- An OpenAI-compatible API key

### 1. Clone the Repository

```bash
git clone https://github.com/krishna-dhulipalla/CoreLink-AI.git
cd CoreLink-AI
```

### 2. Install Dependencies

Use `uv` to quickly sync the project environment:

```bash
uv sync
```

### 3. Setup Configuration

Create your local environment file by copying the example:

```bash
cp .env.example .env
```

Ensure your provider credentials are set. Open `.env` and set at minimum:

```ini
OPENAI_API_KEY=your_key_here
```

### 4. Run Locally

You can verify the runtime behaves correctly by running the live engine smoke tests:

```bash
uv run python scripts/run_live_engine_smoke.py
```

### 5. Launch the Server

Start the A2A compatible streaming server using the modular entrypoint:

```bash
uv run python -m engine.a2a.server --host 127.0.0.1 --port 9009
```

## Configuration

The runtime is configured securely through environment variables in `.env`. Common settings include:
- **Provider Access:** `OPENAI_API_KEY`, `OPENAI_BASE_URL`
- **Model Overrides:** Option to specify targeted models for solvers and reviewers (e.g., `SOLVER_MODEL`, `REVIEWER_MODEL`).
- **Runtime Controls:** Adjust execution boundaries with variables like `MAX_TOOL_CALLS`, `MAX_REVISE_CYCLES`, and `MAX_CONTEXT_TOKENS`.

## Evaluation & Adaptability

CoreLink AI has been hardened using benchmark-driven testing (such as the document-heavy OfficeQA dataset). However, the engine itself is highly adaptable and decoupled from any specific format. It seamlessly wraps around custom environments, executing complex orchestrations while maintaining a highly transparent, stateless, and automatable execution trace. 

## Project Structure

A high-level overview of the reorganized engine namespace:

```text
CoreLink-AI/
├── data/persistence/       # Local database storage
├── src/engine/             # Main application namespace
│   ├── a2a/                # A2A streaming server and messenger
│   ├── mcp/                # Model Context Protocol bridge and client
│   ├── runtime/            # Shared utilities and state management
│   └── agent/              # Core reasoning engine logic
│       ├── nodes/          # Graph execution nodes
│       ├── tools/          # Specialized agent tools
│       └── graph.py        # Central state machine definition
├── scripts/                # Evaluation and smoke testing harnesses
├── tests/                  # Unit and E2E regression tests
├── pyproject.toml          # Project dependencies
└── Dockerfile              # Containerization specs
```

## Acknowledgements

CoreLink AI's structural design draws conceptual inspiration from advanced agent workflow architectures. Ensure you configure your required API keys correctly within the isolated environment before extensive execution.
