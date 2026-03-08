# Architecture v2: Agile Multi-Agent Evolution Plan

The current architecture (a monolithic ReAct StateGraph) has proven reliable for basic tasks but lacks the scalability, cost-efficiency, and rigor required for a 246-task benchmark over an 89,000-page corpus (OfficeQA).

Inspired by state-of-the-art frameworks (PRIME, AgentNet, MaAS, and AgentPrune) as well as the competitor `purple-agent-finance-worker`, we will transition to a decentralized, dynamically routed Multi-Agent System (MAS).

To ensure stability, we will follow an **Agile Methodology**, evolving the system step-by-step and validating against benchmarks after each phase.

---

## Phase 1: The MaAS-Inspired Coordinator (Dynamic Routing)

_Theme: Cost-Aware Efficiency & Decentralization_

Instead of forcing every task through a heavy ReAct loop, we will implement a lightweight **Coordinator/Router Agent** at the entry point.

- **Classification**: The Coordinator quickly assesses the query type (e.g., Simple Calculation, Deep Retrieval, Multi-Step Algorithmic, Formatting).
- **Early Exit / Direct Execution**: Simple queries bypass the heavy multi-agent loop entirely, saving tokens and time (MaAS).
- **Sub-Agent Dispatch**: Complex queries are routed to specialized sub-agents (e.g., an OfficeQA Retrieval Agent, an Algorithmic Finance Agent).
- **Format Normalizer**: Adopting the competitor's 100% compliance strategy, the Coordinator acts as a strict final gate, invoking a dedicated, cheap LLM pass whose solely purpose is JSON/XML formatting.

## Phase 2: The PRIME-Inspired Executor-Verifier Triad

_Theme: Rigorous Algorithmic Reasoning & Feedback Loops_

For complex tasks routed by the Coordinator, we will replace the standard `reasoner -> tool_executor -> reflector` loop with a localized **PRIME Triad** where the Verifier is the step-level gatekeeper.

1. **Executor Agent**: Focuses strictly on constructive reasoning and selecting actions (tool calls).
2. **Verifier Agent**: A separate LLM node that evaluates the Executor's _every step_ (not just the final answer) against constraints. It emits a structured verdict: `PASS`, `REVISE`, or `BACKTRACK`, along with machine-readable reasons.
3. **Checkpoint Stack & Backtracking**: We will maintain a real checkpoint stack of verified states. If the Verifier issues a `BACKTRACK`, the state reverts to the last verified step instead of trapping the LLM in a failed "try again" loop.
   _(Note: To maintain a lean competition runtime, we are excluding PRIME's heavy RL/GRPO search features for now)._

## Phase 3: AgentNet-Inspired Execution Memory & Repair Reuse

_Theme: Local Fragments & Learned Recovery_

Instead of generic trajectory RAG, memory is role-specific to the Coordinator, Executor, and Verifier loop established in Sprint 2.

- **Coordinator Memory**: Stores compact records (task summary, operator layers, cost, success). Used before planning.
- **Executor Memory**: Stores local execution fragments (partial context, tool chosen, args, quality). Retrieves top-k compact structured hints instead of raw dumps.
- **Verifier/Repair Memory**: Stores failure patterns, verdicts, and repair strategies (revise vs backtrack success). Enables learned recovery.
- **Strict Admission & Bounded Storage**: Only verified successful fragments or high-signal backtrack/revise recoveries are stored. Begins with SQLite or JSON.

## Phase 4: Runtime Guardrails, Pruning, and Budget Control

_Theme: Hardening the Runtime for Reliability & Efficiency_

This sprint tightens the existing Coordinator → Executor → Verifier → Memory architecture rather than adding new agent capabilities. Inspired by AgentPrune's spatial-temporal message pruning and adversarial isolation, MaAS's cost-constrained early exits, and PRIME's state stack compaction.

### 4A — State Pruning at Node Boundaries

- **Coordinator → Executor**: Strip internal system warnings, stale tool results, and memory hint blocks before the Reasoner receives the conversation. Only pass the HumanMessage, the system prompt, and the latest verified context window.
- **Verifier → Executor (on REVISE/BACKTRACK)**: The warning message includes only the verdict reasoning and repair hints, never the full message history or the raw checkpoint payload.
- **Persisted History**: `runner.py` already strips reflection/warning messages. Extend to also strip injected memory hint SystemMessages and expired tool-result ToolMessages to keep multi-turn history lean.
- **Memory Records**: Before writing `ExecutorMemory`/`VerifierMemory`, truncate `arguments_pattern` and `failure_pattern` fields to a configurable max length to prevent bloat.

### 4B — Memory Hygiene & Deduplication

- **Near-duplicate suppression**: Before storing a new memory record, check if a record with the same `task_signature` + `tool_used` (executor) or `failure_pattern` prefix (verifier) already exists. Skip the write if the existing record is recent enough (configurable staleness window).
- **Compaction pass**: On store init (or periodically), merge near-identical router records by keeping only the best-cost successful run per task_signature.
- **Multi-turn noise filter**: Do not store executor/verifier memory for trivial runs (e.g., direct_answer path, runs under 2 steps).

### 4C — Per-Run Budget Enforcement

- **Max tool calls**: Enforce a hard cap (`MAX_TOOL_CALLS`, default 15) in `tool_executor`. After the cap, force the reasoner to produce a final answer without tools.
- **Max verifier cycles**: Enforce `MAX_REVISE_CYCLES` (default 3) and `MAX_BACKTRACK_CYCLES` (default 2). After exhaustion, accept the best available answer and log a budget-exit event.
- **Memory hint token budget**: Cap the total injected hint tokens per node to `MAX_HINT_TOKENS` (default 200). Truncate or skip hints if they exceed the budget.
- **Budget-exit visibility**: All budget exhaustion events appear in the `steps` list returned by `run_agent` with a clear `budget_exit` node type.

### 4D — Guardrails Before Tool Use

- **Content sanitization**: After tool execution returns, scan the raw output for prompt-injection signatures (e.g., `IGNORE PREVIOUS INSTRUCTIONS`, `<system>`, role reassignment patterns). If detected, replace the content with a sanitized summary and log the event.
- **Tool-description hijacking**: Before the Reasoner's LLM call, validate that dynamically loaded MCP tool descriptions haven't been tampered with (length cap, no instruction-like content).
- **Untrusted-content tagging**: ToolMessage content from external file fetches is wrapped with `[EXTERNAL CONTENT START]` / `[EXTERNAL CONTENT END]` markers so the LLM treats it as data, not instruction.

### 4E — Observability & Logging

- **Structured pruning log**: Log why each pruning action happened (e.g., "stripped 3 stale tool results from coordinator→executor handoff").
- **Memory injection log**: Log which hints were injected or skipped per node, with reason (e.g., "coordinator: 2 router hints injected" or "executor: hints skipped, no match").
- **Budget log**: Log when a budget cap is hit and what action was taken.
- **Guardrail log**: Log when content sanitization fires, including the pattern matched and the original content length.

---

## What We Build First (Agile Sprint 1) - **[COMPLETED]**

We successfully built Phase 1 into the [src/agent.py](file:///c:/Users/vamsi/OneDrive/Desktop/Gtihub_repos/Project-Pulse-Generalist-A2A-Reasoning-Engine/src/agent.py) graph:
~~1. Break [src/agent.py](file:///c:/Users/vamsi/OneDrive/Desktop/Gtihub_repos/Project-Pulse-Generalist-A2A-Reasoning-Engine/src/agent.py) into a lightweight **Coordinator** node.~~
~~2. Create dedicated **Execution Paths** (e.g., `fast_compute` vs `heavy_research`).~~
~~3. Add the **Format Normalization Pass** at the end of the graph to guarantee benchmark strictness.~~

## Sprint 1.5: MaAS-Lite Runtime Foundations - **[COMPLETED]**

1. ~~Operator Abstraction (`agent/operators.py`): 6 operators with metadata.~~
2. ~~Layered Policy: `RouteDecision` emits `layers`, `confidence`, `needs_formatting`.~~
3. ~~Cost Tracker (`agent/cost.py`): Per-node token/cost/latency accounting.~~
4. ~~Conditional Format Normalizer: Skips LLM when `needs_formatting=False`.~~
5. ~~24 New Tests: Operator registry, cost tracker, routing, prompt safety.~~

**Ready for Sprint 2 (Phase 2: PRIME Triads)**.
