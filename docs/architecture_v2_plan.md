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

## Phase 4: AgentPrune & AgentTaxo Guardrails

_Theme: Communication Efficiency & Robustness_

Multi-agent systems suffer from a severe "communication tax" (passing the entire conversation history back and forth between agents).

- **State Pruning**: We will implement pruning filters. When the Verifier sends feedback to the Executor, it _only_ sends the critique, not the generic system prompt or extraneous context.
- **Security Guardrails**: The Coordinator will run lightweight checks to detect adversarial prompt injections (common in complex evaluation datasets) before passing the task to specialized sub-agents.

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
