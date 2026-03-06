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

For complex tasks routed by the Coordinator, we will replace the standard `reasoner -> tool_executor` loop with a localized **PRIME Triad**:

1. **Executor Agent**: Focuses strictly on constructive reasoning and selecting actions (e.g., writing the Black-Scholes formula or querying a specific SEC filing).
2. **Verifier Agent**: A separate LLM node that strictly evaluates the Executor's state against constraints (e.g., "Did the option price match the retrieved strike?") and provides immediate, dense feedback.
3. **Backtracking Mechanism**: If the Verifier repeatedly rejects the Executor's work, the Coordinator orchestrates a "backtrack", reverting the state to the last known good step instead of blindly looping into a `RecursionError`.

## Phase 3: AgentNet-Inspired Task Memory (RAG)

_Theme: Self-Evolving Expertise_

As the agent successfully completes benchmark tasks, we want it to learn.

- **Trajectory Storage**: Successful tool sequences and strategies are stored in a local vector database or JSON registry. _(Note: Storage limits and token-cost for retrieval will be carefully analyzed before full deployment)._
- **RAG Pre-Warming**: When the Coordinator receives a new, similar task, it retrieves past successful trajectories and injects them into the Executor's prompt as few-shot examples.

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

**Ready for Sprint 2 (Phase 2: PRIME Triads)**. We will proceed to build localized Executor and Verifier pairs to replace the monolithic reasoning loop for `heavy_research` tasks.
