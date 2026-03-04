# Purple Agent: Brain-to-Arm Architecture

## Overview

The Purple Agent is designed as a protocol-compliant reasoning engine for the AgentX-AgentBeats Competition. The architecture relies on the **"Brain-to-Arm"** paradigm:

- **The Brain**: A LangGraph-powered reasoning engine executing a **Plan-Act-Learn** loop.
- **The Arm**: A Model Context Protocol (MCP) client that allows the agent to dynamically discover and invoke tools provided by the Green Agent (evaluator).

## 1. The "Plan-Act-Learn" Loop

Based on the methodologies from _Agentic Reasoning for Large Language Models_ (arXiv 2601.12538), the core control flow transitions the LLM from a passive text generator into an autonomous agent:

- **Plan (Deliberation)**: The agent analyzes the task environment and formulates a multi-step plan. For complex benchmark tasks, it uses **Orchestration-Based Tool-Integration** to sequence tool dependencies properly.
- **Act (Interaction)**: The agent interacts with the Green Agent using the MCP tools. It applies **In-Context Tool-Integration** to generate action trajectories in a "thought-tool-observation" structure.
- **Learn (Feedback)**: The agent uses **Self-Evolving Reasoning** to improve its success rate during a session.

## 2. Foundational Reasoning (Tool-Use Optimization)

The agent operates in a domain-agnostic paradigm. Tools are **never hardcoded** — they are discovered at runtime via MCP:

- **Tool Discovery**: On initialization, the Purple Agent connects to all configured MCP servers via `MultiServerMCPClient` and dynamically registers their tools into the LangGraph state. New domains require only a `.env` change.
- **Multi-Server Architecture**: Domain-specific logic lives in **separate, independent MCP servers**. Current finance servers:
  - `finance` — Black-Scholes pricing, Greeks, mispricing analysis
  - `options_chain` — Options chains, IV surfaces, multi-leg strategy analysis
  - `trading_sim` — Paper trading engine (portfolio management, simulated execution)
  - `risk_metrics` — VaR, Sharpe/Sortino/Calmar ratios, portfolio Greeks, stress testing
- **Prompt Construction**: Tool definitions are injected into the context dynamically from the tool's `description` field — the system prompt contains no domain-specific routing rules.
- **Action Routing**: LangGraph routes sub-tasks to tool-execution nodes, processing the tool calls and passing the results back to the reasoning nodes.

## 3. Self-Evolving Reasoning (Feedback Loops)

To maximize accuracy, the agent implements a robust feedback loop:

- **Reflective Feedback**: Before submitting a final answer, the agent critiques its own intermediate reasoning for logical errors. If validation fails, it generates revised steps.
- **Tool-Failure Recovery**: Explicit failure state (`tool_fail_count`, `last_tool_signature`) in `AgentState` prevents the agent from re-calling the same failing tool. After 2 consecutive failures, the graph routes to the reflector to force a strategy change.
- **Validator-Driven Feedback**: If the environment provides a failure signal, the agent repeatedly re-evaluates and submits independent attempts.

## 4. Memory Management & Observation Masking (Windowing)

Long A2A tasks can quickly exhaust the LLM's context window. To save tokens and compute costs, we implement memory pruning and observation masking:

- **Observation Masking**:
  - Truncation of verbose tool outputs (e.g., HTML, long lists).
  - Creating "compressed representations" of the state where only the relevant diffs or summaries are kept in context.
- **Summarize-and-Forget**:
  - LangGraph's short-term memory (message list) is periodically compressed.
  - Older "thought-tool-observation" steps are summarized into a condensed "learned facts" node, and the raw messages are removed from the active context window.
  - Group-based windowing (`_group_messages()`) ensures `tool_calls` + `ToolMessage` pairs are never split by compression boundaries.
- **Structured Outputs**: Finance tools emit a `STRUCTURED_RESULTS:` prefix block with exact key-value pairs that the agent is instructed to copy verbatim — preventing LLM paraphrasing from corrupting graded numeric fields.
