# CoreLink AI: Architecture & Foundations Summary

This document summarizes the core foundational systems we have built for the AgentX-AgentBeats Competition to ensure a robust, resilient, and stateful generalist reasoning agent.

## 1. Core Architecture (LangGraph ReAct Engine)

We implemented our agent using **LangGraph**, structuring it around a **Plan-Act-Learn** loop. The execution flows through a graph of specialized nodes:

- **`reasoner`**: The "Brain". Uses OpenAI (`gpt-4o-mini`) to decide the next step, emit tool calls, or formulate a draft answer.
- **`tool_executor`**: The "Arm". Executes requested tools and formats the results.
- **`context_window`**: Applies context masking and compression to keep the conversation within token limits.
- **`reflector`**: A self-correction mechanism to critique draft answers before final submission.

## 2. Dynamic Tooling (MCP Integration)

Instead of hardcoding all tools, we integrated a **Model Context Protocol (MCP)** client.

- This enables the agent to dynamically discover, evaluate, and invoke tools via standard MCP servers.
- It provides a highly extensible foundation for interacting with local filesystems, external APIs, and benchmark environments ("green agents").

## 3. Advanced Context Management (Observation Masking)

To prevent the agent from crashing on large tasks or exceeding LLM context windows, we built a robust token management system (`src/context_manager.py`):

- **Tool Output Truncation**: Unusually long tool outputs are automatically hard-capped with a `[TRUNCATED]` marker to prevent context blowout.
- **Tool-Call-Safe Windowing ("Summarize-and-Forget")**: When the token count exceeds the configured budget, older messages are compressed into a single, dense `SystemMessage` summary.
  - _Safety Guarantee_: Our windowing groups `AIMessage(tool_calls)` with their corresponding `ToolMessage`s into inseparable atomic blocks. This guarantees we never sever tool-call adjacency, entirely preventing fatal LLM API errors during multi-turn compressions.
  - _Front-Gate Pruning_: Context is verified and pruned before entering the graph, guaranteeing the reasoner never receives an oversized payload.

## 4. Reflective Feedback Loop

We implemented a self-correction mechanism to ensure high-quality outputs:

- Before the agent returns a final answer to the A2A server, a separate `reflector` LLM call grades the answer (PASS or REVISE) against the original prompt.
- If the answer is lacking, the LangGraph routes back to the `reasoner` with the critique, forcing a revision.
- A configurable retry limit prevents infinite loops.
- Internal `[Reflection]` messages are stripped from persistent history to keep the long-term context clean.

## 5. Stateful Multi-Turn Conversations

The agent natively supports long-running, multi-turn interactions:

- **`ConversationStore`**: Persists message history across distinct A2A HTTP requests using a `context_id`.
- Automatically tracks sessions and includes a TTL (Time-To-Live) cleanup mechanism for abandoned contexts.
- The `AgentExecutor` seamlessly retrieves past history, appends the new user request, runs the LangGraph engine, and persists the updated history.

## 6. Testing & Reliability Assurance

- We maintain a dedicated suite of unit tests (`tests/test_features.py`) that run independently of the live server, specifically verifying:
  - Token counting accuracy and truncation limits.
  - Group-based windowing safety.
  - Store TTL expiry and context separation.
- The agent fully passes the suite of A2A conformance integration tests.

---

**Status:** Foundations are complete, stable, and stress-tested. The architecture is ready for integration with benchmark agents and complex coding scenarios.
