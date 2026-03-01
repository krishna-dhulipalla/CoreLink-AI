# Project Progress & Agent Handoff (progress.md)

## Purpose

This document is the "brain space" for agents to communicate and leave context. Instead of reading noisy terminal output, new agents will read this file to understand the current state of the project, who did what, what needs to be done next, and what are the major milestones/blockers.

## Structure & Rules

This file operates in a "Chat" structure. Whenever an agent finishes a major unit of work, it will append a new "Chat" block.

- **Maximum Limit:** The `[Recent Chats]` section can only hold a maximum of 20 chats.
- **Windowing/Summarization:** If you are adding the 21th chat, you MUST remove the oldest chat(s) and concisely summarize their value into the `[Long-Term Memory]` section below. Keep the long-term memory brief but actionable.

---

## 🧠 [Long-Term Memory]

_(When the chat limit is reached, older context is summarized here.)_

- Initialized project with `RDI-Foundation/agent-template`.
- Agreed on "Brain-to-Arm" architecture (LangGraph for logic, MCP for tool execution).
- Documented specifications in `DESIGN.md`, `A2A_INTERFACE_SPEC.md`, and `MILESTONES.md`.

---

## 💬 [Recent Chats]

### Chat 1: Foundation Planning

- **Role:** Technical Planner
- **Actions Taken:** Analyzed requirements and agent-template. Created reference docs (`DESIGN.md`, `A2A_INTERFACE_SPEC.md`, `MILESTONES.md`, `project_plan_details.md`) and initial `progress.md`.
- **Blockers:** None currently.
- **Handoff Notes:** The foundational logic is planned. The next agent should focus on the codebase itself, specifically setting up the local A2A testing environment and beginning the `LangGraphAgentExecutor` implementation.

### Chat 2: Initial Agent Logic Implementation

- **Role:** Coder
- **Actions Taken:** Implemented the LangGraph ReAct reasoning engine. Updated `pyproject.toml` (added langgraph, langchain-openai, python-dotenv), rewrote `agent.py` (StateGraph with reasoner/tool_executor nodes, conditional routing, built-in calculator & time tools), rewrote `executor.py` (A2A-to-LangGraph bridge with streaming status), updated `server.py` (CoreLink AI branding, skill metadata). All 3 A2A conformance tests pass.
- **Blockers:** None.
- **Handoff Notes:** The basic ReAct loop works end-to-end. Next priorities: (1) MCP client integration for dynamic tool discovery from Green Agents, (2) Observation masking / message windowing for long tasks, (3) Reflective feedback loop before final answer submission. The `.env` file must contain `OPENAI_API_KEY`.

### Chat 3: MCP Client Integration & CI Hold

- **Role:** Coder
- **Actions Taken:** Integrated `langchain-mcp-adapters` for dynamic tool discovery. Created `src/mcp_client.py` (wraps `MultiServerMCPClient`, supports HTTP and stdio transports, configurable via `MCP_SERVER_URLS` / `MCP_SERVER_STDIO` env vars). Updated `agent.py` (`build_agent_graph()` now accepts `external_tools` param, merged with built-ins). Updated `executor.py` to load MCP tools at init and pass to graph. Disabled CI workflow auto-triggers (`workflow_dispatch` only). All 3 A2A conformance tests pass.
- **Blockers:** None.
- **Handoff Notes:** MCP integration is in place but untested with a live MCP server. To test: set `MCP_SERVER_URLS=math=http://localhost:3000/mcp` in `.env` and run a local MCP math server. Next priorities: (1) Observation masking for long contexts, (2) Reflective feedback loop, (3) End-to-end test with a real MCP server.

### Chat 4: Tester Review & MCP Test Coverage

- **Role:** Tester
- **Actions Taken:** Reviewed the current implementation against `docs/` and runtime code. Confirmed the repo contains an A2A agent with MCP client integration, not an MCP server implementation. Identified key risks: Windows `MCP_SERVER_STDIO` path parsing can break on drive-letter paths, MCP loading is static at `Executor` initialization, and current tests only cover A2A conformance. Added `tests/test_mcp_integration.py` with three checks: direct MCP discovery from env, executor registration of discovered tools, and a live end-to-end streamed-status assertion gated by `EXPECTED_MCP_TOOL_NAME`.
- **Blockers:** Live MCP verification still depends on a running external MCP server plus a prompt that uniquely requires one of its tools. If the prompt overlaps with built-in tools, the model may choose the built-in path and create a false negative/positive signal.
- **Handoff Notes:** Preferred live path is HTTP MCP via `MCP_SERVER_URLS`; use `MCP_SERVER_STDIO` on Windows only after fixing path parsing in `src/mcp_client.py`. For end-to-end testing, run the external MCP server first, then start `uv run src/server.py`, then run `uv run pytest -k mcp --agent-url http://localhost:9009`. Set `EXPECTED_MCP_TOOL_NAME` and `MCP_TEST_PROMPT` in the environment to make the live assertion deterministic.

### Chat 5: Local Mock MCP Server & Windows Stdio Fix

- **Role:** Coder
- **Actions Taken:** Added `src/mock_mcp_server.py`, a minimal local MCP server built with `FastMCP` that exposes deterministic `echo_magic` and `sum_magic` tools for end-to-end testing. Updated `src/mcp_client.py` so `MCP_SERVER_STDIO` now accepts a Windows-safe shell-style command (`python src/mock_mcp_server.py --transport stdio`) and remains backward-compatible with the older colon-delimited format while repairing a drive-letter first argument.
- **Blockers:** The live end-to-end assertion still depends on the LLM choosing the external tool. Use a prompt that explicitly names `echo_magic` or `sum_magic`, and set `EXPECTED_MCP_TOOL_NAME` accordingly.
- **Handoff Notes:** Simplest verification path is HTTP: run `uv run src/mock_mcp_server.py --host 127.0.0.1 --port 3001`, set `MCP_SERVER_URLS=mock=http://127.0.0.1:3001/mcp`, then start the A2A server and run the MCP tests. Stdio mode is now viable on Windows with `MCP_SERVER_STDIO=mock=python src/mock_mcp_server.py --transport stdio`.

### Chat 6: Executor MCP Initialization Fix

- **Role:** Coder
- **Actions Taken:** Fixed `src/executor.py` MCP initialization so it now uses `asyncio.run(load_mcp_tools_from_env())` when `Executor()` is constructed outside a running event loop. This avoids the test failure where `asyncio.run(...)` in the test clears the current event loop and the old `asyncio.get_event_loop()` path fell back to an empty MCP tool list.
- **Blockers:** If `Executor()` is ever constructed inside an already-running event loop, MCP loading is still intentionally skipped because `__init__` is synchronous. The current server startup path is safe because construction happens before `uvicorn` starts its loop.
- **Handoff Notes:** Re-run `uv run pytest -k mcp --agent-url http://127.0.0.1:9009 -v`. The `test_executor_registers_mcp_tools` assertion should now match the direct discovery result.

### Chat 7: Observation Masking for Long Contexts

- **Role:** Coder
- **Actions Taken:** Implemented observation masking / message windowing in the LangGraph agent. Created `src/context_manager.py` with three capabilities: (1) `count_tokens()` using tiktoken for client-side token counting, (2) `truncate_tool_output()` caps verbose tool responses at `MAX_TOOL_OUTPUT_CHARS` (default 4000) with a `[TRUNCATED]` marker, (3) `summarize_and_window()` implements the Summarize-and-Forget strategy — compresses older messages into a summary when token count exceeds `MAX_CONTEXT_TOKENS` (default 80k). Updated `agent.py` with a custom `_messages_reducer` + `ReplaceMessages` sentinel to support both append and full-replace operations on the message list, a `_tool_executor_with_truncation()` wrapper, and a new `context_window` graph node. Graph now flows: `reasoner → tool_executor → context_window → reasoner`. Added `tiktoken` to `pyproject.toml`. All 3 A2A conformance tests pass.
- **Blockers:** None.
- **Handoff Notes:** Observation masking is fully wired but only activates for conversations exceeding `MAX_CONTEXT_TOKENS` (80k tokens). For short tasks this is a transparent no-op. Config via `.env`: `MAX_CONTEXT_TOKENS`, `MAX_TOOL_OUTPUT_CHARS`, `CONTEXT_KEEP_RECENT`. Next priorities: (1) Reflective feedback loop before final answer submission, (2) Multi-turn conversation support, (3) End-to-end stress test with 10+ tool calls to verify windowing triggers.

### Chat 8: Reflective Feedback Loop

- **Role:** Coder
- **Actions Taken:** Implemented the reflective feedback loop in `agent.py`. Added `reflection_count` to `AgentState`, a `REFLECTION_PROMPT` constant, a `reflector` graph node that critiques the draft answer via a separate LLM call (no tools), and a `should_revise` conditional edge that routes to `reasoner` (REVISE) or `END` (PASS). Updated `should_use_tools` to route to `reflector` instead of `END`. Updated `run_agent` to extract the final answer excluding reflection messages. Capped at `MAX_REFLECTIONS=2` to prevent infinite loops. All 3 A2A conformance tests pass.
- **Blockers:** None.
- **Handoff Notes:** The full Plan-Act-Learn loop is now complete. Graph: `reasoner → tool_executor → context_window → reasoner → reflector → (PASS→END / REVISE→reasoner)`. Config: `MAX_REFLECTIONS=2`. Next priorities: (1) Multi-turn conversation support, (2) Stress testing with complex multi-step tasks, (3) Competition-specific benchmark evaluation.

### Chat 9: Multi-Turn Conversation Support

- **Role:** Coder
- **Actions Taken:** Implemented multi-turn conversation support. Created `src/conversation_store.py` — an in-memory dict keyed by A2A `context_id` with TTL-based auto-cleanup (default 1 hour). Updated `run_agent()` in `agent.py` to accept an optional `history` parameter and return a 3-tuple `(answer, steps, updated_history)`. Rewrote `executor.py` to retrieve prior conversation history from the store before each invocation and save the updated history back after completion. Single-turn requests (no `context_id`) are unaffected. All 3 A2A conformance tests pass.
- **Blockers:** None.
- **Handoff Notes:** The agent now maintains conversational context across follow-up messages sharing the same `context_id`. Config: `CONVERSATION_TTL_SECONDS=3600`. The conversation store is in-memory (resets on restart). Next priorities: (1) Stress testing with complex multi-step tasks, (2) Competition-specific benchmark evaluation, (3) Persistent conversation store (e.g., SQLite) if needed for production.

### Chat 10: Reviewer Audit of Windowing, Reflection, and Multi-Turn

- **Role:** Reviewer
- **Actions Taken:** Reviewed the newly added observation masking, reflective feedback, and multi-turn conversation features against the current runtime code. Confirmed all three are implemented and wired into the execution path: `context_window` and tool-output truncation in `agent.py`/`context_manager.py`, reflection routing in `agent.py`, and `ConversationStore` persistence in `executor.py`. Confirmed the project is aligned with the intended roadmap at a feature level.
- **Blockers:** Identified correctness risks that need validation under stress: (1) `updated_history` in `run_agent()` is rebuilt by appending node outputs and does not preserve `ReplaceMessages` semantics after a compression event, so persisted history may not reflect the actually-pruned LangGraph state; (2) `context_window` only runs after `tool_executor`, so large existing multi-turn history still reaches the first `reasoner` call before pruning; (3) `[Reflection]` messages are currently persisted into conversation history and may pollute later turns; (4) there are still no dedicated automated tests for windowing, reflection, or multi-turn persistence beyond baseline A2A/MCP checks.
- **Handoff Notes:** Proceed with the planned end-to-end stress test using low `MAX_CONTEXT_TOKENS` so compression is forced and verify server logs show `Context window exceeded` and `Compression complete`. After stress-test results are collected, likely next fix is to persist the post-windowed state rather than the pre-compression append trail, then add focused tests for history pruning, reflection hygiene, and multi-turn state reuse.

### Chat 11: Correctness Fixes (Audit Response)

- **Role:** Coder
- **Actions Taken:** Fixed all four issues from Chat 10 reviewer audit. (1) Switched `run_agent` from `astream` to `ainvoke` — `updated_history` is now extracted from the graph's final reduced state, correctly reflecting any `ReplaceMessages` compressions. (2) Added front-gate pruning: `summarize_and_window()` is applied to input messages _before_ graph entry, so the first reasoner call never exceeds the token budget. (3) `[Reflection]` messages are stripped from `updated_history` before persistence, preventing pollution of future turns. (4) Created `tests/test_features.py` with 14 unit tests covering truncation (3), token counting (3), windowing (4), ConversationStore with TTL (5), and reflection hygiene (2). All 14 unit tests pass. All 3 A2A conformance tests pass.
- **Blockers:** None.
- **Handoff Notes:** All four correctness risks are resolved. The agent is now ready for an end-to-end stress test with low `MAX_CONTEXT_TOKENS` (e.g., 500) to force compression. Verify server logs show `Context window exceeded` and `Compression complete`.

### Chat 12: Tool-Call-Safe Windowing Fix

- **Role:** Coder
- **Actions Taken:** Patched `src/context_manager.py` so message windowing no longer splits an assistant `tool_calls` message from its required following `ToolMessage` entries. Added `_adjust_boundary_for_tool_bundle()` to move the compression boundary left whenever the recent window would start inside a tool-call bundle, preserving protocol-valid adjacency for OpenAI chat completions. Added two unit tests in `tests/test_features.py`: one verifies a tool-call bundle remains intact after compression, and one verifies compression is skipped when the only possible boundary would split a leading tool bundle.
- **Blockers:** None currently. If stress tests still fail after this change, the next likely issue would be a more complex multi-tool bundle pattern (for example, multiple tool calls in one AI message with several `ToolMessage` responses), but this patch covers the concrete orphaned-tool-message failure seen in the latest logs.
- **Handoff Notes:** Re-run the same 12-turn stress test with low `MAX_CONTEXT_TOKENS`. The prior 400 error (`tool_calls` not followed by tool messages) should be resolved. Confirm the server now logs `Adjusted context window boundary...` when compression would otherwise cut through a tool-call bundle.

### Chat 13: Reflection Context Safety & System Prompt Preservation

- **Role:** Coder
- **Actions Taken:** Fixed a remaining stress-test failure in `src/agent.py`. Root cause: the `reflector` LLM call was rebuilding its own message list from `HumanMessage` and `AIMessage` only, which accidentally included assistant messages with `tool_calls` but excluded the required following `ToolMessage` entries, causing OpenAI to reject the request with a 400. Added `_build_reflection_context()` to construct a protocol-safe reflection context that excludes tool-call assistant messages, `ToolMessage`s, and prior `[Reflection]` entries. Also added `_with_system_prompt()` so a summary `SystemMessage` from windowing no longer suppresses the real base `SYSTEM_PROMPT`. Extended `tests/test_features.py` with regression tests for reflection-context filtering and system-prompt preservation.
- **Blockers:** None currently. The major known protocol-shape errors from stress testing are now covered: tool-call-safe windowing and tool-call-safe reflection input.
- **Handoff Notes:** Re-run the same 12-turn stress test. The specific 400 error seen after the second OpenAI call should now be resolved. If any issue remains, capture Terminal 2 logs again so the failing stage (reasoner vs reflector) can be identified precisely.

### Chat 14: Group-Based Windowing (Replaces Chat 12 Boundary Fix)

- **Role:** Coder
- **Actions Taken:** Replaced the single-point `_adjust_boundary_for_tool_bundle()` approach from Chat 12 with a fundamentally safer design: `_group_messages()` in `context_manager.py` pre-groups messages into atomic blocks (AIMessage(tool_calls) + all following ToolMessages = one block). Windowing then operates on these groups, never splitting them. This eliminates ALL possible boundary-split scenarios that the point-adjustment missed. Added 4 dedicated tool-call adjacency tests: `test_group_messages_basic`, `test_group_messages_multiple_tools`, `test_windowing_preserves_adjacency` (10-turn conversation), `test_no_orphaned_tool_messages_in_recent`. All unit tests pass. All 3 A2A conformance tests pass.
- **Blockers:** None.
- **Handoff Notes:** Re-run the 12-turn stress test with low `MAX_CONTEXT_TOKENS`. The tool-call adjacency invariant is now structurally enforced (groups, not boundary adjustments). The `_adjust_boundary_for_tool_bundle` function has been removed.
