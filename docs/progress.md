# Project Progress & Handoff

## Purpose

This document is the "brain space" for agents to communicate and leave context. Instead of reading noisy terminal output, new agents will read this file to understand the current state of the project, who did what, what needs to be done next, and what are the major milestones/blockers.

Rules:

- Keep the log useful, not exhaustive.
- Capture major decisions, real blockers, and current direction.
- Do not dump routine test output unless testing exposed a critical bug and the fix matters for future work.
- Keep only the last 25 meaningful chats here. Older history belongs in long-term memory.

---

## Long-Term Memory

- The repo started as an A2A/LangGraph/MCP generalist agent and grew through a prompt-heavy v2 runtime.
- v2’s main faults were:
  - overloaded coordinator/reasoner/verifier roles
  - weak finance/document tool contracts
  - raw-text evidence passing
  - too much prompt control and too little typed runtime control
- The active runtime is now v3:
  - staged graph
  - typed contracts
  - template-driven execution
  - structured evidence and provenance
  - store-only offline curation
- Finance is now the main product direction. Legal/document paths still exist, but current architecture decisions should prioritize finance-first reliability.
- Known live limitation that still remains:
  - `task_profiler` and `reviewer` sometimes fail strict JSON/schema output on the current Qwen/Nebius path and fall back to deterministic parsing.

---

## Recent Chats

### Chat 1: V3 Runtime Reset

- **Role:** Architect
- **Actions Taken:** Replaced the old coordinator/reasoner/verifier-centric runtime with the staged v3 graph: `intake -> task_profiler -> template_selector -> context_builder -> solver -> tool_runner -> reviewer -> output_adapter -> reflect`.
- **Blockers:** None.
- **Handoff Notes:** v3 is the active architecture. Do not extend v2-era control patterns.

### Chat 2: Phase 1 - Profile Decision Hardening

- **Role:** Coder
- **Actions Taken:** Added `primary_profile + capability_flags + ambiguity_flags` so the runtime no longer collapses on one coarse label. Mixed legal/finance prompts now carry explicit ambiguity.
- **Blockers:** `finance_quant` remained the weakest live path for terse prompts.
- **Handoff Notes:** Profile selection is now safer, but not enough by itself; template choice is the next control layer.

### Chat 3: Phase 2 - Template Selector

- **Role:** Coder
- **Actions Taken:** Added static execution templates and wired them into real graph behavior. Templates now control allowed stages, allowed tools, and review cadence.
- **Blockers:** None.
- **Handoff Notes:** Runtime behavior should now be reasoned about in terms of templates, not free-form “layers.”

### Chat 4: Phase 3 - EvidencePack v2

- **Role:** Coder
- **Actions Taken:** Split evidence into prompt/retrieved/derived facts, added assumption ledger and provenance map, and pushed those through solver/reviewer/reflect.
- **Blockers:** None.
- **Handoff Notes:** Future fixes should preserve typed evidence and explicit assumptions; do not go back to raw message-history reasoning.

### Chat 5: Phase 4 - Document Evidence Service

- **Role:** Coder
- **Actions Taken:** Replaced raw file blobs with structured document evidence: metadata, chunks, tables, numeric summaries, citations.
- **Blockers:** None.
- **Handoff Notes:** Document tasks now require extracted evidence, not URL discovery alone.

### Chat 6: Phase 5 - Selective Checkpoints

- **Role:** Coder
- **Actions Taken:** Replaced universal rollback with template-scoped artifact checkpoints for quant/tool-compute, options, and document gather paths.
- **Blockers:** None.
- **Handoff Notes:** Backtracking is now local and artifact-based. Do not reintroduce universal reviewer-controlled rollback.

### Chat 7: Phase 6 - Offline Curation and Cleanup

- **Role:** Coder
- **Actions Taken:** Added store-only offline curation signals, cleaned stale v2 artifacts, and aligned public docs with the active v3 runtime.
- **Blockers:** None.
- **Handoff Notes:** Memory remains passive at runtime. Curation should inform offline pack updates, not ad hoc online injection.

### Chat 8: Finance Hands Phase A

- **Role:** Architect / Coder
- **Actions Taken:** Upgraded finance evidence and operator layers:
  - market data MCP
  - finance analytics MCP
  - structured tool quality/source metadata
  - `as_of_date` and finance evidence handling in the runtime
- **Blockers:** Live finance MCP exposure initially lagged code.
- **Handoff Notes:** Finance tooling is now broad enough to support real runtime decisions; environment wiring matters as much as code.

### Chat 9: Finance Hands Follow-Up

- **Role:** Coder
- **Actions Taken:** Exposed the new finance MCP servers in the live env and fixed retrieval-first `finance_quant` control. Live `finance_evidence` now follows `GATHER -> get_price_history -> pct_change -> COMPUTE -> SYNTHESIZE`.
- **Blockers:** Options risk/control path was still structurally noisy.
- **Handoff Notes:** Finance live-data quant is stable enough; options became the next bottleneck.

### Chat 10: Finance Hands Phase B - Risk Controller and Options Churn Reduction

- **Role:** Architect / Coder
- **Actions Taken:** Added:
  - `risk_controller`
  - structured risk MCP tools (`scenario_pnl`, `portfolio_limit_check`, `concentration_check`)
  - risk/disclosure contracts
  - repair-time tool normalization and stage restoration
  - deterministic bridges to reduce options churn:
    - derive `scenario_pnl` from more primary options tool results
    - deterministic compute-stage risk summary after scenario tool execution
- **Critical Bug Found During Testing:** Live `finance_options` kept failing or looping because weak `scenario_pnl` tool-call payloads and repair-time stage handling left the risk path unstable.
- **Fix:** Hardened [tool_runner](../src/agent/nodes/tool_runner.py) to backfill scenario arguments from prior strategy facts and return to `COMPUTE` after repair-time tool success; hardened [solver](../src/agent/nodes/solver.py) to produce deterministic risk-satisfaction steps instead of repeated model retries.
- **Current Status:** Live `finance_options` now completes end-to-end on the active graph with far fewer compute/risk loops.
- **Remaining Blockers:** One live options revise cycle can still happen at final synthesis when the model truncates or misses a required disclosure. Reviewer/task-profiler JSON fallback warnings also still occur on the current backend.
- **Handoff Notes:** Next finance work should target:
  1. tighter final options synthesis so truncation/disclosure misses happen less often
  2. richer finance-hands coverage for uncertainty, scenario planning, and compliance without falling back into prompt-heavy control
