# Project Progress & Handoff

## Purpose

This file is the short handoff log for active work.

Rules:

- Keep the log useful, not exhaustive.
- Capture major decisions, real blockers, and current direction.
- Do not dump routine test output unless testing exposed a critical bug and the fix matters for future work.
- Keep only the last 25 meaningful chats here. Older history belongs in long-term memory.

---

## Long-Term Memory

- The repo started as an A2A/LangGraph/MCP generalist agent and grew through a prompt-heavy v2 runtime.
- V2's main faults were:
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
- Finance is now the main product direction. Legal and document paths still exist, but current architecture decisions should prioritize finance-first reliability.
- Structured-output fallback still exists as a safeguard, but it is no longer the normal path on the current live backend.

---

## Recent Chats

### Chat 1: Phase 1 - Profile Decision Hardening

- **Role:** Coder
- **Actions Taken:** Added `primary_profile + capability_flags + ambiguity_flags` so runtime control no longer depends on a single coarse label.
- **Blockers:** Terse `finance_quant` prompts were still weak live.
- **Handoff Notes:** Profile choice is safer now, but template choice is the next real control layer.

### Chat 2: Phase 2 - Template Selector

- **Role:** Coder
- **Actions Taken:** Added static execution templates that now control stages, tool policy, and review cadence.
- **Blockers:** None.
- **Handoff Notes:** Runtime behavior should be reasoned about in terms of templates, not legacy layers.

### Chat 3: Phase 3 - EvidencePack v2

- **Role:** Coder
- **Actions Taken:** Split evidence into prompt, retrieved, and derived facts; added assumption ledger and provenance map.
- **Blockers:** None.
- **Handoff Notes:** Preserve typed evidence and explicit assumptions. Do not drift back to raw message-history control.

### Chat 4: Phase 4 - Document Evidence Service

- **Role:** Coder
- **Actions Taken:** Replaced raw file blobs with structured document evidence: metadata, chunks, tables, numeric summaries, and citations.
- **Blockers:** None.
- **Handoff Notes:** Document tasks should rely on extracted evidence, not raw file dumps or URL discovery alone.

### Chat 5: Phase 5 - Selective Checkpoints

- **Role:** Coder
- **Actions Taken:** Replaced universal rollback with template-scoped artifact checkpoints for quant/tool-compute, options, and document gather flows.
- **Blockers:** None.
- **Handoff Notes:** Backtracking is now local and artifact-based.

### Chat 6: Phase 6 - Offline Curation and Cleanup

- **Role:** Coder
- **Actions Taken:** Added store-only offline curation, removed stale v2 artifacts, and aligned docs with the active v3 runtime.
- **Blockers:** None.
- **Handoff Notes:** Memory remains passive at runtime.

### Chat 7: Finance Hands Phase A

- **Role:** Architect / Coder
- **Actions Taken:** Added real finance evidence and exact operators through market-data and finance-analytics MCP surfaces, plus structured source and quality metadata.
- **Blockers:** Live MCP exposure initially lagged the code.
- **Handoff Notes:** Finance tool wiring matters as much as finance code.

### Chat 8: Finance Live Wiring

- **Role:** Coder
- **Actions Taken:** Exposed the new finance MCP servers in the live environment and stabilized the retrieval-first `finance_quant` path.
- **Blockers:** Options became the main unstable finance path.
- **Handoff Notes:** `finance_evidence` is now stable enough to use as the reference live-data flow.

### Chat 9: Finance Hands Phase B - Risk Controller

- **Role:** Architect / Coder
- **Actions Taken:** Added `risk_controller`, structured risk tools, repair-time tool normalization, and deterministic compute/final bridges for options.
- **Critical Bug Solved:** Live `finance_options` was looping because weak `scenario_pnl` payloads and repair-stage routing kept the risk path unstable.
- **Fix:** Hardened [tool_runner](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\agent\nodes\tool_runner.py), [solver](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\agent\nodes\solver.py), and [reviewer](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\agent\nodes\reviewer.py) so the options path now reaches a deterministic compute milestone and a deterministic final after risk pass.
- **Handoff Notes:** Standard options flow is now stable on the live graph.

### Chat 10: Finance Hands Phase C - Compliance Guard

- **Role:** Architect / Coder
- **Actions Taken:** Added `compliance_guard`, extracted finance policy context from prompts, and added deterministic policy-constrained options compute/final paths for defined-risk and no-naked mandates.
- **Critical Bug Solved:** The retirement-account options prompt kept hitting recursion because the final deterministic policy answer missed the reviewer's recommendation keyword and required risk disclosures, so reviewer and compliance kept cycling on an otherwise compliant branch.
- **Fix:** Hardened [solver](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\agent\nodes\solver.py) to force a defined-risk primary strategy under mandate constraints and to synthesize a policy-compliant deterministic final carrying recommendation class, mandate, risk cap, and risk-controller disclosures.
- **Current Status:** Live `finance_options_policy` now completes cleanly on the active graph.
- **Remaining Blockers:** `task_profiler` and `reviewer` still sometimes fall back to deterministic parsing on the current backend.

### Chat 11: Structured Output Stabilization

- **Role:** Coder
- **Actions Taken:** Fixed strict JSON reliability for `task_profiler` and `reviewer` on the Nebius/Qwen backend by moving both onto `invoke_structured_output()` with backend-aware JSON-object mode and thinking disabled. Then narrowed reviewer LLM usage so gather/compute milestones and deterministic options/document passes do not bounce through the LLM reviewer unnecessarily.
- **Critical Bug Solved:** Once JSON transport started working consistently, reviewer began overreaching on gather/compute-heavy paths and caused live recursion on `finance_options`, `finance_options_policy`, `document_qa`, and `finance_evidence`.
- **Fix:** Kept the transport fix, but made reviewer escalation selective: deterministic review first, LLM review only for ambiguous or final judgment cases where it adds value.
- **Handoff Notes:** Live staged smoke is stable again and no longer emits the old reviewer/task-profiler JSON parse warnings on the current backend.

### Chat 12: Finance Hands Phase D - Template Expansion

- **Role:** Architect / Coder
- **Actions Taken:** Added `equity_research_report`, `portfolio_risk_review`, and `event_driven_finance` templates; expanded finance analytics and risk operators; extended reviewer completeness checks and live smoke coverage for the new finance templates.
- **Critical Bugs Solved:** `quant_inline_exact` still recursed on terse live finance prompts, and `get_corporate_actions(as_of_date=...)` failed on tz-aware indexes during event-driven runs.
- **Fix:** Added a deterministic inline-formula path in [solver](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\agent\nodes\solver.py) for exact finance quant tasks, and fixed timezone-safe `as_of_date` filtering in [market_data/server.py](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\mcp_servers\market_data\server.py).
- **Handoff Notes:** Live staged smoke now passes for the expanded finance suite, including `finance_quant`, equity research, portfolio risk review, and event-driven finance.
