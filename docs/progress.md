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

### Chat 13: Options Primary Compute Cleanup

- **Role:** Coder
- **Actions Taken:** Removed the remaining wasted first-step churn on the standard `finance_options` path by adding a deterministic primary `analyze_strategy` seed for non-policy options prompts.
- **Critical Bug Solved:** Standard options used to emit an empty compute milestone first, then let `risk_controller` force the real strategy tool call on revise. That path was stable but inefficient and obscured the intended graph behavior.
- **Fix:** Narrowed the new deterministic options seed to the initial compute turn only, so ordinary options prompts now go straight to `analyze_strategy`, while non-risk revise paths keep their old behavior.
- **Handoff Notes:** Live `finance_options` now starts `COMPUTE -> analyze_strategy -> scenario_pnl -> deterministic compute -> risk pass -> deterministic final`, which is the intended clean path.

### Chat 14: Finance Template Quality and Checkpoint

- **Role:** Coder
- **Actions Taken:** Strengthened the deterministic finals for `equity_research_report`, `portfolio_risk_review`, and `event_driven_finance` so those templates now finish with fuller recommendation, action, catalyst, and watchpoint sections without adding more LLM churn. Added the detailed architecture checkpoint in [finance_hands_checkpoint.md](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\docs\finance_hands_checkpoint.md).
- **Blockers:** None.
- **Handoff Notes:** The main finance paths are now structurally complete enough to document as the current checkpoint. Further work should target depth of evidence and domain breadth, not another control-flow rewrite.

### Chat 15: Finance Hands Cleanup Before Refactor

- **Role:** Coder
- **Actions Taken:** Tightened small runtime issues without changing the architecture: broadened `compliance_guard` section parsing beyond bold-only headings, made risk/disclosure matching less brittle to wording changes, and made deterministic options seeding prefer already-available market evidence before falling back to generic defaults.
- **Blockers:** None.
- **Handoff Notes:** One review concern was not acted on because it was incorrect: actionable `portfolio_risk_review` flows can already trigger `compliance_guard` through `policy_context.action_orientation` on `finance_quant` paths.

### Chat 16: Structural Refactor - Package Boundaries

- **Role:** Coder
- **Actions Taken:** Split the oversized runtime helpers into real packages: `agent.context` for profiling/evidence/stage policy, `agent.solver` for deterministic solver helpers, and `agent.tools` for tool normalization. Kept `agent.runtime_support`, `agent.tool_normalization`, and `agent.nodes.solver` as compatibility surfaces so tests, imports, and live runtime behavior stayed stable.
- **Critical Bug Solved:** The first split would have broken `build_evidence_pack` callers because the new context implementation required an explicit labeled-JSON extractor.
- **Fix:** Restored old behavior at the compatibility layer in [runtime_support.py](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\agent\runtime_support.py) so existing callers still work without signature changes.
- **Handoff Notes:** The active runtime behavior is unchanged, but the main maintenance surface is now the new package layout described in [README.md](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\README.md).

### Chat 17: V3 Recovery Hardening

- **Role:** Coder
- **Actions Taken:** Hardened the staged runtime against the simple-finance failures seen in the benchmark traces. Added benchmark/stateless mode to the runner/executor path, deduped adjacent persisted messages, preserved partial final state on recursion-limit exits, introduced task-complexity-aware evidence compaction for the solver, added task-focused evidence fields and row selection, strengthened legal reviewer dimensions, and added repeat-review loop diagnostics plus bounded terminal loop breaking.
- **Critical Bug Solved:** Exact `finance_quant` tasks were computing correctly and then looping through repeated final reviews because the runtime stayed in open-ended synthesis instead of collapsing into a terminal answer path; recursion handling also discarded the best partial answer.
- **Fix:** The quant path now runs as `simple_exact`, solver-facing evidence is reduced to relevant formula/row/output data, recursion fallback keeps the real partial final answer, and unchanged final-review loops terminate deterministically instead of burning graph steps.
- **Handoff Notes:** The live staged smoke now shows `finance_quant` finishing cleanly as exact JSON with no LLM/tool churn. The next benchmark-focused work should target deeper legal completeness and broader QA/long-document readiness rather than another control-flow rewrite.

### Chat 18: Budget, Cost, and Benchmark Mode Cleanup

- **Role:** Coder
- **Actions Taken:** Aligned budget tracking with the staged runtime by making it complexity-tier aware, enforcing tool-call budgets at runtime, and cleaning cost summaries so unpriced models no longer produce misleading USD totals. Renamed model roles to the v3-native `profiler` / `solver` / `reviewer` surface while keeping legacy env compatibility. Added a dedicated benchmark/stateless smoke and updated the public README to document `BENCHMARK_STATELESS` and the canonical model env vars.
- **Critical Bug Solved:** Trace and run summaries implied dollar precision even when the active backend model had no reliable pricing entry, and tool budgets were counted but not enforced in the actual tool path.
- **Fix:** Cost summaries now expose `cost_estimate_status` and `unpriced_models`, `total_cost_usd` is withheld unless pricing is known, and tool-budget exhaustion is enforced inside [tool_runner](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\agent\nodes\tool_runner.py).
- **Handoff Notes:** Use `BENCHMARK_STATELESS=1` for benchmark slices so each task runs as a fresh item with no conversation carryover. Prefer `PROFILER_MODEL`, `SOLVER_MODEL`, and `REVIEWER_MODEL` over the older env names in new setups.

### Chat 19: Options Duplicate-Call Cleanup

- **Role:** Coder
- **Actions Taken:** Removed low-value duplicate behavior in the deterministic options path. The first primary options tool result now becomes a compute milestone immediately, so the graph moves to `risk_controller` instead of calling `analyze_strategy` twice. Also normalized and deduped equivalent option assumption disclosures like `300` vs `300.0`.
- **Critical Bug Solved:** The deterministic smoke was still issuing `analyze_strategy` twice before risk review and could repeat the same spot-price assumption in the final answer.
- **Fix:** Added a primary-strategy compute summary in [options.py](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\agent\solver\options.py), normalized assumption dedupe in both [options.py](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\agent\solver\options.py) and [evidence.py](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\agent\context\evidence.py), and added targeted solver regressions.
- **Handoff Notes:** The intended deterministic path is now `analyze_strategy -> compute milestone -> risk revise -> scenario_pnl -> compute milestone -> synthesize`. This is the right place to evaluate any future final-only self-reflection layer because the core options path is no longer duplicating its own primary analysis.

### Chat 20: Final-Only Self-Reflection Gate

- **Role:** Coder
- **Actions Taken:** Added a bounded `self_reflection` node for benchmark-style qualitative finance/legal finals. It runs only after reviewer/risk/compliance have already passed, only on selected qualitative templates, and only in benchmark mode or explicit opt-in. It can trigger at most one targeted final revise pass, then stops.
- **Critical Bug Solved:** None. This is a controlled quality layer, not a bug fix.
- **Fix:** Added [self_reflection.py](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\agent\nodes\self_reflection.py), routed eligible final passes through it in [graph.py](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\agent\graph.py) and [reviewer.py](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\agent\nodes\reviewer.py), and added targeted tests in [test_staged_self_reflection.py](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\tests\test_staged_self_reflection.py).
- **Handoff Notes:** The design follows the useful part of self-reflection idea: heuristic-first, single bounded improvement pass, final-only. It is not a new general reflection loop.

### Chat 21: Inline Quant Row-Selection Safety Fix

- **Role:** Coder
- **Actions Taken:** Fixed the exact-quant evidence bug where fallback prompt-prefix matching could select unrelated table rows and silently feed wrong `ROE` / `ROA` values into deterministic compute. Tightened focus-query extraction, row scoring, row disambiguation, and conflicting assignment handling.
- **Critical Bug Solved:** Multi-entity or multi-period inline tables could previously contaminate `relevant_rows`, after which deterministic quant compute could use the wrong row without surfacing ambiguity.
- **Fix:** Replaced the unsafe first-600-character fallback in [evidence.py](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\agent\context\evidence.py), made table-row selection score and disambiguate against the actual focus line, kept a safe one-row fallback only for single-row tables, and made [quant.py](c:\Users\vamsi\OneDrive\Desktop\Gtihub_repos\Project-Pulse-Generalist-A2A-Reasoning-Engine\src\agent\solver\quant.py) refuse deterministic compute when conflicting row-derived assignments appear.
- **Handoff Notes:** The deterministic exact-quant path is now conservative by design: resolve one row cleanly or refuse the shortcut.
