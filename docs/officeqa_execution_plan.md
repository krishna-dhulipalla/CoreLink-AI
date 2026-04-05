# OfficeQA Execution Plan

Date: 2026-03-31
Status: Active
Source analysis:
- `docs/officeqa_integration_plan.md`
- `docs/v5_runtime_walkthrough.md`
- 2026-04-02 architecture review follow-up
- 2026-04-03 smoke-trace retrieval audit follow-up

## Purpose

This is the canonical execution plan for the OfficeQA runtime.

Use this document as the operational source of truth for coding agents. The analysis document explains why the old architecture failed. This plan tells agents what to build, in what order, how to mark progress, and what "done" means for each phase.

Current state:

- V5 baseline is complete.
- A new hardening backlog is now open because the V5 walkthrough review exposed remaining OfficeQA-only cleanup, retrieval flexibility gaps, hidden execution decisions, and future multi-document needs.

## Planning Rules

- Treat this file as a living backlog and delivery plan.
- Use markdown checkboxes for task status:
  - `[ ]` not started
  - `[x]` complete
- If a task is dropped, strike it through and add a short reason on the next line.
- Do not rewrite completed history. Append clarifications instead.
- After completing meaningful work, update `docs/progress.md` with:
  - date
  - role
  - phase / task ids completed
  - key code paths changed
  - blockers or design changes
- If scope changes, add tasks under the most relevant existing phase before creating a new phase.

## Strategic Decision

- [x] Replatform in place instead of starting a brand new repo

Reason:

- There are reusable components worth keeping:
  - A2A executor shell
  - request-scoped Judge MCP bridge
  - tracer and benchmark run artifacts
  - output adapter pattern
  - bounded self-reflection pattern
  - budget and retry controls
- The core architecture still needs major change:
  - finance-first routing
  - profile/template dependence
  - generic capability resolution
  - heuristic retrieval path
  - generic file parsing instead of Treasury-table extraction

Conclusion:

- Keep infrastructure that is benchmark-agnostic.
- Replace or isolate the finance-first reasoning core.
- Build the new OfficeQA path as the primary runtime, not as another side branch.

## OfficeQA North Star

Target runtime:

`intake -> benchmark adapter -> corpus index search -> page/table retrieval -> structured extraction -> scope validation -> deterministic compute -> validator/reviewer -> final answer adapter`

Deployment modes:

- `competition mode`: ship the OfficeQA corpus and index with the deployable runtime artifact, image layer, or mounted read-only volume
- `local dev mode`: use `OFFICEQA_CORPUS_DIR` plus the local index as a development and regression path
- `judge-assisted mode`: if the benchmark exposes document tools, use them as an optional auxiliary path, not as the only corpus access path

Requirement:

- Do not make the competition runtime depend on undocumented Judge or A2A corpus access.
- Do not require a repo-local corpus checkout in development.
- Keep the OfficeQA corpus out of git history, but make competition deployment self-sufficient by packaging the dataset separately from source control.
- Treat Judge MCP document tools as optional acceleration or validation surfaces, not as a guaranteed benchmark contract.

Non-goals:

- general finance assistant behavior
- benchmark-string routing hacks
- web-search-first OfficeQA solving
- another prompt-only retrofit of the current runtime

## Post-V5 Review Decisions

- The runtime is OfficeQA-first, but it still carries generic `TaskProfile`, prompt-guidance, and reviewer branches that are not useful for the active benchmark path.
- The remaining generic profile/task-family shell should be removed deliberately, not by scattered opportunistic edits.
- The active model stack must be explicit in docs and startup logs. Solver and reviewer strength matter for hard OfficeQA cases.
- The canonical local corpus layout should be stable and documented so teammates do not improvise paths.
- `RetrievalIntent` should become the main control plane for retrieval strategy, evidence planning, compute routing, and fallback behavior.
- Retrieval needs multiple structured strategies, not only `metadata search -> table -> row -> cell`.
- Hidden execution stages must emit explicit trace artifacts so teammates can debug table choice, aggregation choice, and evidence gaps directly.
- Deterministic compute remains the primary numeric path, but mixed evidence tasks need grounded synthesis rather than immediate insufficiency.
- Validator output should become actionable enough to drive retries automatically.
- The architecture should stay OfficeQA-native now while leaving a clean adapter seam for similar document-grounded benchmarks later.

## Research-Guided Hardening Principles

These papers do not define the runtime directly, but they do point to the next correct direction for Treasury-table hardening.

- `PubTables-1M` and `Aligning benchmark datasets for TSR` both reinforce the same lesson: canonicalization is not a cleanup detail, it is part of the extraction system. We should not project evidence directly from flattened HTML tables when the representation still contains header oversegmentation or ambiguous span structure.
- `RealHiTBench` is the closest conceptual match to our Treasury failure mode. Hierarchical headers and implicit joins should be represented as a header tree or resolved path structure before reasoning, not inferred from repeated flat header text after the fact.
- `TABLET` is useful as a systems reference because it treats structure recovery as `split -> grid -> merge`, which is much closer to what we need for dense Treasury tables than simple DOM flattening.
- `Uncertainty-Aware Complex Scientific Table Data Extraction` is the main reference for the next safety layer: do not run deterministic compute when table-structure confidence is low. Flag low-confidence extraction and route it to fallback handling instead.

Implication for this plan:

- extraction quality is now a first-class backlog area
- canonical table normalization must be evaluated independently from retrieval and compute
- structured evidence should be built from resolved row/column/header paths, not raw flattened cells
- deterministic compute must become confidence-aware

## Reusable Components To Keep

- [x] `R1` Keep A2A request/response shell where still useful
- [x] `R2` Keep Judge MCP session bridge and benchmark tool loading
- [x] `R3` Keep tracer infrastructure and trace folder conventions
- [x] `R4` Keep final output adapter pattern for strict answer contracts
- [x] `R5` Keep bounded final self-reflection pattern
- [x] `R6` Keep budget accounting and explicit stop reasons

Status after Phase 1:

- `R1-R6` are preserved and still present in the active runtime.
- These are not blockers for Phase 2.

## Components To Replace Or Isolate

- [x] `X1` Remove OfficeQA dependence on generic task profiles
- [x] `X2` Remove OfficeQA dependence on template library routing
- [x] `X3` Remove OfficeQA-specific heuristics from generic capability resolver
- [x] `X4` Replace generic document retrieval loop with OfficeQA retrieval state machine
- [x] `X5` Replace generic PDF text-window parsing with Treasury table extraction
- [x] `X6` Replace prompt-compacted document reasoning with structured evidence objects

Final status:

- `X1-X6` are complete.
- OfficeQA is now the primary benchmark-native runtime path in this repository.

## Phase 0: Repo Direction And Docs Cleanup

Objective:

- make the repository clearly OfficeQA-only and remove misleading docs/artifacts.

Tasks:

- [x] `P0.1` Clean the `docs/` folder so only relevant docs remain visible
- [x] `P0.2` Keep `docs/progress.md` as the execution log
- [x] `P0.3` Keep `docs/officeqa_integration_plan.md` as the analysis / rationale document
- [x] `P0.4` Keep this file as the implementation and tracking document
- [x] `P0.5` Update `README.md` so it no longer presents the repo as a generic finance-first system
- [x] `P0.6` Remove references to obsolete diagrams or legacy V3/V4 planning docs
- [x] `P0.7` Define the canonical OfficeQA doc set in one short section of the README

Deliverables:

- a clean `docs/` directory
- a README that points to the OfficeQA analysis and execution plan

Exit criteria:

- a new contributor can open the repo and understand that the project focus is OfficeQA only

## Phase 1: Benchmark Boundary And Runtime Contract

Objective:

- create an explicit OfficeQA benchmark boundary so the runtime no longer depends on benchmark strings scattered across the codebase.

Tasks:

- [x] `P1.1` Create a benchmark adapter package, for example `src/agent/benchmarks/`
- [x] `P1.2` Add an OfficeQA adapter module responsible for benchmark-specific behavior
- [x] `P1.3` Define explicit OfficeQA runtime config:
  - answer contract
  - allowed tool families
  - fallback policy
  - validation requirements
  - output normalization rules
- [x] `P1.4` Move OfficeQA detection and overrides out of generic profiling logic
- [x] `P1.5` Route OfficeQA mode at intake and carry it explicitly through runtime state
- [x] `P1.6` Add tests that prove OfficeQA execution no longer depends on prompt keyword luck

Suggested code targets:

- `src/agent/nodes/intake.py`
- `src/agent/context/profiling.py`
- `src/agent/state.py`
- `src/agent/graph.py`
- new modules under `src/agent/benchmarks/`

Exit criteria:

- OfficeQA tasks are activated by benchmark/runtime configuration, not by brittle text heuristics alone

## Phase 2: Corpus Ingest And Index Build

Objective:

- make the Databricks OfficeQA corpus the primary retrieval substrate.

Tasks:

- [x] `P2.1` Create an ingestion script for the OfficeQA corpus
- [x] `P2.2` Ingest parsed JSON and transformed text artifacts
- [x] `P2.3` Build a persistent corpus manifest keyed by source document
- [x] `P2.4` Build searchable metadata fields:
  - document id
  - year
  - month
  - page markers
  - section titles
  - table headers
  - row labels
  - unit hints
- [x] `P2.5` Store normalized numeric values where possible
- [x] `P2.6` Link benchmark `source_files` to indexed artifacts
- [x] `P2.7` Add validation tooling to detect malformed or partially parsed documents

Suggested code targets:

- `scripts/build_officeqa_index.py`
- `src/agent/benchmarks/officeqa_index.py`
- `src/agent/benchmarks/officeqa_manifest.py`

Exit criteria:

- the runtime can query indexed Treasury content without open web search

Deployment note after Phase 2:

- local indexing is complete, but competition deployment still needs a reproducible corpus-delivery path outside git
- the correct pattern is a pinned dataset artifact, container image layer, or mounted read-only volume, not an assumption that Judge will expose the files

## Phase 2.5: Competition Corpus Packaging

Objective:

- make competition runs self-sufficient for OfficeQA corpus access without storing the dataset in git.

Tasks:

- [x] `P2.8` Define the competition dataset delivery model:
  - baked into the runtime image
  - attached as a read-only mounted volume
  - or downloaded from a pinned public artifact at build time
- [x] `P2.9` Add a corpus bootstrap or verification script that checks:
  - corpus root exists
  - manifest exists
  - index metadata version matches runtime expectations
  - required source documents are readable
- [x] `P2.10` Add deployment configuration so competition startup can locate the packaged corpus without repo-local paths
- [x] `P2.11` Add a fail-fast startup check with a clear error when corpus access is missing in competition mode
- [x] `P2.12` Document the exact build and deployment path for the OfficeQA corpus artifact

Suggested code targets:

- `scripts/build_officeqa_index.py`
- new deployment/bootstrap scripts
- runtime startup or executor bootstrap code
- `README.md`

Exit criteria:

- a competition deployment can answer OfficeQA tasks with no dependence on Judge-exposed files and no dependence on a developer workstation path

Status:

- competition mode now assumes a packaged or mounted OfficeQA corpus bundle
- startup verification is implemented before the A2A executor finishes booting
- `scripts/verify_officeqa_corpus.py` is the preflight check for packaged deployments

## Phase 3: OfficeQA Retrieval Tools And State Machine

Objective:

- replace heuristic finance retrieval with a benchmark-native OfficeQA retrieval loop.

Tasks:

- [x] `P3.0` Make the retrieval backend explicit:
  - packaged OfficeQA corpus is primary
  - Judge-exposed document tools are optional secondary adapters
  - open web search is not part of normal OfficeQA execution
- [x] `P3.1` Implement OfficeQA retrieval tools, for example:
  - `search_officeqa_documents`
  - `fetch_officeqa_pages`
  - `fetch_officeqa_table`
  - `lookup_officeqa_rows`
  - `lookup_officeqa_cells`
- [x] `P3.2` Build an OfficeQA retrieval state machine with explicit stages:
  - identify source
  - locate page or table
  - extract values
  - validate scope
  - compute
- [x] `P3.3` Remove default web fallback for OfficeQA normal execution
- [x] `P3.4` Add retrieval ranking based on indexed metadata, not title/snippet heuristics alone
- [x] `P3.5` Add clear failure states for:
  - wrong source family
  - missing table
  - partial table
  - missing month coverage
  - unit ambiguity
- [x] `P3.6` Add tests for wrong-source rejection and table-first retrieval

Suggested code targets:

- `src/agent/nodes/orchestrator.py`
- `src/agent/retrieval_reasoning.py`
- `src/agent/retrieval_tools.py`
- new OfficeQA retrieval modules

Exit criteria:

- retrieval actions are driven by explicit missing evidence, not generic heuristics
- the primary retrieval path works in competition with only the packaged corpus and runtime image

Status:

- OfficeQA retrieval now prefers explicit document -> table -> row -> cell -> page actions over the old generic search/fetch loop
- default OfficeQA web fallback is disabled unless `OFFICEQA_ALLOW_WEB_FALLBACK` is explicitly enabled
- failure reporting now distinguishes wrong source family, missing table, partial table, missing month coverage, and unit ambiguity

## Phase 4: Structured Extraction And Provenance

Objective:

- convert retrieved Treasury evidence into stable structured objects before reasoning or compute.

Tasks:

- [x] `P4.1` Define structured table objects for OfficeQA
- [x] `P4.2` Define structured row/cell provenance objects
- [x] `P4.3` Extract units and normalize thousand / million / billion / percent
- [x] `P4.4` Preserve document id, page number, table id, row label, column label for every extracted value
- [x] `P4.5` Make the solver consume structured evidence objects rather than lossy prompt compaction
- [x] `P4.6` Add tests for:
  - page-to-table mapping
  - row extraction
  - unit normalization
  - provenance completeness

Suggested code targets:

- `src/mcp_servers/file_handler/server.py` or replacement OfficeQA parser modules
- `src/agent/curated_context.py`
- `src/agent/tools/normalization.py`
- new OfficeQA extraction/provenance modules

Exit criteria:

- every numeric answer can be traced back to exact document structure

Status:

- OfficeQA retrieval results now project into typed structured evidence objects before synthesis and review
- solver and review payloads now carry compact structured evidence instead of depending only on lossy tool-finding summaries
- executor refreshes structured evidence after each retrieval tool result so later synthesis sees the current table/cell provenance

## Phase 5: Deterministic Compute Layer

Objective:

- move OfficeQA arithmetic out of prompt synthesis and into deterministic operators over extracted values.

Tasks:

- [x] `P5.1` Create an OfficeQA compute module
- [x] `P5.2` Add deterministic operators for:
  - monthly sum
  - annual total
  - absolute difference
  - percent change
  - inflation-adjusted difference
  - fiscal-year vs calendar-year normalization
- [x] `P5.3` Add an operation ledger for every computed result
- [x] `P5.4` Add compute validation before final answer emission
- [x] `P5.5` Add tests for common OfficeQA calculation patterns

Suggested code targets:

- `src/agent/benchmarks/officeqa_compute.py`
- `src/agent/contracts.py`
- reviewer / validator integration points

Exit criteria:

- the final answer is produced from structured extracted values with a reproducible compute chain

Status:

- OfficeQA now has a deterministic compute layer over structured values with explicit monthly, calendar-year, fiscal-year, absolute-difference, percent-change, and inflation-adjusted operators
- executor prefers deterministic OfficeQA compute before LLM synthesis when structured evidence is sufficient
- compute results now carry a compact operation ledger and validation status into curated context, review packets, and solver payloads

## Phase 6: Validator / Reviewer / Final Adapter

Objective:

- keep the useful second-pass validation pattern, but attach it to structured evidence instead of vague retrieval summaries.

Tasks:

- [x] `P6.1` Add an OfficeQA validator before final answer formatting
- [x] `P6.2` Make validator enforce:
    - source family correctness
    - entity/category correctness
    - time scope correctness
    - aggregation correctness
    - unit consistency
    - provenance presence
- [x] `P6.3` Keep final output adapter for exact answer contract formatting
- [x] `P6.4` Keep bounded self-reflection only as a final completeness guard
- [x] `P6.5` Ensure reflection cannot override structured compute or provenance failures
- [x] `P6.6` Add insufficiency-safe output behavior

Exit criteria:

- validator catches unsupported calculations before final formatting

Status:

- `src/agent/benchmarks/officeqa_validator.py` now validates structured evidence, scope alignment, unit consistency, provenance presence, and deterministic-compute readiness before final formatting
- reviewer integration now records compact validator output in the review packet and converts hard OfficeQA validation failures into bounded stop reasons instead of another free-form revise loop
- route logic now prevents self-reflection from overriding OfficeQA structured compute or provenance failures
- insufficiency-safe final behavior is now emitted for unsupported or under-grounded OfficeQA answers and still flows through the output adapter for benchmark contract formatting

## Phase 7: Runtime Simplification And Old-Path Retirement

Objective:

- remove or quarantine the old finance-first runtime paths that will otherwise keep corrupting OfficeQA behavior.

Tasks:

- [x] `P7.1` Remove OfficeQA special cases from generic profiles
- [x] `P7.2` Remove OfficeQA special cases from generic templates
- [x] `P7.3` Remove OfficeQA special cases from generic capability widening
- [x] `P7.4` Remove obsolete OfficeQA prompt detectors once adapter path is stable
- [x] `P7.5` Decide whether to keep a minimal generic runtime shell or collapse directly to OfficeQA-only routing
- [x] `P7.6` Delete dead code and tests related only to retired finance-first paths

Exit criteria:

- OfficeQA path is the primary runtime, not a compatibility layer inside old finance logic

Phase 7 completion notes:

- OfficeQA activation is now explicit-benchmark-only. Prompt-shape heuristics and compatibility env toggles no longer enable the benchmark adapter or XML contract.
- Generic capability widening now reads benchmark runtime policy through the benchmark adapter boundary instead of carrying OfficeQA-only flags in generic resolver code.
- The old static template-library routing path was removed. The runtime keeps a minimal generic shell, but OfficeQA now runs through benchmark intent plus execution-mode stubs rather than legacy template ids.
- Decision for `P7.5`: keep the minimal generic runtime shell for shared infrastructure and tests, but treat OfficeQA as the only benchmark-native path that should continue evolving in this repo.

## Phase 8: Evaluation Harness And Delivery Discipline

Objective:

- make progress measurable by failure type, not by vague "benchmark still bad" impressions.

Tasks:

- [x] `P8.1` Create a small curated OfficeQA regression slice grouped by failure mode
- [x] `P8.2` Create run reports that classify failures as:
  - routing
  - retrieval
  - extraction
  - compute
  - validation
  - formatting
- [x] `P8.3` Add a benchmark smoke path for rapid iteration
- [x] `P8.4` Add artifact capture for:
  - chosen source files
  - extracted tables
  - compute ledger
  - final answer
- [x] `P8.5` Define go/no-go criteria before full benchmark runs

Exit criteria:

- every failed task can be mapped to a concrete subsystem

Phase 8 completion notes:

- Added the curated slice at `eval/officeqa_regression_slice.json`, grouped by the subsystem each case is meant to stress.
- Added `src/agent/benchmarks/officeqa_eval.py` to classify runs, capture OfficeQA artifacts, and summarize go/no-go readiness.
- Added `scripts/run_officeqa_regression.py` as the new smoke/full regression entrypoint for OfficeQA iteration.
- Go/no-go rule: block full benchmark runs if there are any routing or formatting failures, or if fewer than 60% of the selected cases produce table-backed final answers.

## Phase 9: OfficeQA-Only Cleanup And Operational Clarity

Objective:

- finish removing non-OfficeQA runtime debt and make models plus corpus setup explicit for local and benchmark execution.

Tasks:

- [x] `P9.1` Audit remaining non-OfficeQA `TaskProfile`, `ExecutionMode`, and `ProfileContextPack` branches that still participate in the active runtime
- [x] `P9.2` Remove or isolate finance/options/legal prompt guidance from the active OfficeQA path:
  - `src/agent/profile_packs.py`
  - `src/agent/prompts.py`
  - `src/agent/review_utils.py`
  - `src/agent/context/evidence.py`
  - `src/agent/nodes/orchestrator_intent.py`
- [x] `P9.3` Reduce the active runtime taxonomy to an OfficeQA-minimal set, or move legacy task families behind an explicit archive-compatibility boundary
- [x] `P9.4` Make the active role-model mapping visible at startup and in teammate docs
- [x] `P9.5` Choose and document the strong default OfficeQA role stack:
  - strong document-grounded solver
  - strong reviewer for ambiguity and scope checks
  - lightweight adapter and reflection models
- [x] `P9.6` Define the canonical local corpus layout under `data/officeqa/` and keep it reproducible across teammates
- [x] `P9.7` Add regression coverage proving OfficeQA execution no longer depends on retired finance/legal prompt branches

Exit criteria:

- teammates can see the active models and corpus path without reading source
- OfficeQA runtime no longer depends on finance/legal profile guidance in normal execution

Phase 9 completion notes:

- OfficeQA benchmark intent is now resolved before generic task-family inference, so benchmark runs no longer depend on legacy finance/legal/options routing to reach the active path.
- The centralized prompt layer is now OfficeQA financial-document guidance rather than a general finance router. It explicitly supports extraction, inflation-adjusted comparisons, statistical analysis, forecasting, weighted averages, and risk-style metrics as document-grounded question classes.
- The active reviewer path no longer imports or uses legal/options review gaps. Those old heuristics are removed from the live OfficeQA flow.
- `src/agent/profile_packs.py` is reduced to a minimal compatibility surface instead of a broad finance/legal catalog.
- `context_curator` now carries OfficeQA analysis-mode facts into curated context so the solver sees the benchmark's financial reasoning surfaces without relying on retired profiles.

## Phase 10: Adaptive Retrieval Control Plane

Objective:

- turn `RetrievalIntent` from a query container into the main control plane for retrieval strategy and evidence acquisition.

Tasks:

- [x] `P10.1` Extend `RetrievalIntent` with:
  - retrieval strategy
  - strategy confidence
  - evidence requirements
  - fallback chain
  - join requirements
- [x] `P10.2` Add explicit retrieval strategies:
  - `table_first`
  - `text_first`
  - `hybrid`
  - `multi_table`
  - `multi_document`
- [x] `P10.3` Add a typed `EvidencePlan` object before retrieval that states:
  - required values
  - metric identity
  - expected units
  - time scope
  - minimum source requirements
- [x] `P10.4` Use early retrieval signals to switch strategies instead of looping inside one fixed table flow
- [x] `P10.5` Add a text-first extraction path for text-only and implicit-metric questions
- [x] `P10.6` Add a hybrid table-plus-text retrieval path for implicit metrics and narrative support
- [x] `P10.7` Add multi-table joins within a single source document
- [x] `P10.8` Add predictive evidence sufficiency checks against the evidence plan before compute
- [x] `P10.9` Extend regression slices so failures are grouped by retrieval strategy, not only by subsystem

Exit criteria:

- retrieval is flexible but still structured
- OfficeQA no longer depends on one brittle `table -> row -> cell` pipeline

## Phase 11: Traceability And Diagnostic Artifacts

Objective:

- make hidden retrieval, extraction, compute, and validation decisions visible in traces and reports.

Tasks:

- [x] `P11.1` Add explicit trace artifacts for:
  - `retrieval_decision`
  - `strategy_reason`
  - `candidate_sources`
  - `table_selection_reason`
  - `text_selection_reason`
  - `aggregation_reason`
  - `evidence_gaps`
- [x] `P11.2` Capture rejected retrieval candidates and why they were rejected
- [x] `P11.3` Persist compute-path selection reasoning and rejected aggregation alternatives
- [x] `P11.4` Persist validator remediation guidance in both run traces and regression reports
- [x] `P11.5` Update teammate docs and flow diagrams so embedded stages are visible without source diving

Exit criteria:

- a teammate can answer "why did it pick this table?" and "why did it choose this aggregation?" from artifacts alone

## Phase 12: Hybrid Compute And Grounded Synthesis

Objective:

- keep deterministic compute for strict numeric tasks while allowing grounded synthesis for mixed or ambiguous OfficeQA questions.

Tasks:

- [x] `P12.1` Split questions into:
  - deterministic-compute eligible
  - grounded-synthesis required
  - hybrid numeric-plus-narrative
- [x] `P12.2` Add hybrid answer mode:
  - deterministic numeric core
  - grounded narrative wrapper
- [x] `P12.3` Add grounded synthesis fallback when compute is impossible but evidence is still sufficient for a partial or qualitative answer
- [x] `P12.4` Ensure reviewer can approve bounded partial answers when fully deterministic output is impossible
- [x] `P12.5` Add stronger model routing for synthesis-heavy, ambiguity-heavy, and long-context OfficeQA tasks
- [x] `P12.6` Add regression cases for mixed tasks, partial answers, and synthesis-with-provenance paths

Exit criteria:

- hard numeric tasks stay deterministic
- mixed tasks do not collapse into unnecessary insufficiency

Phase 12 completion notes:

- `RetrievalIntent` now carries `analysis_modes`, `answer_mode`, `compute_policy`, and `partial_answer_allowed`, so answer strategy is explicit runtime state instead of hidden prompt behavior.
- Point-lookups and synthesis-heavy finance questions no longer force deterministic compute by default. Strict calendar/fiscal/paired-comparison tasks still do.
- Executor now supports three OfficeQA answer paths:
  - deterministic finalization for strict numeric tasks
  - hybrid synthesis that wraps a deterministic numeric core with grounded narrative
  - grounded synthesis fallback when compute is preferred but not available
- Validator now distinguishes compute-required versus synthesis-compatible tasks, and reviewer can approve bounded partial answers when the supported portion is explicit and source-backed.
- Model routing now supports synthesis-heavy and ambiguity-heavy OfficeQA tasks through stronger solver/reviewer override hooks.
- Regression slices and eval summaries now capture answer mode in addition to subsystem and retrieval strategy.

## Phase 13: Actionable Validator And Adaptive Orchestration

Objective:

- turn validation from a strict gate into a diagnostic control surface that can steer retries and orchestration choices.

Tasks:

- [x] `P13.1` Make validator return machine-actionable remediation codes plus human-readable repair guidance
- [x] `P13.2` Add orchestration strategies for:
  - table compute
  - text reasoning
  - hybrid join
  - cross-document comparison
- [x] `P13.3` Bind orchestration strategy selection to `TaskIntent`, `RetrievalIntent`, and `EvidencePlan`
- [x] `P13.4` Add retry policies keyed to validator remediation instead of generic revise loops
- [x] `P13.5` Add stop rules that prevent useless retries when evidence requirements cannot be met
- [x] `P13.6` Capture orchestration choice and retry path in the trace and regression report

Exit criteria:

- the runtime can adapt its flow instead of repeating one best-effort pipeline

Phase 13 completion notes:

- `src/agent/benchmarks/officeqa_validator.py` now returns machine-actionable remediation codes, repair guidance, recommended repair targets, and orchestration strategy hints instead of only hard-failure labels.
- `src/agent/nodes/orchestrator.py` now converts OfficeQA validator revisions into targeted gather or compute retries, applies orchestration-specific retrieval strategy overrides, and stops early with explicit `officeqa_*` stop reasons when no useful repair path remains.
- `src/agent/curated_context.py`, `src/agent/benchmarks/officeqa_eval.py`, and runtime traces now preserve validator codes, orchestration strategy, retry allowance, and retry stop reason so debugging and regression reports expose the same control decisions the runtime used.

## Phase 14: Multi-Document Support And Future Document Adapters

Objective:

- future-proof the runtime for similar document-grounded benchmarks without restoring the old finance-first architecture.

Tasks:

- [x] `P14.1` Add cross-document table merge support with explicit provenance retention
- [x] `P14.2` Add unit and time alignment across multiple documents
- [x] `P14.3` Separate retrieval, compute, and validator interfaces into document-benchmark adapter seams
- [x] `P14.4` Prove OfficeQA remains first-class while a second document benchmark can plug in without prompt-hack routing
- [x] `P14.5` Document what stays OfficeQA-specific versus what becomes generic document-runtime infrastructure

Exit criteria:

- similar document-grounded tasks can reuse the architecture without reintroducing finance-specific profiles or template routing

Phase 14 completion notes:

- `src/agent/officeqa_structured_evidence.py` now emits explicit cross-document merged series and alignment summaries with provenance retention, so multi-document support is no longer just a retrieval strategy flag.
- `src/agent/retrieval_reasoning.py` and `src/agent/benchmarks/officeqa_validator.py` now check cross-document unit and time alignment explicitly when the evidence plan requires cross-source comparison.
- `src/agent/benchmarks/__init__.py`, `src/agent/curated_context.py`, and `src/agent/nodes/orchestrator.py` now route structured-evidence build, compute, and final validation through benchmark document-adapter hooks rather than hardwiring those interfaces directly into the runtime.
- OfficeQA remains the default first-class adapter, but tests now prove that a second document benchmark can register retrieval/compute/validate hooks without changing prompt routing or reviving template-based dispatch.

## Phase 15: Canonical Treasury Table Normalization

Objective:

- replace lossy HTML flattening with a canonical table representation that preserves header semantics before evidence projection.

Why now:

- the April 3 smoke traces show retrieval can reach the right Treasury file and table, but structured evidence becomes noisy because repeated multi-row headers and spanning cells are flattened too early.
- this phase is directly guided by `PubTables-1M`, `Aligning benchmark datasets for TSR`, `TABLET`, and `RealHiTBench`.

Tasks:

- [x] `P15.1` Define a normalized internal table schema for Treasury-style tables that keeps:
  - row count / column count
  - explicit spanning-cell relationships
  - header-vs-data functional labels
  - row header tree
  - column header tree
  - unit metadata
  - page / table locator provenance
- [x] `P15.2` Add a header canonicalization pass that collapses repeated header spans and reconstructs hierarchical headers before row/cell projection
- [x] `P15.3` Add a split/merge-oriented normalization step for parsed HTML tables:
  - infer the dense grid first
  - then reconstruct merged cells
  - then assign content to resolved cells
- [x] `P15.4` Distinguish structural rows from data rows:
  - repeated header rows
  - section divider rows
  - subtotal / total rows
  - notes / footnotes
- [x] `P15.5` Build a normalization regression suite for Treasury-style fixtures with:
  - merged headers
  - repeated year bands
  - fiscal-year comparisons
  - unit shifts
  - blank spacer rows
- [x] `P15.6` Add normalization quality metrics that can be checked before evidence projection:
  - duplicate-header collapse score
  - span consistency
  - header/data separation quality
  - recovered unit coverage

Suggested code targets:

- `src/agent/retrieval_tools.py`
- new normalization modules under `src/agent/tools/`
- `src/agent/tools/normalization.py`
- new Treasury fixture tests under `tests/`

Exit criteria:

- extracted Treasury tables are represented canonically enough that header trees, spans, and units survive into the next stage
- evidence projection no longer starts from raw flattened HTML rows

Phase 15 completion notes:

- Added a canonical Treasury-table normalization layer in `src/agent/tools/table_normalization.py`.
- OfficeQA retrieval tools now normalize parsed HTML and flat tables into a canonical schema before emitting table payloads.
- Canonical table payloads now carry:
  - hierarchical column paths
  - row records
  - row-header depth
  - structural row typing
  - normalization metrics
- `src/agent/officeqa_structured_evidence.py` now projects value evidence from canonical row/cell records when available, instead of consuming only raw flattened rows.
- Regression coverage now includes direct normalization tests and retrieval/evidence integration checks for canonical table payloads.

## Phase 16: Confidence-Aware Structured Evidence And Point Lookup

Objective:

- rebuild structured evidence on top of the normalized table schema and prevent deterministic compute from running on low-confidence structure.

Why now:

- Task 1 is failing after the right table is fetched because the evidence layer still turns too many noisy cells into candidate values.
- this phase is guided mainly by `RealHiTBench` and `Uncertainty-Aware Complex Scientific Table Data Extraction`.

Tasks:

- [x] `P16.1` Rework structured evidence projection so each value is keyed by resolved:
  - row path
  - column path
  - year or period
  - unit
  - table / page provenance
- [x] `P16.2` Add confidence fields for:
  - header reconstruction
  - row classification
  - cell assignment
  - unit assignment
  - period resolution
- [x] `P16.3` Add a confidence gate before deterministic compute:
  - allow compute on high-confidence evidence
  - stop or fall back on low-confidence structure
- [x] `P16.4` Rewrite `point_lookup` and related selection logic to query against resolved header paths rather than raw row/cell text overlap
- [x] `P16.5` Add explicit low-confidence stop reasons and diagnostics to traces and regression reports
- [x] `P16.6` Add fallback policy for low-confidence tables:
  - slower alternate extractor
  - alternate normalization strategy
  - or bounded insufficiency when neither path is trustworthy

Suggested code targets:

- `src/agent/officeqa_structured_evidence.py`
- `src/agent/benchmarks/officeqa_compute.py`
- `src/agent/retrieval_reasoning.py`
- `src/agent/benchmarks/officeqa_eval.py`

Exit criteria:

- deterministic point lookup uses resolved structural paths instead of raw flattened labels
- compute can explicitly refuse low-confidence extraction instead of silently producing noisy candidate values

Phase 16 completion notes:

- Structured evidence now carries resolved `row_path`, `column_path`, and a `structure_confidence` score per value, plus a top-level `structure_confidence_summary`.
- Predictive evidence checks now emit `low-confidence structure` before deterministic compute when normalized table structure is too weak.
- Deterministic OfficeQA compute now applies a structure-confidence gate and refuses to compute from low-confidence tables.
- `point_lookup` scoring now prefers resolved path semantics and explicit leaf-year matches instead of relying mostly on flattened cell text.
- Low-confidence structure is now visible in curated provenance, traces, regression artifacts, and stop reasons.
- Fallback behavior is now explicit:
  - required deterministic tasks stop with bounded insufficiency on low-confidence structure
  - non-required compute paths can continue through the existing grounded-synthesis fallback path

## Phase 17: Retrieval Surface And State Deduplication

Objective:

- remove duplicate search surfaces and collapse overlapping runtime state so traces and orchestration have one authoritative source for each concept.

Why now:

- the April 3 trace audit showed the retrieval path is harder to debug because the same search results and state fields are repeated in several places.

Tasks:

- [x] `P17.1` Choose one authoritative OfficeQA local-index search surface for benchmark mode
- [x] `P17.2` Keep generic `search_reference_corpus` as a fallback only when OfficeQA-native search is unavailable, not as a parallel duplicate path
- [x] `P17.3` Define the authoritative owner for each repeated concept:
  - source-file expectations
  - source-file matches
  - document family
  - query candidates
  - retrieval strategy
  - evidence plan
- [x] `P17.4` Remove duplicate propagation of those fields into generic facts when typed state already exists
- [x] `P17.5` Simplify trace payloads so `execution_summary` shows the authoritative fields once and raw details stay in the lower-level node payload
- [x] `P17.6` Add a node-state audit test that fails when the same benchmark-specific field is redundantly carried across too many layers

Completion notes:

- OfficeQA benchmark mode now chooses `search_officeqa_documents` as the authoritative local-index search surface whenever it is available; `search_reference_corpus` remains as a fallback, not a parallel duplicate path.
- Curated provenance is now the authoritative owner for source-file expectations/matches and retrieval-plan state such as `document_family`, `query_candidates`, `strategy`, and evidence-plan summaries.
- Generic `facts_in_use` no longer repeat those benchmark-specific fields when typed provenance already carries them.
- `execution_summary` now shows a compact authoritative retrieval view with a top candidate and candidate counts, while the raw candidate lists remain only in the lower-level node payload.
- Added regression coverage for tool-plan deduplication, curated-context state ownership, and compact trace summaries.

Suggested code targets:

- `src/agent/capabilities.py`
- `src/agent/curated_context.py`
- `src/agent/retrieval_tools.py`
- `src/agent/tracer.py`
- `src/agent/contracts.py`

Exit criteria:

- benchmark mode does not perform duplicate OfficeQA-local searches
- teammates can identify one authoritative field owner for source and retrieval state

## Phase 18: Planner, Reviewer, And Finalization Alignment

Objective:

- align planner intent, compute policy, reviewer expectations, and final answer requirements with the actual document-grounded numeric runtime.

Why now:

- the latest smoke traces show one path can successfully retrieve and compute a value while downstream reviewer expectations still ask for a richer answer shape than the deterministic path emits.

Tasks:

- [x] `P18.1` Improve benchmark-mode metric/entity extraction for Treasury-style finance questions without reintroducing hardcoded benchmark-string hacks
- [x] `P18.2` Align reviewer requirements with deterministic structured-evidence outputs so successful compute does not fail only because quote-style support text was not emitted
- [x] `P18.3` Make the reason an LLM was skipped explicit in traces and regression reports
- [x] `P18.4` Separate routing-only regression cases from solvable QA regression cases so smoke results do not conflate activation checks with retrieval/compute quality
- [x] `P18.5` Add benchmark go/no-go checks that depend on:
  - extraction quality
  - evidence confidence
  - compute reliability
  - final-answer contract success
- [x] `P18.6` Extend teammate docs with a clear debug ladder for:
  - retrieval miss
  - normalization miss
  - evidence miss
  - compute miss
  - reviewer/finalization miss

Completion notes:

- Narrative Treasury questions now route into `text_first` or `hybrid` using generic narrative cues like `reason was given`, `narrative`, and `discussion`, while simple numeric extraction questions still remain deterministic.
- Deterministic point lookup now prefers numeric candidate cells over matching label cells, which closes the remaining simple point-lookup misalignment seen in broader OfficeQA runtime slices.
- Reviewer requirements are now aligned with deterministic structured answers: a validator-approved deterministic compute result no longer fails only because the public answer text omitted inline quote-style support.
- Executor and reviewer traces now record explicit `llm_decision_reason` values, and regression reports surface the solver LLM decision for each case.
- Regression reporting now distinguishes `case_kind`, keeps routing-only checks separate from solvable QA cases, and computes go/no-go readiness from QA-only thresholds for extraction quality, structure confidence, compute reliability, and final-contract success.
- The broader OfficeQA runtime slice that was still failing after Phase 16 is now green on the targeted suite used to track planner/reviewer alignment.

Suggested code targets:

- `src/agent/context/extraction.py`
- `src/agent/nodes/orchestrator.py`
- `src/agent/benchmarks/officeqa_validator.py`
- `src/agent/benchmarks/officeqa_eval.py`
- `docs/v5_runtime_walkthrough.md`

Exit criteria:

- a deterministic benchmark answer can pass end to end when evidence and compute are correct
- traces explain whether failure came from retrieval, normalization, evidence, compute, or review/finalization

## Phase 19: Experimental TSR Fallback Track

Objective:

- evaluate whether a model-assisted table-structure path is needed for the hardest Treasury tables, without immediately replacing the default lightweight pipeline.

Why now:

- the papers suggest our current bottleneck may be representational rather than purely logical, but we should validate that with an isolated experimental track instead of splicing a heavy TSR model into the main runtime too early.

Tasks:

- [x] `P19.1` Evaluate whether Treasury parsed HTML plus canonical normalization is sufficient for the current regression slice before adding a heavyweight TSR dependency
- [x] `P19.2` Prototype a slow-path extractor for hard tables using a model-assisted TSR approach inspired by `Table Transformer` / `TABLET`
- [x] `P19.3` Compare:
  - default parser
  - canonical normalized parser
  - slow TSR fallback
  on a Treasury fixture set with gold-like normalized outputs
- [x] `P19.4` Define promotion criteria for any slow-path extractor:
  - meaningful quality gain
  - bounded runtime cost
  - deployable in local and competition modes
- [x] `P19.5` Keep the TSR fallback optional until it clearly outperforms the normalized default path on hard cases

Completion notes:

- Added an experimental split/merge fallback seam in `src/agent/tools/tsr_fallback.py` that operates on parsed HTML table grids and stays disabled unless `OFFICEQA_ENABLE_TSR_FALLBACK=1`.
- Wired the fallback only into dense HTML table normalization in `src/agent/retrieval_tools.py`; when disabled, the runtime still uses the default canonical normalizer with no behavior change.
- Added a hard-table fixture set in `eval/officeqa_tsr_fixture_set.json` and a comparison harness in `scripts/evaluate_officeqa_tsr_fallback.py`.
- Current fixture evaluation result:
  - `fixture_count=2`
  - `fallback_wins=2`
  - `avg_score_delta=0.0709`
  - recommendation: `candidate_for_promotion`
- Promotion decision:
  - keep the fallback optional for now
  - only consider default-on promotion after it shows the same advantage on live OfficeQA hard-table regressions with acceptable runtime cost in both local and competition packaging modes

Exit criteria:

- the repo has an evidence-based decision on whether canonical normalization alone is enough or whether a slower TSR fallback is justified

## Post-Benchmark Hardening Track: 2026-04-05 Trace Audit

Context:

- The first original-benchmark local run exposed a different class of failures than the earlier smoke slice.
- The main failures are now system faults, not single-case bugs:
  - false semantic passes
  - wrong document or wrong table-family selection
  - weak repair behavior after predictive evidence gaps
  - deterministic-only routing that never escalates to bounded LLM-assisted reformulation or review
  - overlapping state fields that make traces harder to reason about than they need to be
- These phases must not hardcode for the three failed benchmark questions.
- The fixes should generalize to similar document-grounded financial reasoning tasks.

Reference frame for this track:

- keep using the earlier table-structure guidance already captured in this plan:
  - `PubTables-1M`
  - `Aligning benchmark datasets for TSR`
  - `RealHiTBench`
  - `TABLET`
  - `Uncertainty-Aware Complex Scientific Table Data Extraction`
- keep using the Purple-agent lesson that the runtime should stay explicitly staged:
  - retrieve
  - parse
  - analyze
  - validate
  - compute
  - complete
- do not use benchmark-string hacks or task-specific allowlists as a substitute for fixing the runtime.

## Phase 20: Semantic Question Decomposition And Query Planning

Objective:

- rebuild the prompt-to-intent layer so the system extracts the right entity, metric, scope, qualifiers, and exclusions before retrieval starts.

Why now:

- Task 1 turns the entity into `U.S national defense in the calendar year of 1940` instead of separating:
  - entity
  - metric
  - time scope
- Task 2 turns `all individual calendar months in 1953` into the entity rather than a monthly-coverage constraint.
- Task 3 partially extracts the entity correctly, but still under-specifies exclusion and inclusion constraints for the target evidence.

Tasks:

- [x] `P20.1` Split question decomposition into typed slots:
  - target entity or program
  - metric identity
  - period scope
  - granularity requirement
  - include constraints
  - exclude constraints
  - unit expectation
- [x] `P20.2` Convert question qualifiers into explicit evidence-plan constraints rather than folding them into the entity string
- [x] `P20.3` Replace low-signal query variants with a typed query-plan object:
  - primary semantic query
  - alternate lexical query
  - granularity query
  - qualifier query
- [x] `P20.4` Add targeted decomposition regressions from benchmark traces:
  - category-specific annual value
  - monthly-series aggregation
  - historical fiscal-year category extraction with exclusions
- [x] `P20.5` Add a bounded LLM-assisted decomposition fallback:
  - only when rule-based decomposition confidence is low
  - outputs typed retrieval fields, not free-form reasoning
- [x] `P20.6` Ensure the decomposition layer remains benchmark-agnostic and reusable for similar financial document tasks

Suggested code targets:

- `src/agent/retrieval_reasoning.py`
- `src/agent/context/extraction.py`
- `src/agent/contracts.py`
- `tests/test_engine_runtime.py`

Exit criteria:

- entity, metric, period, and granularity are no longer collapsed into one string
- query planning exposes semantically distinct candidates instead of near-duplicate lexical variants

## Phase 21: Source Ranking, Table-Family Selection, And Semantic Retrieval Repair

Objective:

- make retrieval choose the right document family and table family, and reopen source search when the first document is clearly wrong.

Why now:

- Task 2 finds a plausible 1953 Treasury file, but selects an annual summary table when the task requires monthly values.
- Task 3 commits to `treasury_bulletin_1959_09.json` with a weak score and never truly recovers after `missing_row`.
- Task 1 finds a usable 1940 document but still permits the wrong row and period slice to satisfy the request.

Tasks:

- [x] `P21.1` Add document-ranking features for:
  - year proximity
  - granularity fit
  - category fit
  - exclusion fit
  - historical table-family fit
- [x] `P21.2` Add table-family classification before compute:
  - monthly series
  - annual summary
  - fiscal-year comparison
  - category breakdown
  - debt or balance sheet
  - navigation or contents
- [x] `P21.3` Reject weak top candidates when ranking confidence is below threshold and continue re-query instead of immediately fetching the first document
- [x] `P21.4` On `missing_row`, `missing month coverage`, or category mismatch:
  - reopen source search when needed
  - do not stay pinned to the same document by default
- [x] `P21.5` Add retrieval repair policies that distinguish:
  - wrong document
  - wrong table family
  - right table family but incomplete row set
- [x] `P21.6` Add benchmark regressions that fail if the runtime answers from:
  - wrong time slice
  - wrong category row
  - annual summary when monthly support is required

Suggested code targets:

- `src/agent/nodes/orchestrator_retrieval.py`
- `src/agent/retrieval_tools.py`
- `src/agent/retrieval_reasoning.py`
- `src/agent/benchmarks/officeqa_validator.py`

Exit criteria:

- the runtime can abandon a bad source instead of looping on it
- table-family and granularity mismatches are caught before deterministic compute

## Phase 22: Evidence Suitability And Compute-Admissibility Guards

Objective:

- prevent compute from succeeding on semantically wrong evidence and make the validator enforce task-level suitability, not just structural completeness.

Why now:

- Task 1 currently passes internally even though it uses the wrong row and a 6-month column for a different semantic target.
- The current validator is strong on missing evidence but still too weak on semantically wrong evidence that looks numerically clean.

Tasks:

- [x] `P22.1` Add table-suitability checks before compute:
  - row-category fit
  - period-scope fit
  - aggregation-fit
  - granularity-fit
- [x] `P22.2` Add compute-admissibility rules so:
  - annual totals do not compute from partial-year columns
  - category-specific answers do not compute from all-government total rows
  - monthly-sum tasks do not compute from annual summary tables
- [x] `P22.3` Add a second semantic validator pass after compute that can reject a numerically clean but semantically wrong answer
- [x] `P22.4` Add explicit diagnostics for:
  - wrong row family
  - wrong column family
  - wrong period slice
  - wrong aggregation grain
- [x] `P22.5` Update regression scoring so local benchmark dry runs can detect false internal passes, not just pipeline completion
- [x] `P22.6` Add benchmark fixtures where the wrong table yields a plausible number and ensure the runtime still fails safely

Suggested code targets:

- `src/agent/benchmarks/officeqa_compute.py`
- `src/agent/benchmarks/officeqa_validator.py`
- `src/agent/benchmarks/officeqa_eval.py`
- `tests/test_officeqa_compute.py`
- `tests/test_officeqa_eval.py`

Exit criteria:

- semantically wrong evidence cannot pass deterministic compute and reviewer together
- local benchmark validation can detect false positives, not just hard failures

## Phase 23: Repair Orchestration And Bounded LLM Escalation

Objective:

- introduce a bounded semantic-repair path so the runtime can recover from retrieval ambiguity without abandoning the deterministic OfficeQA architecture.

Why now:

- All three benchmark traces show `total_llm_calls = 0`.
- That is correct under the current policy, but it also means:
  - there is no semantic query rewrite when source ranking is weak
  - there is no table-suitability review when several plausible tables exist
  - there is no bounded disambiguation pass after validator-guided repair

Tasks:

- [x] `P23.1` Define exact points where LLM use is allowed:
  - low-confidence decomposition
  - low-confidence source ranking
  - ambiguous table family
  - validator-directed repair suggestion
- [x] `P23.2` Keep the final numeric answer deterministic when compute is supported; the LLM may help reformulate or review, but not replace compute
- [x] `P23.3` Add a retrieval-review microstep inspired by Purple's explicit staged loop:
  - retrieve
  - parse
  - review evidence suitability
  - then compute
- [x] `P23.4` Add bounded query-rewrite and table-rerank prompts that return structured outputs only
- [x] `P23.5` Add stop rules so the LLM cannot create open-ended loops or bypass validator constraints
- [x] `P23.6` Add trace artifacts that clearly show:
  - why the LLM was invoked
  - what structured decision it returned
  - whether that changed the retrieval path

Suggested code targets:

- `src/agent/nodes/orchestrator.py`
- `src/agent/nodes/orchestrator_retrieval.py`
- `src/agent/prompts.py`
- `src/agent/tracer.py`
- `tests/test_engine_runtime.py`

Exit criteria:

- the runtime can perform bounded semantic repair when deterministic retrieval stalls
- LLM use remains explicit, typed, and non-ad hoc

## Phase 24: State Model Simplification And Trace Semantics Cleanup

Objective:

- remove overlapping fields that confuse teammates and traces while keeping the minimum state needed for retrieval, compute, and review.

Why now:

- the trace audit shows several fields overlap or differ only by audience:
  - `answer_focus` vs `routing_rationale`
  - `task_text` vs `focus_query` vs `objective`
  - repeated low-signal `query_candidates`
- some repetition is intentional history, but some is schema debt.

Tasks:

- [ ] `P24.1` Define one authoritative owner for:
  - raw user task text
  - retrieval seed text
  - solver objective
  - routing explanation
- [ ] `P24.2` Remove `answer_focus` if it remains only a wrapper around `routing_rationale`
- [ ] `P24.3` Replace free-form `query_candidates` with a typed query-plan payload if Phase 20 lands that structure cleanly
- [ ] `P24.4` Update traces so repeated raw history remains available but compact summaries show only authoritative fields
- [ ] `P24.5` Add a schema audit test that fails when the same concept is redundantly serialized through too many layers
- [ ] `P24.6` Update teammate docs with a field-ownership table for runtime state

Suggested code targets:

- `src/agent/contracts.py`
- `src/agent/curated_context.py`
- `src/agent/nodes/orchestrator_intent.py`
- `src/agent/tracer.py`
- `docs/v5_runtime_walkthrough.md`

Exit criteria:

- teammates can explain why each top-level state field exists
- compact traces stop repeating equivalent intent and objective fields

## Phase 25: Original Benchmark Regression Harness And Failure Taxonomy

Objective:

- convert original-benchmark failures into a repeatable local harness so future fixes are measured against real benchmark behavior, not only the curated smoke slice.

Why now:

- the three April 5 traces exposed failure classes that the earlier smoke slice did not cover.
- we need a stable way to prevent regressions once Phases 20-24 begin.

Tasks:

- [ ] `P25.1` Capture anonymized benchmark-failure fixtures as local regression metadata without baking task-specific hacks into runtime logic
- [ ] `P25.2` Add a benchmark failure taxonomy:
  - wrong source
  - wrong table family
  - wrong row or column semantics
  - incomplete evidence
  - false semantic pass
  - repair stall
- [ ] `P25.3` Extend evaluation reports so they classify both pipeline stage failure and semantic correctness failure
- [ ] `P25.4` Add go or no-go thresholds that require:
  - no false semantic passes
  - bounded repair-stall rate
  - acceptable source-ranking accuracy on sampled benchmark cases
- [ ] `P25.5` Keep this harness benchmark-agnostic enough to reuse for future document-grounded financial benchmarks

Suggested code targets:

- `src/agent/benchmarks/officeqa_eval.py`
- `scripts/run_officeqa_regression.py`
- `eval/`
- `tests/test_officeqa_eval.py`

Exit criteria:

- the team can measure progress on real benchmark-like failures without relying only on manual trace review

## Optional Backlog: Shared Global Workpad

Recommendation:

- useful only if it is small, structured, and run-scoped
- harmful if it becomes another vague memory store

If implemented, treat it as a checklist object, not free-form memory.

Possible shape:

- `question_type`
- `source_files_expected`
- `source_files_found`
- `tables_needed`
- `tables_extracted`
- `values_missing`
- `computations_pending`
- `final_contract_status`

Tasks:

- [x] ~~`B1` Decide whether a run-scoped `global_workpad` adds clarity after Phases 2-4 exist~~
  Skipped. The structured evidence, compute result, validator packet, and regression artifacts now cover the clarity this workpad was meant to provide.
- [x] ~~`B2` If yes, implement it as a typed checklist attached to runtime state~~
  Skipped. Adding another checklist layer would duplicate state already carried in OfficeQA provenance and report artifacts.
- [x] ~~`B3` Ensure it never replaces provenance objects or benchmark artifacts~~
  Resolved by not implementing the optional global workpad.

## Working Agreement For Coding Agents

Before starting a task:

- read `docs/officeqa_integration_plan.md`
- read this plan
- identify the exact task ids you are taking

While working:

- update this file by checking off completed tasks
- add new sub-tasks only when the current phase truly needs them
- keep code changes aligned to one phase at a time where possible

After finishing meaningful work:

- update `docs/progress.md`
- record which task ids were completed
- note any architecture decision that changed the original plan

## First Recommended Sprint

If starting immediately, the first sprint should target:

- [x] `P0.1` through `P0.7`
- [x] `P1.1` through `P1.6`
- [x] `P2.1` through `P2.4`

Do not start Phase 5 compute work before Phases 2-4 produce stable structured evidence.
