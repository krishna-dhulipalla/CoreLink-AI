# OfficeQA Execution Plan

Date: 2026-03-31
Status: Completed
Source analysis: `docs/officeqa_integration_plan.md`

## Purpose

This is the canonical execution plan for turning the repository into an OfficeQA-only system.

Use this document as the operational source of truth for coding agents. The analysis document explains why the old architecture failed. This plan tells agents what to build, in what order, how to mark progress, and what "done" means for each phase.

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
