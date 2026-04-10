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

### Repository Direction

- The repo started as a general A2A/LangGraph/MCP reasoning engine with finance-first routing and prompt-heavy runtime control.
- That mixed generalist plus finance-first direction is now archive context only. The active target is an OfficeQA-only benchmark agent.
- Current architecture goals are:
  - explicit benchmark adapter boundary
  - packaged or mounted OfficeQA corpus access for competition
  - OfficeQA-native retrieval state machine
  - structured evidence and provenance objects
  - deterministic compute over Treasury-derived values
  - strict output adaptation for benchmark answer contracts

### Legacy Runtime Arc

- Early work built V2/V3 around:
  - task profiles
  - template routing
  - finance/legal prompt specialization
  - selective checkpoints
  - bounded reviewer and self-reflection loops
- Finance benchmark work added:
  - options/risk/compliance deterministic paths
  - richer finance analytics and live MCP wiring
  - model/profile cleanup
  - budget and tracing improvements
- Structural cleanup before OfficeQA added:
  - package boundaries under `agent.context`, `agent.solver`, and `agent.tools`
  - benchmark-stateless mode
  - exact-quant and legal loop hardening
  - custom tracer infrastructure

### V4 History And Failure Context

- V4 introduced a new hybrid runtime with:
  - explicit planner/capability/context/executor/reviewer nodes
  - curated context instead of duplicated evidence packs
  - prompt and tracer hardening
  - legal front-loading and options-output formatting fixes
- V4 still failed for OfficeQA because it remained:
  - prompt-sensitive
  - retrieval-heuristic-heavy
  - corpus-weak
  - dependent on generic profiles/templates/capabilities for a benchmark that needs document-native retrieval and deterministic computation
- That failure led to the OfficeQA-only replatform plan captured in `docs/officeqa_integration_plan.md` and then the canonical implementation plan in `docs/officeqa_execution_plan.md`.

### OfficeQA Replatform Decisions

- Do not start a separate new repo. Keep reusable benchmark-agnostic infrastructure and replace the finance-first core in place.
- Keep:
  - A2A request/response shell
  - Judge MCP session bridge and benchmark tool loading
  - tracer infrastructure and trace folder conventions
  - final output adapter pattern
  - bounded final self-reflection
  - budget accounting and explicit stop reasons
- Replace or isolate:
  - OfficeQA dependence on generic task profiles
  - template-library routing as a control surface
  - OfficeQA heuristics in generic capability code
  - generic document retrieval loop
  - generic PDF text-window reasoning as the primary extraction method
  - prompt-compacted document reasoning as the main compute substrate

### Competition Deployment Memory

- There is no documented public guarantee that Judge or A2A will expose the OfficeQA corpus to the purple agent.
- Competition-safe design therefore assumes:
  - packaged dataset artifact, image layer, or mounted read-only OfficeQA corpus
  - local index or manifest built ahead of runtime
  - Judge document tools, if present, are optional adapters only
- Local `OFFICEQA_CORPUS_DIR` remains useful for development and regression, but not as the only assumed competition access pattern.

### OfficeQA Execution Milestones Before Recent Chats

- Phase 0:
  - docs cleanup and OfficeQA-first repo framing completed
- Phase 1:
  - benchmark adapter boundary introduced
  - OfficeQA output-contract resolution moved out of generic profiling
  - OfficeQA activation no longer depends on XML tags or prompt luck
  - major OfficeQA-specific capability policy moved toward the adapter surface
- Key pre-Phase-2 corrections:
  - keep/remove status clarified
  - reusable runtime infrastructure explicitly retained
  - remaining architecture debt concentrated in retrieval, extraction, and structured evidence

### Runtime Hygiene Memory

- Request-scoped tracing was added so benchmark runs do not overwrite each other.
- Judge bridge evolved toward session-scoped tool discovery and explicit tool invocation.
- Competition corpus bootstrap now fails fast at startup instead of failing late inside retrieval.
- Maintenance cleanup completed after Phase 5:
  - `src/agent/nodes/orchestrator.py` was split so intent heuristics live in `src/agent/nodes/orchestrator_intent.py`
  - retrieval state-machine helpers live in `src/agent/nodes/orchestrator_retrieval.py`
  - duplicate in-file retrieval planner code was removed
  - `orchestrator.py` became materially smaller and easier to reason about

### Validation Memory

- Important stable slices before the later benchmark-hardening work:
  - OfficeQA runtime slice stabilized
  - OfficeQA index and compute slices stabilized
  - combined index and compute validation became reliable enough to support the later benchmark-facing phases

### OfficeQA Runtime Buildout Memory

- Phase 2 introduced the local OfficeQA manifest and index layer, source-file resolution, numeric normalization, and benchmark override plumbing so `source_files` could reach retrieval instead of being dropped in runner/intake flow.
- Competition assumptions were corrected before deeper retrieval work:
  - packaged or mounted corpus is the main competition path
  - Judge or A2A document tools are optional only
  - startup fails fast if competition corpus bootstrap is missing
- Phase 3 replaced the old generic document loop with OfficeQA-native retrieval tools and a staged retrieval state machine.
- Phase 4 moved the runtime from prompt-compacted document snippets to structured evidence with stable provenance.
- Phase 5 added deterministic OfficeQA compute over structured values.
- Phase 6 added validator/reviewer enforcement over evidence and compute before final formatting.
- Phase 7 retired or isolated the remaining generic finance-first runtime paths around OfficeQA.
- Phase 8 added the OfficeQA eval and reporting layer plus go/no-go style regression reporting.

### Documentation And Local Ops Memory

- The README and walkthrough were reframed so the system is presented as a document-grounded financial reasoning system, with OfficeQA as the benchmark harness rather than the product identity.
- A dedicated V5 walkthrough was added to explain:
  - local vs competition resource access
  - corpus bootstrap
  - extraction
  - structured evidence
  - compute
  - validator / reviewer / self-reflection / output adapter flow
- Persistent memory is off by default for benchmark runs because V5 should not depend on older V4-style memory drift.
- Local corpus setup for practical iteration uses the parsed OfficeQA JSON corpus under `data/officeqa/source/treasury_bulletins_parsed/jsons`, followed by index build and verification.

### Retrieval, Evidence, And Multi-Stage Hardening Memory

- Phase 10 introduced adaptive retrieval strategies and evidence planning instead of a single fixed table-first loop.
- Phase 11 exposed hidden retrieval and compute decisions as explicit diagnostics in traces and reports.
- Phase 12 added answer-mode control so deterministic compute, grounded synthesis, and hybrid paths are explicit.
- Phase 13 turned validator output into an orchestration control surface rather than a generic revise loop.
- Phase 14 added explicit multi-document support and a cleaner benchmark-adapter seam.
- Research-guided extraction work then added:
  - canonical Treasury table normalization
  - confidence-aware structured evidence and compute gating
  - retrieval-surface and state deduplication
  - planner/reviewer/finalization alignment
  - an optional TSR fallback seam

### Benchmark Debug Memory Before The Current Chat Window

- Early smoke failures showed that retrieval was often reaching the right document family but failing later at table normalization, structured evidence quality, or deterministic point lookup.
- Later hardening fixed:
  - HTML table extraction gaps from parsed JSON
  - extractor hangs and timeout handling
  - navigational table false passes
  - duplicate candidate/source state entering runtime owners
- The original benchmark audit then exposed the next systemic issues:
  - weak question decomposition
  - weak semantic source ranking
  - semantically wrong but numerically valid compute
  - shallow repair behavior
- That led to:
  - typed decomposition and query planning
  - semantic source ranking and table-family selection
  - evidence-suitability and compute-admissibility guards

### Trace Interpretation Memory

- Local traces and LangSmith describe the same run from different surfaces:
  - local traces are runtime-state snapshots
  - LangSmith reflects callback/run structure
- Support LLM calls can exist even when solver usage remains `false`.
- Repeated values across local node snapshots are often historical state snapshots rather than duplicated evidence reaching compute.

---

## Recent Chats

### Chat 1: Phase 25 Completed With Original-Benchmark Failure Taxonomy And Harness Cleanup

- Added a benchmark-facing failure taxonomy so the local harness can distinguish mechanical pipeline success from benchmark-wrong answers.
- Reports now classify issues like `wrong_source`, `wrong_table_family`, `wrong_row_or_column_semantics`, `incomplete_evidence`, `false_semantic_pass`, and `repair_stall`.
- Added the sampled original-benchmark regression slice and tightened readiness logic so benchmark-like regressions affect go/no-go decisions.

### Chat 2: Phases 26-28 Completed For Benchmark Output, Trace Alignment, Decomposition/Ranking, And Deeper Repair

- Hardened output adaptation so simple final answers survive `<FINAL_ANSWER>` formatting correctly.
- Aligned local trace accounting with request-scoped tracker data so local traces and LangSmith no longer disagree as badly on call counts.
- Improved decomposition and ranking for monthly/year language, publication-year drift, and Treasury table semantics.
- Deepened repair and TSR fallback behavior while keeping it bounded and code-owned.

### Chat 3: Validation Closed And Legacy Runtime Tests Pruned

- Removed stale pre-OfficeQA runtime expectations from the regular test surface and fixed a dead import hole left behind by older options code.
- Tightened retrieval reasoning so cross-document deterministic numeric paths no longer fail on fake narrative-support requirements.
- This closed the remaining legacy test debt that was obscuring benchmark-runtime regressions.

### Chat 4: April 6 Benchmark Regression Audit Documented Without Code Changes

- Audited the April 6 sampled benchmark run and found the next real system bottlenecks were semantic source commitment, table-family admissibility, stale-state reuse after repair, and too little explicit LLM control.
- Documented how local traces and LangSmith differ:
  - local traces are state snapshots
  - LangSmith is callback/run structure
  - support LLM calls can exist even when solver usage remains `false`
- This pass was documentation-only and reopened planning around the real benchmark failures.

### Chat 5: Retrieval Roadmap Rebased Around Temporal Evidence Units

- Rebased the next plan around the discovery that some answers for year `Y` are actually published in later Treasury bulletins such as `Y+1`.
- Added the generic temporal-neighborhood idea to planning:
  - question
  - temporal intent
  - candidate evidence units
  - structural rerank
  - header-aware extraction
- numeric validation
- Kept repair invalidation and explicit LLM control as important, but moved temporal evidence-unit selection ahead of repair work.

### Chat 6: Phase 30 Implemented With Temporal-Neighborhood Evidence Retrieval

- Implemented temporal intent and evidence-unit retrieval rather than relying on exact-year file bias.
- `RetrievalIntent` now carries `period_type`, `target_years`, `publication_year_window`, and preferred publication years.
- The manifest and index now store publication metadata and table-unit metadata so search can rank by evidence units rather than only whole-file lexical overlap.
- The index schema moved to `2`, requiring local index rebuild.

### Chat 7: Phase 29 Completed With Explicit Repair Invalidation And Fresh-Hop Enforcement

- Completed repair-state invalidation so repair transitions now force fresh retrieval hops instead of reusing stale evidence.
- Added explicit reporting for `repair_applied_but_no_new_evidence` and `repair_reused_stale_state`.
- The walkthrough explains that repairs must yield new evidence, not just rewritten rationale over the same artifacts.

### Chat 8: Phase 31 Completed With An Explicit LLM Control Plane

- Added an explicit OfficeQA LLM control plane for hard financial document questions.
- The runtime now has typed, bounded LLM stages for:
  - semantic planning
  - weak-confidence source reranking
  - table admissibility reranking
  - repair support
  - final synthesis when needed
- Deterministic compute remains authoritative for supported numeric answers, but the system now has an explicit semantic assist path instead of relying almost entirely on regex and heuristics.

### Chat 9: April 8 Extraction Audit Rebased The Next Plan Around Context And Hierarchy Loss

- Reviewed the live extraction path against the real parsed corpus shape in `data/officeqa/source/treasury_bulletins_parsed/jsons/treasury_bulletin_1941_11.json`.
- Confirmed that the current extractor still parses table nodes too locally:
  - sibling `section_header` and `title` elements are present in the source JSON
  - but `_extract_tables_from_json_payload()` mainly reads node-local fields like `section_title`, `title`, and `description` from the current table wrapper
  - so tables can lose the document heading chain before ranking begins
- Confirmed that normalization is still destroying hierarchy too early:
  - `_clean_text()` and `_html_cell_text()` collapse whitespace aggressively
  - `_row_records()` replaces `section_stack` with `leading_headers[:1]`, which is too shallow for deep Treasury row hierarchies
- Confirmed that ranking is still compensating with overfit shortcuts:
  - `_classify_table_family()` uses brittle keyword tuples
  - `_rank_tables()` still has manual category boosts such as `national defense` and `veterans`
  - `_table_candidate_matches_query()` still gates candidates through a fixed HTML preview window
- Planning correction:
  - the next priority is not more compute logic
  - it is extraction robustness in this order:
    - stateful document-context projection
    - hierarchical row/header preservation
    - structure-aware candidate filtering and overfit removal
  - the bounded LLM control plane remains important, but it should consume stronger structural inputs instead of compensating for earlier state loss
- Docs updated in:
  - `docs/officeqa_execution_plan.md`
  - `docs/v5_runtime_walkthrough.md`
- No code changes in this pass by request.

### Chat 10: Phase 32 Completed With Stateful Context Projection And Continued-Table Linking

- Replaced node-local parsed-JSON table extraction with a state-aware traversal that carries page, title chain, section-header chain, nearby unit text, and nearby note context.
- Live OfficeQA table extraction now projects cleaned `heading_chain` plus raw context fields onto each extracted table payload before ranking.
- Continued tables are now linked across page breaks by detecting `(Continued)` titles and footnotes such as `(Continued on following page)`, then inheriting parent headers and context for later segments.
- The OfficeQA manifest/index now stores the same richer context on evidence units, so ranking can use sibling heading context instead of only raw HTML body overlap.
- The OfficeQA index schema moved to `3` in this phase, so local corpus indexes must be rebuilt after pulling this change.

### Chat 11: Phase 33 Completed With Hierarchy-Preserving Normalization

- Normalization now separates raw structural text from cleaned display text so hierarchy inference does not lose indentation or nonbreaking-space signals too early.
- Treasury-style visual depth encoded through leading empty `<td>` cells is now treated as structure, not discarded as empty noise.
- `row_header_depth` inference is no longer tied mainly to seeing a numeric cell early in the table:
  - header alignment
  - late descriptive rows
  - leading empty-cell structure
  now contribute to the inferred row-header boundary.
- Canonical row records now preserve hierarchy features such as:
  - `row_depth`
  - `leading_empty_cells`
  - `indentation_depth`
- The row hierarchy stack is now depth-aware, so child rows can retain their full parent path instead of collapsing to a shallow single-label section context.
- Docs updated in:
  - `docs/officeqa_execution_plan.md`
  - `docs/v5_runtime_walkthrough.md`

### Chat 12: Phase 34 Completed With Structure-Aware Candidate Filtering And Overfit Removal

- Replaced fixed-character HTML preview gating with structural table signatures built from locator/context, sampled headers, sampled row paths, unit hints, and page metadata.
- Removed the remaining benchmark-shaped ranking shortcuts such as manual `national defense` and `veterans` score boosts from table selection logic.
- Table-family classification is now driven more by structure:
  - monthly coverage
  - fiscal/calendar period shape
  - debt/balance-sheet semantics
  - row-path diversity
  - normalization confidence
- The bounded `table_rerank_llm` path now receives richer structured candidates:
  - heading chain
  - row-path samples
  - period type
  - table confidence
  - structural signature
- Evidence-unit ranking in the OfficeQA index now weights heading-chain and row-path fit more explicitly instead of relying mainly on coarse lexical overlap.
- No index schema bump in this phase; local indexes do not need a rebuild just for Phase 34.

### Chat 13: Phase 35 Completed With A Refreshed Bounded LLM Control Plane

- Reopened the old Phase 31 control plane as a post-Phase-34 refresh instead of pretending the original control logic was already calibrated to the stronger structural pipeline.
- OfficeQA LLM control budgets are now retrieval-intent aware:
  - simple deterministic year-neighborhood questions keep the smaller default rerank budget
  - harder semantic cases such as multi-table, multi-document, lower-confidence decomposition, or richer analysis modes can use a slightly larger bounded rerank budget
- Source rerank LLM use is now triggered by stronger semantic ambiguity signals:
  - publication-year mismatch
  - evidence-unit period mismatch
  - weak best-evidence confidence
  - low decomposition confidence
- Table admissibility LLM use is now triggered by stronger structural ambiguity signals:
  - period-type mismatch
  - low table confidence
  - low family confidence
  - narrow structural ranking margins
- Deterministic compute remains authoritative for supported numeric answers; this refresh only deepens bounded semantic assist behavior, not free-form compute replacement.
- Added focused regressions in `tests/test_llm_control.py` for the refreshed gating and budget logic.

### Chat 14: Retrieval Semantic Bias And Repair Diagnostics Tightened

- Removed the source-hint truncation in retrieval query planning so multi-document source bundles no longer collapse to the first two file hints when building `source_file_query`.
- Strengthened generic semantic matching with phrase-aware scoring:
  - token overlap still matters
  - but ordered 2- to 4-gram phrase matches now add explicit weight
  - exact metric/entity phrases can now outrank broader generic token coverage instead of being drowned out by long summary tables
- Kept the repair metric semantics explicit:
  - `path_changed` remains the LLM-mutation signal
  - new repair artifacts now separately record whether execution actually pivoted documents after the decision
  - traces can now distinguish `llm_path_changed` from `document_pivot_triggered`
- This avoids the earlier confusion where a repair looked like a no-op in trace history even though the runtime had actually jumped from one bulletin file to another during tool execution.

### Chat 15: Phase 36 Completed With Soft Source Constraints, Fast Rerank, And Heavy Repair Widening

- Introduced explicit OfficeQA source-constraint policy:
  - `hard`
  - `soft`
  - `off`
- Source hints are now treated as a soft prior by default whenever benchmark-linked source files exist, instead of silently becoming a hard candidate fence.
- Removed fixed truncation from the active source-hint path:
  - retrieval query planning now keeps the full source hint set
  - OfficeQA search tool args now carry the full hinted list instead of slicing it down
- Initial OfficeQA retrieval is now:
  - direct-fetch first for true single-source cases
  - search-first when the hinted source pool is ambiguous or empty
- Rebalanced the bounded LLM control plane:
  - source rerank and table admissibility now use the fast control lane
  - structured repair now uses the heavier reasoning lane
- Added an explicit `widen_search_pool` repair action that can:
  - relax source constraints
  - widen publication-year scope
  - clear stale query overrides before the next fresh retrieval hop
- Updated:
  - `docs/officeqa_execution_plan.md`
  - `docs/v5_runtime_walkthrough.md`

### Chat 16: Phase 37 Completed With Semantic-First Seeds And Generic Search-Pool Escalation

- Fixed the remaining soft-hint failure mode where OfficeQA retrieval still used the giant filename bundle as the first search query even after hard source filtering was removed.
- Active OfficeQA retrieval seeds are now semantic-first:
  - temporal query
  - primary semantic query
  - granularity query
  - qualifier / alternate lexical query
- `source_file_query` is now kept only for:
  - trace/debug visibility
  - hard-mode fallback when no semantic query exists
- Removed source-file names from `must_include_terms`, so hinted filenames no longer behave like hidden lexical match requirements during ranking.
- Added a generic `source pool too narrow` signal for candidate pools that stay inside the target-year publication slice even when the preferred publication year is outside that slice.
- When that signal appears:
  - the executor tries another semantic/temporal query first
  - fast source rerank is skipped
  - the heavy repair lane can widen the search pool
- `widen_search_pool` now also clears stale source-file query seeds so widened retrieval does not fall back to the old hinted-file string again.

### Chat 17: Phase 38 Completed With Source-Cue Cleanup And Evidence-Unit Admissibility Tightening

- Semantic planning now strips generic source-document cues out of the target entity slot:
  - `Treasury Bulletin`
  - `report`
  - `document`
  and similar provenance-only phrases no longer become the entity used for retrieval planning.
- Those source phrases can still remain as include/source constraints when they are part of the user instruction.
- The OfficeQA index now scores evidence units with explicit family fit:
  - expenditure / receipts questions prefer `category_breakdown`
  - debt questions prefer `debt_or_balance_sheet`
  - monthly-series questions prefer `monthly_series`
  - cross-family mismatches are penalized
- Fast source rerank now backs off on narrow-margin cases when the deterministic top candidate is already semantically stable:
  - strong evidence confidence
  - preferred publication year
  - same evidence-unit family among the leading candidates
- This phase reduced one class of LLM-caused regressions, but the smoke rerun still shows broader retrieval/repair problems. The main remaining bottleneck has shifted again:
  - task 2 now enters the later-year pool correctly
  - but later-year source selection and repair follow-through are still not strong enough to finish the case
