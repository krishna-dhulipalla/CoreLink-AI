# OfficeQA Failure Analysis and Evolution Plan

Date: 2026-03-26

## Scope

This report analyzes why the current engine failed on OfficeQA-style tasks and what should change next.

Important terminology note:

- The repository no longer runs a separate `src/agent/v4/` runtime.
- The OfficeQA-era V4 work was folded into the active top-level runtime.
- In this document, "V4" means the OfficeQA transition recorded in `docs/progress.md` chats 27-47 and the behavior visible in the current active graph under `src/agent/`.

## Primary Evidence

Local code and traces:

- `src/agent/graph.py`
- `src/agent/nodes/orchestrator.py`
- `src/agent/context/profiling.py`
- `src/agent/capabilities.py`
- `src/agent/retrieval_reasoning.py`
- `src/agent/retrieval_tools.py`
- `src/agent/curated_context.py`
- `src/mcp_servers/file_handler/server.py`
- `src/agent/model_config.py`
- `src/agent/budget.py`
- `traces/2026-03-23_00-41-24/task_001__bd928fa6-e114-45dc-8f93-6b78da4c86f1.json`
- `traces/2026-03-22_20-32-20/task_005__fa9cd56d-58ae-4e50-a971-b4f2ee3c972c.json`

External references:

- Databricks OfficeQA README: <https://github.com/databricks/officeqa>
- OfficeQA reward function: <https://raw.githubusercontent.com/databricks/officeqa/main/reward.py>
- OfficeQA AgentBeats page: <https://agentbeats.dev/arnavsinghvi11/officeqa>
- Purple Finance Worker README: <https://github.com/abhishec/purple-agent-finance-worker>
- Purple `self_reflection.py`: <https://raw.githubusercontent.com/abhishec/purple-agent-finance-worker/main/src/self_reflection.py>
- Purple `finance_output_adapter.py`: <https://raw.githubusercontent.com/abhishec/purple-agent-finance-worker/main/src/finance_output_adapter.py>

## Executive Summary

V4 did not fail on OfficeQA because one prompt was weak or one model choice was bad. It failed because the system is still architecturally finance-first, while OfficeQA is corpus-first.

OfficeQA expects an agent to:

1. identify the right Treasury Bulletin source files,
2. retrieve the right pages or tables,
3. extract exact values from document structure,
4. compute deterministically on those values,
5. return the result in the exact final-answer format.

The current engine is optimized for a different problem:

1. classify the task into a finance profile,
2. pick a generic execution mode,
3. use a mixed set of finance, retrieval, or benchmark tools,
4. synthesize from compacted evidence.

That mismatch creates two failure classes:

1. some OfficeQA tasks never enter the right document-grounded path at all;
2. tasks that do enter the document path still rely on heuristic search, weak PDF paging, lossy table extraction, and under-structured computation.

My conclusion:

- Yes, profile/template dependence still contributes to failure.
- No, profiles/templates are not the only or even the deepest root cause.
- The real blocker is the lack of a benchmark-native retrieval -> parse -> validate -> compute pipeline over the Treasury corpus.

## What OfficeQA Actually Requires

From the Databricks OfficeQA repository and AgentBeats benchmark page:

- OfficeQA uses Treasury Bulletin documents from 1939-2025.
- The corpus contains 696-697 PDFs and about 89,000 pages.
- The benchmark provides `source_docs` and `source_files` mappings.
- Databricks ships parsed JSON plus transformed text versions of the corpus.
- The parsed JSON contains richer structure, including tables and document metadata.
- The evaluation is strict on the final extracted answer, with exact or near-exact numeric comparison through `reward.py`.
- The benchmark is intentionally difficult because the relevant facts usually live inside the corpus, not in model parametric memory or general web results.

This is not a normal web-search finance task. It is document-grounded retrieval and computation over a fixed historical corpus.

## What The Current Engine Is Still Optimizing For

The active graph in `src/agent/graph.py` is:

`intake -> fast_path_gate -> task_planner -> capability_resolver -> context_curator -> executor -> reviewer -> self_reflection -> output_adapter -> reflect`

That is a reasonable general finance graph, but it treats OfficeQA as an adaptation layer inside a broad runtime, not as a first-class benchmark mode.

Evidence of finance-first assumptions still embedded in the active system:

- `src/agent/context/profiling.py` detects OfficeQA with `_looks_like_officeqa_prompt()` using a score over benchmark-like phrases.
- `src/agent/nodes/orchestrator.py` still decides `task_family` and `execution_mode` heuristically before the deeper plan.
- `src/agent/capabilities.py` widens and filters tool families through generic finance families.
- `src/agent/retrieval_reasoning.py` constructs retrieval intent from benchmark-specific heuristics and keyword patterns.
- `src/agent/retrieval_tools.py` only exposes local corpus search if a local corpus directory already exists.
- `src/mcp_servers/file_handler/server.py` is still a generic file reader, not a Treasury Bulletin parser.

## Why V4 Failed For OfficeQA

## 1. Benchmark Detection Is Still Brittle

The system still depends on prompt-shape detection to activate OfficeQA behavior.

Concrete evidence:

- `_looks_like_officeqa_prompt()` in `src/agent/context/profiling.py` requires a scored mix of years, phrases like `calendar year`, `fiscal year`, `individual calendar months`, `percent change`, `treasury bulletin`, `national defense`, `veterans administration`, and similar terms.
- In trace `traces/2026-03-23_00-41-24/task_001__bd928fa6-e114-45dc-8f93-6b78da4c86f1.json`, the task ended with:
  - `officeqa_mode: false`
  - `officeqa_like_prompt: false`
  - `officeqa_xml_contract: false`
  - `final_profile: general`
  - `final_template: advisory_analysis`
  - `stop_reason: no_bindable_capability`

That means some real OfficeQA questions were simply not recognized as OfficeQA questions.

Result:

- the engine routed them as general finance or general retrieval,
- widened to the wrong tool families,
- then stopped because no safe bounded plan could be formed.

This is a hard architectural failure, not a retrieval-quality issue.

## 2. OfficeQA Is Still Controlled By Profiles, Templates, And Routing Heuristics

Short answer to the direct question:

Yes, V4 still depends on profiles and templates enough to hurt OfficeQA.

Why:

- `fast_path_gate` in `src/agent/nodes/orchestrator.py` decides `task_family`, `execution_mode`, `review_mode`, and initial tool families before deeper execution.
- `infer_task_profile()` and `select_execution_template()` still choose from finance-centric profiles and templates.
- `src/agent/profile_packs.py` and `src/agent/template_library.py` are mostly shaped around finance quant, options, legal, research, and market workflows.
- OfficeQA currently enters mostly through the generic `document_qa` or `external_retrieval` surfaces, not through a benchmark-native Treasury workflow.

However, this needs nuance:

- profiles/templates are not the deepest root cause;
- they are the first control surface where the deeper mismatch becomes visible.

The real problem is that once OfficeQA is routed, the downstream retrieval and computation substrate is still too generic and too heuristic.

## 3. The Retrieval Substrate Is Wrong For OfficeQA

OfficeQA is fundamentally a corpus task. The engine still behaves like retrieval is optional support around a broader finance reasoner.

Evidence:

- `src/agent/graph.py` only registers built-in corpus tools when `local_corpus_available()` is true.
- `src/agent/retrieval_tools.py` searches only text-like files and does simple lexical scoring over text files.
- `src/agent/retrieval_tools.py` does not build or query a page-level or table-level index over the OfficeQA parsed corpus.
- `src/mcp_servers/file_handler/server.py` parses PDFs by extracting page text windows with `pypdf`; it does not reconstruct Treasury tables robustly.
- `MAX_TEXT_CHARS` in `src/mcp_servers/file_handler/server.py` is 12,000, which is still a generic chunking decision rather than a structured extraction design.

What that means in practice:

- PDF retrieval is page-window text extraction, not table retrieval.
- corpus search is token overlap, not source-aware retrieval over `source_files`, page markers, or table metadata.
- the system has no persistent notion of "this question needs table X, row Y, column Z, unit normalization U, then calculation C."

That is a poor fit for a benchmark whose difficulty mostly lives in table lookup and exact document-grounded arithmetic.

## 4. The Retrieval Planner Is Too Heuristic And Too Shallow

The retrieval loop in `src/agent/nodes/orchestrator.py` is deterministic and bounded, which is good. The issue is that the heuristics inside it are still too weak for OfficeQA.

Examples:

- `_rank_search_candidates()` scores titles, snippets, URLs, and a few intent tokens.
- `_retrieved_window_is_promising()` uses token overlap and citation heuristics.
- `_fallback_retrieval_action()` chooses between search, fetch, pagination, or answer using these shallow signals.
- `build_retrieval_intent()` in `src/agent/retrieval_reasoning.py` still seeds query candidates using hardcoded phrases like `Treasury Bulletin`, `official government finance`, `BLS CPI-U`, `Federal Reserve Bank of Minneapolis`, and even a fallback entity of `national defense and associated activities`.

Concrete trace evidence:

In `traces/2026-03-22_20-32-20/task_005__fa9cd56d-58ae-4e50-a971-b4f2ee3c972c.json`:

- `officeqa_mode: true`
- `officeqa_like_prompt: true`
- `officeqa_xml_contract: true`

So routing did activate.

But the retrieval path still pulled `Internal Revenue Bulletin` PDFs from GovInfo and paged through them as if they were promising evidence for a defense-expenditure task.

That trace shows:

- wrong-source search results,
- repeated pagination into irrelevant PDFs,
- insufficiency as the final answer,
- reviewer missing dimensions including aggregation semantics, metric scope, source grounding, and inline support.

This means the benchmark path can activate and still fail badly because the retrieval stack cannot reliably distinguish the correct Treasury evidence from nearby but irrelevant government PDFs.

## 5. Computation Happens Too Late And Without Structured Inputs

OfficeQA is not only about finding documents. It is about computing on extracted document values.

The current engine has exact-compute tools, but it does not yet have a strong document-to-compute bridge.

Current weakness:

- `src/agent/tools/normalization.py` creates generic `numeric_summaries` like `row_count`, `column_count`, and `numeric_cell_count`.
- `src/agent/retrieval_reasoning.py` correctly tries to ignore generic summaries during sufficiency checks.
- But there is still no durable structured table object that can drive deterministic calculations with full provenance.
- The file parser can preview tables, but that is not the same thing as extracting normalized row/column/cell values from Treasury tables.

So even when the engine reaches a potentially relevant page:

- it does not reliably build a clean monthly series,
- it does not normalize units at the extraction layer,
- it does not keep a provenance ledger for each computed number,
- it cannot robustly validate fiscal-year vs calendar-year vs monthly-sum semantics before computing.

That is why retrieval and computation currently feel loosely coupled instead of one pipeline.

## 6. Context Curation Is Too Lossy For Document-Grounded Numerical Work

`src/agent/curated_context.py` intentionally compacts tool outputs before sending them to the model. That makes sense for generic reasoning, but it is harmful for OfficeQA-style extraction.

Examples:

- strings are compacted aggressively,
- tool findings are reduced to short snippets,
- tables keep only limited headers,
- revision mode skips tool findings entirely because they are assumed to be reflected already.

For normal finance tasks this reduces prompt bloat.

For OfficeQA it can remove exactly the structural details the model needs to:

- verify the right table,
- compare month rows,
- check year alignment,
- cite the exact extracted cell,
- and decide whether a computation is safe.

In short: the system compresses evidence before it has converted that evidence into a stable structured intermediate representation.

## 7. Web Search Was A Bad Fallback For This Benchmark

This problem is partly benchmark-specific and already visible in external benchmark notes.

The OfficeQA AgentBeats page explicitly says:

- the benchmark is difficult because the needed facts live inside the corpus;
- no-tools baselines perform poorly;
- web-search baselines are non-deterministic and still weak.

That matches what happened locally:

- the engine widened into `internet_search`,
- retrieved weak or wrong government PDFs,
- then tried to recover with pagination and sufficiency heuristics.

This is exactly the path OfficeQA was designed to punish.

For OfficeQA, web search should be a last-resort debugging aid, not the main retrieval backbone.

## 8. Hardcoding Exists, But It Solves The Wrong Layer

There is a paradox in the current design:

- the runtime is already too hardcoded for OfficeQA,
- but it is not hardcoded at the right layer.

The code contains many OfficeQA-specific branches:

- prompt detection,
- XML contract activation,
- tool-family allowlists,
- retrieval-intent wording,
- source-family heuristics,
- benchmark model-profile defaults.

But those hardcodings do not create a true corpus-native execution path. They mostly patch routing, formatting, and retrieval heuristics.

So the system suffers from both:

- over-coupling to benchmark strings,
- under-investment in benchmark-native retrieval and computation.

That is the worst combination.

## Bottlenecks In The Current Runtime

| Bottleneck | Current Symptom | Why It Hurts OfficeQA |
|---|---|---|
| Benchmark recognition | Some real tasks stay `general` or `external_retrieval` | OfficeQA mode never activates |
| Tool planning | OfficeQA still flows through generic finance family logic | Wrong tools enter or valid tools never bind |
| Corpus indexing | No page/table/cell index over parsed Treasury corpus | Retrieval starts from weak search and broad page windows |
| PDF parsing | Generic text extraction, not Treasury table extraction | Tables are noisy, incomplete, and hard to compute on |
| Search ranking | Heuristic title/snippet scoring | Nearby but wrong government PDFs outrank the right evidence |
| Pagination | Page-window continuation without table-aware targeting | Many hops get spent on irrelevant pages |
| Evidence sufficiency | Semantic checks exist, but after weak extraction | Wrong-source loops end in insufficiency instead of correction |
| Computation bridge | No structured table-to-compute pipeline | Exact math cannot reliably operate on extracted values |
| Provenance | No cell-level audit trail for computed outputs | Reviewer cannot enforce source-backed numerics strongly enough |
| Context curation | Evidence is compacted before it becomes structured data | The solver sees lossy evidence, not a stable document representation |

## Did Profiles And Templates Cause The OfficeQA Failures?

Direct answer:

- Yes, partially.
- No, not by themselves.

More precise answer:

Profiles and templates caused failures in two ways:

1. they can mis-route tasks before retrieval even begins;
2. they constrain OfficeQA into generic `document_qa` or `external_retrieval` behaviors instead of a benchmark-native Treasury workflow.

But if profiles/templates were perfect, the system would still struggle because:

- retrieval is not corpus-native,
- extraction is not table-native,
- computation is not provenance-native.

So the right framing is:

- profiles/templates are amplifiers of the failure,
- not the whole failure.

## What Purple-Agent-Finance-Worker Appears To Have Solved Better

Based on the Purple README and the fetched `self_reflection.py` and `finance_output_adapter.py`, Purple solved the OfficeQA problem better in five important ways.

## 1. It Treated Format Compliance As A Separate Last-Mile Layer

Purple's `finance_output_adapter.py` explicitly detects benchmark output shape from task text and injects a format directive early, then normalizes the answer again at the end.

That matters because:

- format is handled as compliance, not reasoning;
- the main reasoning loop is not asked to "remember" exact benchmark wrappers under stress;
- OfficeQA's `<FINAL_ANSWER>` handling becomes deterministic.

This is a good pattern, and the current engine already adopted part of it.

## 2. It Used A Bounded Final Self-Reflection Pass

Purple's `self_reflection.py` does:

- a heuristic pre-check,
- then a cheap model reflection only when needed,
- and one more targeted attempt if the score is below threshold.

That is useful, but it is not the main retrieval solution. It is a quality and completeness layer.

In other words:

- reflection can rescue incomplete answers;
- it cannot rescue a broken retrieval/computation substrate.

## 3. It Framed The Runtime Around Retrieve / Parse / Validate / Compute

Purple's README describes a finance FSM with:

- `RETRIEVE`
- `PARSE`
- `ANALYZE`
- `VALIDATE`
- `COMPUTE`
- `COMPLETE`

That is much closer to OfficeQA's real structure than our current finance-first graph.

Why that matters:

- document retrieval and document parsing are explicit stages,
- validation happens before final completion,
- compute is a first-class stage rather than an optional tool family after generic synthesis.

Even if the README is high-level, that stage design is directionally correct for Treasury Bulletin tasks.

## 4. It Added Explicit Compute Verification

Purple's README also calls out a compute verifier and arithmetic audit before completion.

That is important for OfficeQA because many failures are not "I found nothing" failures. They are:

- wrong sum,
- wrong percent-change formula,
- wrong unit normalization,
- wrong period alignment,
- or a valid-looking but unsupported number.

The current engine has reviewer and sufficiency checks, but it still lacks a strong document-derived numeric audit layer.

## 5. It Treated Benchmark Tooling As Benchmark-Native

Purple's README and structure emphasize:

- MCP bridge,
- schema adaptation,
- format compliance,
- bounded quality passes,
- benchmark-native process types.

The main architectural lesson is not "copy ACE everywhere."

The main lesson is:

- separate benchmark compliance from reasoning,
- make document parsing explicit,
- validate extraction before compute,
- validate compute before final output.

## Key Takeaway On Purple

Purple appears to have solved OfficeQA less by "better web search" and more by doing four simpler things correctly and consistently:

1. benchmark-native retrieval/process stages,
2. deterministic output compliance,
3. explicit validation before compute and before final answer,
4. bounded final quality repair.

That is the right direction for this repository too.

## Recommended Evolution Path

Do not keep patching the current OfficeQA path with more benchmark strings, more query heuristics, or more prompt wording changes.

That will increase coupling without fixing the core pipeline.

Instead, split the work into the phases below.

## Phase 0: Create A Real Benchmark Adapter Boundary

Goal:

- stop scattering OfficeQA behavior across generic finance modules.

Implementation:

- add a benchmark adapter package, for example:
  - `src/agent/benchmarks/__init__.py`
  - `src/agent/benchmarks/base.py`
  - `src/agent/benchmarks/officeqa.py`
- move OfficeQA-specific behavior out of:
  - `src/agent/context/profiling.py`
  - `src/agent/capabilities.py`
  - `src/agent/retrieval_reasoning.py`
  - `src/agent/nodes/orchestrator.py`
- benchmark adapter should decide:
  - answer contract,
  - allowed tool families,
  - whether web fallback is allowed,
  - retrieval mode,
  - reviewer requirements,
  - output normalization rules.

Success criterion:

- no generic engine function needs to know what `treasury bulletin` means.

## Phase 1: Build A Corpus-Native OfficeQA Retrieval Layer

Goal:

- use the Databricks corpus as the primary substrate, not generic search.

Implementation:

- build an ingestion script, for example `scripts/build_officeqa_index.py`.
- consume OfficeQA assets:
  - `officeqa_full.csv` or `officeqa_pro.csv`
  - `treasury_bulletins_parsed/jsons/*.json`
  - `treasury_bulletins_parsed/transformed/*.txt`
- build an index with:
  - document id
  - year / month
  - source file
  - page markers
  - section titles
  - table headers
  - table rows
  - normalized numbers
  - unit hints
- persist a manifest mapping benchmark `source_files` to local indexed artifacts.

New tool surface should look more like:

- `search_officeqa_documents`
- `fetch_officeqa_pages`
- `fetch_officeqa_table`
- `lookup_officeqa_rows`
- `lookup_officeqa_cells`

Success criterion:

- OfficeQA retrieval does not depend on open web search for normal execution.

## Phase 2: Replace Generic PDF Paging With Table-First Retrieval

Goal:

- stop treating pages as the main unit of evidence when the answer lives in tables.

Implementation:

- parse transformed text and parsed JSON into stable table objects.
- use page-level search only to locate candidate tables or sections.
- once a table is located, switch to row/cell retrieval.
- store row labels, column labels, units, page numbers, and document identifiers together.

Why this matters:

- monthly sums,
- public debt lookups,
- CPI adjustments,
- year-to-year comparisons,
- and category-specific expenditures

all become much easier when the runtime works on structured tables instead of page snippets.

Success criterion:

- the solver can cite exact table row or cell provenance, not just a page excerpt.

## Phase 3: Add A Deterministic Document-To-Compute Layer

Goal:

- compute from extracted values, not from loosely summarized text.

Implementation:

- add a structured compute module, for example:
  - `src/agent/officeqa_compute.py`
  - or `src/agent/benchmarks/officeqa_compute.py`
- support deterministic operators for:
  - monthly sum
  - annual total
  - percent change
  - absolute difference
  - inflation-adjusted difference
  - fiscal-year vs calendar-year normalization
  - unit normalization for thousand / million / billion / percent
- every computed result should carry:
  - source document
  - page
  - table id
  - row / column provenance
  - operation history

Success criterion:

- reviewer can reject any numeric answer that lacks a provenance-backed compute chain.

## Phase 4: Redesign OfficeQA Retrieval Planning Around Explicit State

Goal:

- replace heuristic action choice with a stateful Treasury workflow.

Recommended OfficeQA state machine:

1. `IDENTIFY_SOURCE`
2. `LOCATE_TABLE_OR_SECTION`
3. `EXTRACT_VALUES`
4. `VALIDATE_SCOPE`
5. `COMPUTE`
6. `FORMAT_FINAL`

Validation checks before compute should include:

- correct source family
- correct entity or category
- correct time scope
- correct aggregation semantics
- sufficient numeric support
- unit consistency

This can still remain deterministic and bounded, but it should be driven by explicit extraction state instead of only title/snippet heuristics.

Success criterion:

- the engine explains why it is still searching in terms of missing structured evidence, not vague "insufficient retrieval."

## Phase 5: Keep Output Adapter And Reflection, But Move Them To The End

Goal:

- preserve the parts that already help, but stop asking them to compensate for retrieval failures.

Keep:

- OfficeQA XML formatting as a final compliance pass,
- bounded self-reflection for incomplete final answers,
- Judge MCP session handling.

Do not rely on them for:

- source identification,
- table extraction,
- numeric reconciliation,
- or retrieval recovery.

Success criterion:

- adapter and reflection improve already-grounded answers instead of masking missing evidence.

## Phase 6: Build A Real OfficeQA Regression Slice

Goal:

- stop measuring progress only by "did the full benchmark still fail."

Create a regression set grouped by failure mode:

- missed OfficeQA routing
- wrong-source retrieval
- monthly-sum extraction
- fiscal-vs-calendar confusion
- inflation adjustment
- multi-document lookup
- wrong-unit normalization
- final-format corruption

For each task, record:

- detected benchmark mode
- selected source file(s)
- table extraction success
- provenance completeness
- compute result
- final answer score

Success criterion:

- we can tell whether a failure is routing, retrieval, extraction, compute, or formatting.

## Suggested Repo-Level Changes

If I were implementing this in this repository, I would change the code in this order:

1. Add benchmark adapters.
2. Add OfficeQA corpus index build script and indexed data model.
3. Add OfficeQA-specific retrieval tools over indexed corpus.
4. Add OfficeQA compute/provenance layer.
5. Route OfficeQA tasks into the benchmark adapter path from intake.
6. Remove OfficeQA-specific hardcoding from generic profiling/capabilities/retrieval logic.
7. Keep generic `search_reference_corpus`, `fetch_reference_file`, and web search as fallbacks for non-OfficeQA tasks.

Files most likely to change:

- `src/agent/graph.py`
- `src/agent/nodes/intake.py`
- `src/agent/nodes/orchestrator.py`
- `src/agent/context/profiling.py`
- `src/agent/capabilities.py`
- `src/agent/retrieval_reasoning.py`
- `src/agent/retrieval_tools.py`
- `src/agent/curated_context.py`
- `src/mcp_servers/file_handler/server.py`
- new benchmark and indexing modules under `src/agent/benchmarks/` or similar

## What Not To Do Next

Do not spend the next round on:

- adding more OfficeQA keyword heuristics,
- improving only prompt wording,
- increasing retrieval hops again,
- widening web search fallback,
- or changing models first.

Those are second-order optimizations.

The first-order change is to give the engine a real Treasury document retrieval and computation substrate.

## Final Recommendation

Treat OfficeQA as a benchmark-native document-computation mode, not as a special case of generic finance retrieval.

The current system already has useful pieces:

- request-scoped Judge MCP bridge,
- output adapter,
- reviewer,
- self-reflection,
- budget controls,
- trace infrastructure.

Keep those.

But move OfficeQA onto a new backbone:

`benchmark adapter -> corpus index -> table retrieval -> structured extraction -> deterministic compute -> provenance validation -> final format adapter`

That is the shortest path from the current architecture to a system that can actually compete on OfficeQA without becoming permanently overfit and permanently brittle.
