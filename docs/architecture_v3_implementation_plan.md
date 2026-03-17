# Architecture v3 Implementation Plan

## Purpose

This document turns [architecture_v3_proposal.md](c:/Users/vamsi/OneDrive/Desktop/Gtihub_repos/Project-Pulse-Generalist-A2A-Reasoning-Engine/docs/architecture_v3_proposal.md) into a concrete execution plan.

The goal is to improve the staged finance-first runtime without repeating the earlier pattern of over-building control flow and then spending days patching regressions.

This plan is intentionally conservative:

- change only one architectural dimension at a time
- preserve a runnable system after every phase
- add behavior-first tests before broad benchmark reruns
- defer high-complexity ideas until the lower layers are stable

## Scope

In scope:

- ambiguity-aware routing
- execution template selection
- richer evidence contracts
- assumptions and provenance tracking
- better document evidence handling
- selective artifact-level checkpoints
- offline context-pack curation foundations

Out of scope for this plan:

- decentralized AgentNet-style runtime
- arbitrary runtime FSM synthesis
- universal per-step verifier loops
- activation steering / SAFEsteer integration
- broad benchmark-specific logic as a primary architecture driver

## Implementation Principles

1. Keep the current staged graph shape unless a phase explicitly changes it.
2. Add one new contract or control mechanism at a time.
3. Do not let the model decide runtime topology.
4. Prefer deterministic or typed interfaces over prompt-only behavior.
5. Every new architectural feature must come with:
   - unit tests
   - at least one staged scenario test
   - a smoke path that proves the graph still completes

## Current Baseline

Current active runtime:

```text
intake
  -> task_profiler
  -> context_builder
  -> solver
       -> tool_runner -> solver
       -> reviewer
  -> output_adapter
  -> reflect
```

Current strengths:

- explicit `AnswerContract`
- explicit `EvidencePack`
- normalized `ToolResult`
- milestone/final reviewer
- store-only memory

Current weaknesses this plan addresses:

- brittle initial profile choice
- weak handling of mixed / ambiguous tasks
- insufficient provenance and assumption tracking
- weak document evidence representation
- overly generic solver behavior on harder finance/legal tasks
- no template-level execution control

## Phase Plan

### Phase 0: Freeze, Measure, and Prepare

#### Goal

Stabilize the current baseline so future changes can be measured against it.

#### Deliverables

1. Add a lightweight architecture capability matrix document:
   - current runtime artifacts
   - current node responsibilities
   - current supported task classes
2. Add stable fixture prompts for:
   - finance quant inline
   - options tool-backed
   - legal transactional
   - document QA
   - explicit live retrieval
3. Add one deterministic staged smoke summary script that records:
   - selected profile
   - tool calls
   - review count
   - final answer shape

#### Code Impact

- docs
- tests
- scripts only

#### Exit Criteria

- current staged runtime behavior is reproducible across the fixed smoke set
- no architecture change yet

---

### Phase 1: Profile Decision Hardening

#### Goal

Reduce failure from wrong first routing decisions without abandoning coarse task profiles.

#### Changes

Replace the current single profile result with a structured `ProfileDecision`:

- `primary_profile`
- `capability_flags`
- `ambiguity_flags`
- `needs_external_data`
- `needs_output_adapter`

#### Deliverables

1. New typed contract:
   - `ProfileDecision`
2. Update `task_profiler` to emit:
   - one coarse profile
   - additive flags
   - ambiguity markers when classification is mixed
3. Keep current graph shape unchanged
4. Add ambiguity-safe defaults:
   - conservative tool allowlist
   - no premature specialization when ambiguity is high

#### Important Constraint

Do not move to flags-only routing.

The runtime still needs a primary profile for:

- profile packs
- reviewer expectations
- tool policies

#### Tests

Core tests:

- legal prompt containing finance vocabulary
- quant prompt containing legal terms like liability ratio
- mixed file + math prompt
- options prompt containing general “structure options” wording

Scenario tests:

- ambiguous finance/legal mixed request must not route to a narrow unsafe tool path

#### Exit Criteria

- no regression on current smoke scenarios
- ambiguous tasks produce conservative behavior rather than brittle hard misrouting

---

### Phase 2: Execution Template Selection

#### Goal

Introduce query-dependent execution control without moving to arbitrary dynamic FSM synthesis.

#### Changes

Add a new node:

- `template_selector`

This node chooses one vetted execution template from a small library.

#### Initial Template Set

1. `quant_inline_exact`
2. `quant_with_tool_compute`
3. `options_tool_backed`
4. `legal_reasoning_only`
5. `legal_with_document_evidence`
6. `document_qa`
7. `live_retrieval`

#### Deliverables

1. New typed contract:
   - `ExecutionTemplate`
2. `template_selector` node
3. Template policy table:
   - allowed stages
   - allowed tool classes
   - review cadence
   - answer-shape expectations

#### Important Constraint

Do not generate templates dynamically with the LLM.

Selection is dynamic.
Template definitions are static and versioned in code.

#### Graph Change

Graph becomes:

```text
intake -> task_profiler -> template_selector -> context_builder -> ...
```

#### Tests

Core tests:

- each template gets selected by at least one fixture
- ambiguous profiles choose safe template defaults

Scenario tests:

- same task profile but different answer contract chooses different template

#### Exit Criteria

- template selection is deterministic and explainable
- no increase in graph instability

---

### Phase 3: EvidencePack v2

#### Goal

Upgrade context quality by making evidence source-aware, assumption-aware, and more reusable.

#### Changes

Split the current flat evidence handling into:

1. `EvidencePack`
2. `AssumptionLedger`
3. `ProvenanceMap`

#### EvidencePack v2 Fields

- `prompt_facts`
- `retrieved_facts`
- `derived_facts`
- `tables`
- `formulas`
- `citations`
- `open_questions`

#### AssumptionLedger Fields

- `assumption`
- `source`
- `confidence`
- `requires_user_visible_disclosure`
- `review_status`

#### ProvenanceMap Fields

For each fact key:

- `source_class`
- `source_id`
- `extraction_method`
- `tool_name`

#### Deliverables

1. New/updated typed contracts
2. `context_builder` upgraded to:
   - eagerly extract prompt-contained structure
   - lazily request external evidence only when justified
3. Solver prompt/input updated to consume evidence classes separately

#### Important Constraint

Do not dump raw large text into the solver if the same information can be represented as typed evidence.

#### Tests

Core tests:

- inline table extraction becomes `prompt_facts` / `tables`
- derived metrics are marked as `derived_facts`
- solver-added assumptions are recorded in `AssumptionLedger`
- provenance is attached to document extraction outputs

Scenario tests:

- options task with assumed spot price must record the assumption explicitly
- legal task with no source document must not fabricate provenance

#### Exit Criteria

- all evidence consumed by solver is source-aware
- hidden assumptions become visible in runtime artifacts

---

### Phase 4: Document Evidence Service

#### Goal

Replace raw file/document blobs with a proper evidence service that works for future long-document tasks.

#### Changes

Add a dedicated document evidence layer behind `tool_runner`.

#### Deliverables

1. New tool/result family for document evidence:
   - metadata
   - chunk extraction
   - table extraction
   - row/column filtering
   - numeric summaries
   - citations
2. Update normalization so document outputs become facts, not narratives
3. Update `context_builder` to request:
   - summary/index first
   - targeted extraction next

#### Important Constraint

Do not implement this as an OfficeQA-specific subsystem.

It must remain a general document evidence service that finance/legal/retrieval tasks can all use.

#### Tests

Core tests:

- file ref present -> metadata fetch before deep extraction
- table extraction returns structured rows/columns
- no raw giant text sent to solver by default

Scenario tests:

- legal-with-document template retrieves only targeted evidence
- document QA can answer from extracted chunk + citation set

#### Exit Criteria

- document tasks no longer rely on raw text dumps into solver prompts

---

### Phase 5: Selective Checkpoints and Backtracking

#### Goal

Introduce PRIME-style correction only where exactness and state consistency matter.

#### Changes

Add artifact-level checkpoints for selected templates only.

#### Target Templates

- `quant_with_tool_compute`
- `options_tool_backed`
- `document_qa` when extraction conflicts exist

#### Checkpoint Scope

Checkpoint only typed artifacts:

- `EvidencePack`
- `AssumptionLedger`
- `last_tool_result`
- `draft_answer`
- `review_feedback`

Do not checkpoint full message histories as the primary rollback mechanism.

#### Deliverables

1. Checkpoint contract
2. Template-level backtracking policy
3. Reviewer ability to request:
   - revise
   - backtrack
   only where the template allows it

#### Important Constraint

Do not restore the old universal verifier-as-traffic-cop behavior.

Backtracking is local and selective, not a global control ideology.

#### Tests

Core tests:

- invalid compute branch restores prior artifact state
- options bad tool result backtracks to last stable compute state
- legal-only template does not trigger heavy backtracking path

Scenario tests:

- compute branch can recover from one bad tool call without restarting the whole run

#### Exit Criteria

- backtracking exists where it pays off
- no global loop explosion

---

### Phase 6: Offline Context Pack Curation

#### Goal

Use runtime traces to improve profile packs without reintroducing prompt-injected memory.

#### Changes

Build a store-only curation pipeline inspired by ACE’s generation/reflection/curation framing.

#### Deliverables

1. Structured reflect output for:
   - repeated failure modes
   - repeated assumption issues
   - missing evidence patterns
2. A curation script that proposes updates to:
   - profile context packs
   - reviewer dimensions
   - template policy defaults
3. Human-reviewed update flow for profile packs

#### Important Constraint

Do not inject this memory directly into runtime prompts in this phase.

This is offline curation, not online memory orchestration.

#### Tests

Core tests:

- reflect emits curation-ready summaries
- curation script produces stable diffs from repeated failures

#### Exit Criteria

- profile-pack improvements can be driven by stored runtime evidence instead of ad hoc prompt edits

---

### Phase 7: Optional Safety / Policy Layer

#### Goal

Add modular deployment policy control if needed later.

#### Changes

This is where ideas adjacent to SAFEsteer belong, if they are ever needed.

#### Important Constraint

This phase should not begin until:

- evidence quality is strong
- template control is stable
- finance/document/legal quality failures are no longer primarily architectural

#### Exit Criteria

- separate from core architecture quality work

## Migration Order

This order matters.

Recommended sequence:

1. Phase 1: Profile decision hardening
2. Phase 2: Template selector
3. Phase 3: EvidencePack v2
4. Phase 4: Document evidence service
5. Phase 5: Selective checkpoints
6. Phase 6: Offline curation
7. Phase 7: Optional safety/policy layer

Do not reorder Phase 5 ahead of Phase 3.

Backtracking before evidence and assumptions are typed will create noise, not robustness.

## Testing Strategy

### 1. Behavior-first tests

Every phase must add:

- unit tests for new contracts and node decisions
- scenario tests for the changed runtime behavior
- one deterministic smoke path proving graph completion

### 2. Live smoke before benchmark

Before any benchmark rerun:

- run the staged deterministic smoke
- run a focused live smoke on the affected templates only

### 3. Benchmark policy

Do not use full benchmark reruns as the primary debugging tool.

Use benchmark slices only after:

- contracts compile
- node tests pass
- scenario tests pass
- live smoke is stable

## Rollback Strategy

Each phase must preserve a clear rollback point.

Rollback rules:

1. if a phase changes control flow, keep the previous graph path behind a feature flag until tests pass
2. if a phase changes contracts, provide temporary adapters for one phase only
3. if a phase increases revise/backtrack counts materially, stop and inspect before continuing

## Risks

### Risk 1: Template explosion

Mitigation:

- start with 6 to 7 templates only
- merge templates unless behavior genuinely diverges

### Risk 2: Evidence contract bloat

Mitigation:

- keep the solver payload compact
- store rich provenance but only pass compact slices to the model

### Risk 3: Reviewer regains too much control

Mitigation:

- milestone/final review only
- template-specific backtracking policy

### Risk 4: Too much work aimed at document corpora too early

Mitigation:

- keep finance-first tasks as the primary design testbed
- build general document evidence primitives, not benchmark-specific logic

## Phase Exit Checklist

Before starting the next phase, all of the following should be true:

1. the active graph still completes on deterministic smoke cases
2. new contracts are typed and tested
3. live smoke for the changed path is stable
4. no major token-cost blowup is introduced without justification
5. progress is documented before moving on

## Final Recommendation

The next implementation cycle should not aim for "more advanced architecture" as a status symbol.

It should aim for:

- fewer brittle decisions
- better evidence
- clearer assumptions
- more controlled specialization
- cheaper debugging

That is the right path to a solid system.
