# Semantic Memory Future Plan

## Purpose

The agent now stores execution memory for future semantic retrieval, but it does **not** use that memory for live inference yet.

This is intentional.

The current goal is:

1. store the right data now
2. avoid another schema redesign later
3. keep runtime latency unchanged until retrieval quality is proven

## Current Decision

Semantic memory is **storage-only** for now.

That means:

- no embeddings at runtime
- no vector search at runtime
- no hint injection from semantic retrieval
- no prompt expansion from stored memory

The agent continues to run only on its existing graph, tools, verifier, and bounded exact-match memory behavior.

## What Must Be Stored Now

Each memory record must be future-ready for semantic indexing.

### Router Memory

Store:

- `task_signature`
- `task_summary`
- `semantic_text`
- `task_family`
- `selected_layers`
- `success`
- `cost_usd`
- `latency_ms`
- `tags`
- `metadata`

Use:

- future route recommendation
- future cost-aware routing analysis
- offline comparison of successful vs failed plans

### Executor Memory

Store:

- `task_signature`
- `partial_context_summary`
- `semantic_text`
- `task_family`
- `tool_used`
- `tool_family`
- `arguments_pattern`
- `outcome_quality`
- `success`
- `tags`
- `metadata`

Use:

- future tool-pattern retrieval
- future argument-pattern reuse
- offline analysis of which tools worked for which task families

### Verifier Memory

Store:

- `task_signature`
- `failure_pattern`
- `semantic_text`
- `task_family`
- `failure_family`
- `verdict`
- `repair_action`
- `repair_worked`
- `tags`
- `metadata`

Use:

- future repair-memory retrieval
- future failure clustering
- offline analysis of revise vs backtrack effectiveness

## Storage Rules

### Required Properties

All stored records must be:

- compact
- normalized
- bounded in size
- typed with coarse families
- serializable without external dependencies

### Required Normalization

- collapse repeated whitespace
- keep semantic text short and stable
- keep tags coarse and reusable
- keep metadata small and structured

### Required Admission Policy

- router memory: only successful runs
- executor memory: successful steps or acceptable high-signal partials
- verifier memory: only repairs that actually worked

### Required Anti-Bloat Rules

- truncate long text fields
- keep bounded table size
- deduplicate near-identical records
- reset outdated schemas instead of trying to carry legacy rows forever

## What We Explicitly Defer

The following are **not** part of the current implementation:

- semantic retrieval during live inference
- embeddings in the hot path
- vector database migration
- memory-based reranking in prompts
- automatic policy learning from memory

These are deferred because they add latency, token cost, and quality risk before the stored data has been validated.

## Future Rollout Plan

### Stage 1: Storage Validation

Status: active

Requirements:

- all three memory tables store semantic-ready fields
- old DB schema is reset safely
- records remain bounded and deduplicated
- current runtime behavior is unchanged

Exit criteria:

- tests prove new fields persist
- tests prove old DB layouts are reset
- DB contents are compact and inspectable

### Stage 2: Offline Memory Quality Review

Do this before any live retrieval work.

Requirements:

- inspect stored rows after benchmark and stress runs
- verify `task_family`, `tool_family`, and `failure_family` are useful
- verify semantic text is concise and not noisy
- verify metadata is informative but small
- measure duplicate rate and low-signal rate

Exit criteria:

- at least 100 useful records across router, executor, and verifier memory
- low-signal records are rare
- family labels are stable enough to filter on

### Stage 3: Offline Embedding Backfill

Requirements:

- generate embeddings offline, not in the live request path
- keep SQLite as source of truth
- store embeddings in a separate index or sidecar table
- allow rebuild without changing core runtime records

Exit criteria:

- nearest-neighbor results are meaningfully similar
- retrieved records are better than exact-hash matches on held-out tasks

### Stage 4: Gated Live Retrieval

Retrieval must be opt-in and narrow.

Allowed only for:

- finance-heavy tasks
- document-heavy tasks
- tasks after verifier rejection
- tool-heavy paths where repeated failure is costly

Rules:

- top 1 to 3 hints only
- strict token budget
- metadata filters before semantic similarity
- quality reranking after similarity
- do not inject raw transcripts

Exit criteria:

- latency increase is acceptable
- answer quality improves on benchmark slices
- verifier loops decrease on hard tasks

### Stage 5: Full Semantic Memory Evaluation

Requirements:

- compare against no-memory baseline
- measure tool success, verifier recovery, and cost
- keep rollback option if retrieval hurts benchmark score

Success criteria:

- better routing choices on repeated hard task families
- fewer repeated tool failures
- faster recovery after verifier rejection
- net benchmark improvement after added latency

## Non-Negotiable Constraints

- memory must remain optional
- memory must not become a raw transcript dump
- retrieval must be gated, not always-on
- SQLite remains the source of truth until semantic value is proven
- any future vector layer must be replaceable without rewriting the graph

## Immediate Next Use

For now, the DB should be treated as:

- a structured execution log
- a future semantic corpus
- an offline analysis asset

It should **not** be treated as a live retrieval engine yet.
