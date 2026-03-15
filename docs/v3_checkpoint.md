# v3 Checkpoint

## Purpose

This document records the current `v3` architecture checkpoint for the staged finance-first runtime.

It exists to answer four practical questions:

1. what `v2` was
2. why `v2` was unstable
3. what changed in `v3`
4. how the current `v3` system flows from request start to completion

This is a checkpoint document, not a forward-looking design pitch.

## What v2 Was

`v2` was the prompt-heavy control-loop runtime built around:

```text
coordinator -> reasoner -> tool_executor -> verifier -> format_normalizer
```

The core behavior depended on large role prompts and runtime patching:

- coordinator tried to emit route JSON
- reasoner handled planning, tool selection, reasoning, and draft generation
- tool output was routed through verifier on nearly every step
- verifier owned critique, repair routing, checkpointing, and often next-step steering
- format normalizer repaired answer shape at the end

`v2` did produce some good results on narrow slices, but it was structurally brittle.

## Why v2 Broke Down

The main faults in `v2` were architectural, not just prompt quality problems.

### 1. Too much responsibility sat inside one general model prompt

The executor prompt was carrying:

- domain behavior
- tool policy
- output rules
- file handling rules
- anti-hallucination rules
- self-repair behavior

That works poorly on hard finance/legal tasks because the model is being asked to act as planner, operator, synthesizer, and controller at once.

### 2. The coordinator was richer on paper than in reality

`v2` coordinator emitted fields like:

- confidence
- estimated steps
- early exit
- selected layers

In practice those fields did not justify their complexity. The runtime mostly reduced them to entry routing and prompt shaping, while live schema failures forced repeated heuristic fallback.

### 3. Verifier had too much control

The verifier became:

- critic
- retry controller
- checkpoint manager
- malformed-tool detector
- truncation detector
- completeness gate

That made the system hard to reason about and easy to destabilize.

### 4. Tool results were not first-class evidence

Many tool outputs still behaved like prose blobs that the model had to reinterpret. That weakened grounding and made prompt growth worse.

### 5. Runtime artifacts were under-typed

Important concepts existed only implicitly in prompts or message history:

- answer contract
- evidence classes
- assumptions
- provenance
- execution policy

That made debugging and recovery expensive.

## What v3 Changed

`v3` replaced the old prompt-heavy loop with a staged runtime built around explicit contracts and narrower node responsibilities.

### Core architectural shift

The active graph is now:

```text
intake
  -> task_profiler
  -> template_selector
  -> context_builder
  -> solver
       -> tool_runner -> solver
       -> reviewer
  -> output_adapter
  -> reflect
```

### New runtime artifacts

`v3` moves explicit artifacts through the graph instead of relying on hidden prompt state:

- `ProfileDecision`
- `ExecutionTemplate`
- `AnswerContract`
- `EvidencePack`
- `AssumptionLedger`
- `ProvenanceMap`
- `ToolResult`
- `ReviewResult`
- selective `ArtifactCheckpoint`

### What changed phase by phase

#### Phase 1: profile decision hardening

- added `primary_profile`
- added `capability_flags`
- added `ambiguity_flags`
- stopped treating first classification as absolute truth

#### Phase 2: execution template selection

- introduced static execution templates
- made routing dynamic only at selection time, not by runtime-generated FSMs
- tied tool policy and review cadence to template policy

#### Phase 3: EvidencePack v2

- separated `prompt_facts`, `retrieved_facts`, and `derived_facts`
- added explicit assumptions
- added provenance tracking

#### Phase 4: document evidence service

- replaced raw file-body use with structured document evidence
- added metadata, chunks, tables, numeric summaries, and citations

#### Phase 5: selective checkpoints and backtracking

- removed global rollback behavior
- added artifact-level checkpoints only where exactness pays off
- scoped backtracking to selected templates and stages

#### Phase 6: offline context-pack curation

- added store-only curation signals
- stored repeated missing-dimension, assumption, evidence-gap, tool-failure, and backtrack patterns
- kept curation offline instead of injecting new prompt memory into live runtime

## Current v3 Flow, Start to End

### 1. Intake

`intake` normalizes the raw request and extracts the `AnswerContract`.

Examples:

- JSON wrapper requirements
- XML root-tag requirements
- exact-format constraints

### 2. Task profiling

`task_profiler` emits a `ProfileDecision`:

- one coarse `primary_profile`
- additive `capability_flags`
- `ambiguity_flags` when the task overlaps domains

This means the system no longer depends on a single brittle label alone.

### 3. Template selection

`template_selector` maps the profile decision to one static `ExecutionTemplate`.

This step decides:

- initial stage
- allowed tools
- review cadence
- answer focus
- whether the path is ambiguity-safe

### 4. Context building

`context_builder` assembles a typed `EvidencePack` using prompt-contained information first.

It also builds:

- `AssumptionLedger`
- `ProvenanceMap`
- document placeholders or targeted evidence records when files/URLs exist

This is the main layer that provides structured finance context to a general model.

### 5. Solver loop

`solver` works stage by stage:

- `PLAN`
- `GATHER`
- `COMPUTE`
- `SYNTHESIZE`
- `REVISE`
- `COMPLETE`

The code owns stage transitions. The model works inside the current stage instead of inventing its own control flow.

### 6. Tool execution

When the solver needs a tool, it emits one structured tool call.

`tool_runner`:

- executes exactly one allowed tool
- normalizes the result into a `ToolResult`
- merges facts back into `EvidencePack`
- records new assumptions and provenance

Tool output returns to `solver`, not to a universal verifier gate on every hop.

### 7. Review

`reviewer` only reviews milestone and final artifacts.

It can return:

- `pass`
- `revise`
- `backtrack`

But backtracking is local and template-scoped, not a global ideology.

### 8. Output adaptation

If the answer contract requires exact wrapping, `output_adapter` reformats the answer without changing its substance.

### 9. Reflect and persistence

`reflect` persists the completed run into versioned SQLite tables:

- `run_memory`
- `tool_memory`
- `review_memory`
- `curation_memory`

`curation_memory` is store-only and feeds offline profile-pack / template-policy review.

## What v3 Fixed Relative to v2

The important gains are structural:

- routing is simpler and more explainable
- tool policy is explicit
- evidence is typed
- assumptions are visible
- provenance is tracked
- reviewer is narrower
- rollback is selective
- formatting is separated from reasoning
- offline improvement no longer depends on ad hoc prompt edits

## What v3 Has Not Solved

`v3` is more solid than `v2`, but it is not “finished.”

Remaining limits are mostly quality-layer issues, not graph-shape issues:

- domain tool quality still matters, especially for finance/legal depth
- live model schema compliance is still imperfect on some role calls
- profile packs and reviewer dimensions still need offline curation
- terse quant prompts can still expose model weakness even with the better graph

## Current Position

`v3` is the first runtime in this repo that is structurally coherent enough to improve systematically.

That is the real checkpoint:

- `v2` needed repeated control-loop patching
- `v3` has explicit contracts, explicit policy, explicit evidence, and offline curation hooks

Future work should now improve:

- profile packs
- evidence services
- tool quality
- template policies

without rebuilding the whole runtime again.
