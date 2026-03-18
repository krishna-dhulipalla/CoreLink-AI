# Finance Hands Plan

## Purpose

This document defines the next concrete plan for the **finance hands** layer of the v3 runtime.

The goal is not to redesign the whole architecture again. The goal is to make the system materially stronger on real financial tasks by improving:

- data access
- exact computation
- scenario and risk analysis
- compliance-aware controls
- auditability

This plan is written against the current v3 runtime:

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

## Why Finance Needs A Different Hands Layer

Finance tasks are not ordinary QA.

The hard parts are:

- decision under uncertainty
- long-horizon reasoning
- real-world consequences
- risk management
- regulatory and mandate constraints

That means a finance agent cannot rely on:

- generic prose tools
- one-shot reasoning over incomplete data
- untracked assumptions
- weak or synthetic market inputs

For finance, the hands layer must do three things well:

1. produce **trusted evidence**
2. produce **exact analytics**
3. enforce **risk/compliance gates**

## What The Current System Still Lacks

### 1. Market and company evidence is still too weak

The current MCP surface has useful beginnings, but much of it is still:

- synthetic
- option-centric
- string-heavy
- narrow in scope

Current problems in repo inspection:

- [finance/server.py](c:/Users/vamsi/OneDrive/Desktop/Gtihub_repos/Project-Pulse-Generalist-A2A-Reasoning-Engine/src/mcp_servers/finance/server.py) is mostly Black-Scholes / Greeks / mispricing logic, not broad finance evidence
- [options_chain/server.py](c:/Users/vamsi/OneDrive/Desktop/Gtihub_repos/Project-Pulse-Generalist-A2A-Reasoning-Engine/src/mcp_servers/options_chain/server.py) is still synthetic-options-analysis oriented
- [risk_metrics/server.py](c:/Users/vamsi/OneDrive/Desktop/Gtihub_repos/Project-Pulse-Generalist-A2A-Reasoning-Engine/src/mcp_servers/risk_metrics/server.py) is useful but still mostly simple analytical formulas, not portfolio governance
- `market_data` is closer to real value because it uses `yfinance`, but it is still just a data fetch layer, not a normalized finance evidence service

### 2. Decision support is missing

The current system can compute values.

It is much weaker at:

- comparing decision alternatives under uncertainty
- explaining risk/reward tradeoffs
- respecting exposure budgets
- escalating when uncertainty is too high for action

### 3. Compliance is mostly absent

Right now the system has almost no real finance/compliance policy substrate:

- no portfolio mandate engine
- no suitability / allowed-product rules
- no restricted-list or jurisdiction constraints
- no disclosure obligations model
- no execution gate before simulated or future live action

### 4. Long-horizon finance reasoning is under-supported

For real financial work, the agent often needs to:

- gather multi-source evidence
- compare multiple scenarios
- track assumptions over time
- keep audit-ready rationale

The current runtime has the right graph shape for this, but the hands layer does not yet supply enough structured financial material.

## What The External References Suggest

### 1. FinRobot: what to borrow

Reference:

- FinRobot repo: <https://github.com/AI4Finance-Foundation/FinRobot>

Important takeaways from the accessible repo README:

- it separates **Perception -> Brain -> Action**
- it treats finance as a **domain with dedicated data-source and functional layers**
- it includes real finance-oriented sources and modules:
  - SEC
  - yfinance
  - Finnhub / FMP / FinNLP utilities
  - quantitative and reporting functions
- it frames risk assessment as a first-class financial function, not an afterthought

What this means for us:

- our current v3 graph is already cleaner than the old v2 loop
- the main gap is not “we need more generic agents”
- the real gap is “our finance perception/action layer is still too thin”

**Borrow from FinRobot:**

- stronger perception/data-source layer
- finance-specific functional modules
- explicit risk-analysis capability
- report-generation from structured evidence

**Do not copy yet:**

- broad multi-agent scheduler complexity
- dynamic director/registration systems before the tool layer is strong

### 2. OfficeQA benchmark: what it warns us about

Reference:

- OfficeQA benchmark repo: <https://github.com/arnavsinghvi11/officeqa_agentbeats>

Important takeaway from the accessible repo README:

- the baseline “just an LLM with no tools” is explicitly expected to perform poorly
- the benchmark is large and operational, not toy QA
- success requires accurate **parsing, retrieval, and reasoning**

What this means for us:

- finance benchmarks will punish weak extraction and weak evidence flow
- the system must handle:
  - tables
  - document facts
  - exact numeric outputs
  - source-aware reasoning

So the finance hands plan must strengthen:

- structured extraction
- calculation reliability
- evidence-to-answer fidelity

not just “smarter prompts”.

### 3. Governance reference: risk management must be explicit

Reference:

- FINMA guidance on governance and risk management when using AI:
  <https://www.finma.ch/en/news/2024/12/20241218-mm-finma-am-08-24/>

The important operational takeaways are clear:

- AI in finance introduces:
  - model risk
  - data risk
  - IT / cyber risk
  - third-party dependency risk
  - legal and reputational risk
- governance, monitoring, and proactive risk management are not optional

What this means for us:

- every finance action path needs auditability
- every recommendation path needs explicit assumptions and provenance
- risk/compliance cannot stay implicit in the solver prompt

### 4. Access note on the ScienceDirect article

User-provided link:

- <https://www.sciencedirect.com/org/science/article/pii/S1546221825010938?utm_source=chatgpt.com>

That article URL returned `429 Too Many Requests` from this environment during drafting, so it was **not** used as a primary design input here. This plan is grounded in:

- the current repo state
- FinRobot’s accessible repo material
- the OfficeQA benchmark repo
- FINMA’s public guidance

If you want that paper integrated precisely, revisit it before implementation and add its title/key claims to this document.

## Decision: Should We Add Specific Risk / Compliance Agents?

**Yes, but not as free-running top-level agents.**

That would be the wrong first move.

The right move is:

- keep the v3 graph
- add **bounded finance specialist services / nodes**
- invoke them only on finance templates where they matter

### Add these specialists

1. `risk_controller`
- role:
  - validate position risk
  - validate scenario risk
  - validate concentration and exposure
  - require disclosure of tail-risk and sizing assumptions
- should run:
  - after `COMPUTE`
  - before final finance answer is accepted

2. `compliance_guard`
- role:
  - check mandate and product constraints
  - check jurisdiction / regulatory flags
  - check disclosure requirements
  - block unsupported recommendations
- should run:
  - after `SYNTHESIZE`
  - before final answer if task profile implies actionability or regulated content

3. `execution_gate`
- role:
  - if future simulated/live execution is allowed, require:
    - complete evidence
    - complete risk checks
    - complete compliance checks
- should run:
  - only on execution-capable templates

### Do not add these yet

- a fully autonomous “chief risk officer” agent that rewrites the whole run
- a broad legal/compliance multi-agent swarm
- any generic “trade executor” beyond simulation until policy controls are real

## The Concrete Plan

## Workstream 1: Build A Real Finance Evidence Layer

### Goal

Make `context_builder -> tool_runner -> solver` operate on real finance evidence rather than synthetic prose outputs.

### Required changes

1. Add a normalized market evidence toolkit
- target tools:
  - `get_price_history`
  - `get_returns`
  - `get_company_fundamentals`
  - `get_corporate_actions`
  - `get_yield_curve`
- change required:
  - standardize output envelopes
  - add schema stability
  - add source timestamps
  - add data quality / missing-field notes

2. Add filing and statement evidence tools
- target tools to add:
  - `get_financial_statements`
  - `get_statement_line_items`
  - `get_segment_breakdown`
  - `get_filings_section`
  - `extract_financial_table`
- expected result:
  - the agent can answer statement / filing questions without inventing values

3. Add entity resolution
- new tool:
  - `resolve_financial_entity`
- job:
  - normalize ticker / company / exchange / currency / jurisdiction / reporting basis

### Priority

High. This is the most valuable finance-hands upgrade.

## Workstream 2: Replace Generic Analytics With Exact Finance Operators

### Goal

Stop making the solver derive everything from generic text and small math utilities.

### Required changes

1. Expand finance analytics from utility math to financial operators
- add:
  - `du_pont_analysis`
  - `valuation_multiples_compare`
  - `dcf_sensitivity_grid`
  - `cashflow_waterfall`
  - `bond_spread_duration`
  - `liquidity_ratio_pack`

2. Upgrade risk metrics from single formulas to portfolio analysis
- add:
  - `scenario_pnl`
  - `factor_exposure_summary`
  - `concentration_check`
  - `drawdown_risk_profile`
  - `liquidity_stress`
  - `portfolio_limit_check`

3. Upgrade options tooling
- current tools are useful, but still too synthetic
- add:
  - strategy comparison operator with equal-format outputs
  - term-structure and skew summaries
  - assignment/exercise risk flags
  - scenario grid across spot x vol x time

### Priority

High.

## Workstream 3: Add Decision-Under-Uncertainty Support

### Goal

Support real financial recommendation tasks instead of only analytical restatement.

### Required changes

1. Add explicit scenario bundles
- new artifact in evidence:
  - `ScenarioSet`
- contains:
  - base case
  - upside case
  - downside case
  - stress case

2. Add uncertainty accounting
- solver must not hide:
  - estimated inputs
  - stale data
  - missing parameters
  - model assumptions

3. Add recommendation confidence bands
- final finance recommendations should classify:
  - high-confidence analytical result
  - scenario-dependent recommendation
  - insufficient-evidence / no-action

### Priority

High.

## Workstream 4: Add Risk Controller As A Template-Scoped Specialist

### Goal

Ensure finance outputs are checked for risk before final acceptance.

### Design

Add a bounded `risk_controller` stage, not a free agent.

### Inputs

- `ExecutionTemplate`
- `EvidencePack`
- `AssumptionLedger`
- `ToolResult`
- draft recommendation

### Outputs

- `pass`
- `revise`
- `blocked`
- `required_disclosures`
- `risk_findings`

### Checks to implement first

- concentration risk
- leverage / max loss visibility
- volatility sensitivity
- drawdown / VaR / stress exposure
- position sizing and stop-loss completeness
- mismatch between recommendation and risk profile

### Where it belongs

- required for:
  - `options_tool_backed`
  - future portfolio / allocation templates
- not required for:
  - pure descriptive finance QA

### Priority

High.

## Workstream 5: Add Compliance Guard As A Narrow Specialist

### Goal

Prevent finance recommendations from ignoring obvious mandate or regulatory constraints.

### Design

Add a bounded `compliance_guard`, not a general legal swarm.

### First version should check

- allowed product universe
- mandate restrictions
- jurisdiction flags
- disclosure obligations
- evidence and timestamp sufficiency for action-oriented claims

### New inputs needed

- `PolicyPack`
- `MandateConstraints`
- `JurisdictionContext`

### Important constraint

Do **not** attempt broad legal reasoning through this component.

This is a policy checker, not outside counsel.

### Priority

Medium-high.

## Workstream 6: Add A Finance Audit Trail

### Goal

Make financial reasoning inspectable after the fact.

### Required changes

Persist these for finance runs:

- data sources used
- timestamps
- tool outputs used in final answer
- assumptions that changed the recommendation
- risk flags raised
- compliance guard decisions
- final recommendation class

### Why

Without this, long-horizon finance tasks are hard to trust and hard to debug.

### Priority

High.

## Workstream 7: Introduce Finance-Specific Templates

The current v3 templates are still too coarse for stronger finance performance.

### Add these templates

1. `equity_research_report`
- filings + market data + valuation + risk summary

2. `portfolio_risk_review`
- exposures + scenarios + limits + recommended actions

3. `options_strategy_review`
- existing options path, but with mandatory risk-controller pass

4. `event_driven_finance`
- earnings / macro / corporate-action sensitive path

5. `regulated_actionable_finance`
- any recommendation that needs compliance guard before final output

### Priority

Medium-high.

## Workstream 8: Improve Tool Contracts

### Goal

Make tool outputs directly usable by the solver and reviewer.

### Required tool contract fields

Every finance tool should return:

```json
{
  "type": "finance_tool_type",
  "facts": {},
  "assumptions": {},
  "source": {
    "tool": "tool_name",
    "timestamp": "",
    "provider": "",
    "ticker": "",
    "jurisdiction": ""
  },
  "quality": {
    "is_synthetic": false,
    "is_estimated": false,
    "missing_fields": []
  },
  "errors": []
}
```

### Important rule

For finance, every result must clearly say whether it is:

- real retrieved evidence
- model-derived estimate
- synthetic or simulated output

This is non-negotiable.

### Priority

High.

## Workstream 9: Benchmark And Live Evaluation Policy

### Goal

Use financial evaluation to drive hands-layer improvements instead of blind prompt edits.

### Required test categories

1. Deterministic finance operator tests
- formulas
- Greeks
- duration
- P&L
- stress scenarios

2. Evidence-path tests
- filings
- statements
- tables
- market data normalization

3. Policy tests
- risk controller findings
- compliance guard blocks
- missing disclosure handling

4. Live smoke tests
- one per finance template
- assert:
  - selected template
  - tools used
  - risk/compliance gates fired when required
  - final answer includes disclosures and sources

### Benchmark expectation

Use benchmark slices to test:

- extraction fidelity
- numeric fidelity
- evidence grounding
- recommendation quality

Do not treat finance benchmarks as just “LLM knowledge tests”.

## What To Build First

This is the recommended order.

### Phase A: Real evidence and exact operators

Build first:

1. entity resolver
2. market evidence normalization
3. filing / statement extraction
4. exact finance operators

### Phase B: Risk control

Build next:

1. `risk_controller`
2. portfolio and options scenario operators
3. required disclosure rules

### Phase C: Compliance guard

Build after risk control:

1. mandate / policy packs
2. `compliance_guard`
3. actionable-finance template

### Phase D: Template expansion

Only after A-C:

1. equity research template
2. portfolio risk template
3. event-driven finance template

## What Not To Do

Do not do these next:

- do not build a large free-form multi-agent finance society first
- do not add live trade execution before risk and compliance gates exist
- do not solve weak tools with larger prompts
- do not mix synthetic and real data without explicit labeling
- do not let the solver invent missing market state silently

## Final Recommendation

The next finance push should focus on **hands quality**, not more global orchestration.

The system already has a better graph than v2.

What it lacks is a finance-grade action layer:

- real evidence
- exact operators
- scenario support
- risk controls
- compliance checks
- audit trail

That is the shortest path from “works on slices” to “can handle serious financial tasks.”

## Sources Used

- FinRobot repo:
  <https://github.com/AI4Finance-Foundation/FinRobot>
- OfficeQA benchmark repo:
  <https://github.com/arnavsinghvi11/officeqa_agentbeats>
- FINMA guidance on governance and risk management when using AI:
  <https://www.finma.ch/en/news/2024/12/20241218-mm-finma-am-08-24/>
- User-provided ScienceDirect article URL:
  <https://www.sciencedirect.com/org/science/article/pii/S1546221825010938?utm_source=chatgpt.com>
  - access note: returned `429 Too Many Requests` in this environment during drafting
