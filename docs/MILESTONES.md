# Project Milestones: Purple Agent

This roadmap outlines the path from initial local setup to the final competition sprint.

## Sprint 0: Foundation & "Hello World" (Current)

- **Goal**: Establish the local infrastructure and prove A2A interoperability.
- **Tasks**:
  - Clone and configure the `RDI-Foundation/agent-template`.
  - Set up Docker builds for `linux/amd64` to publish to GHCR.
  - Run a local "Hello World" A2A test against the agent using `uv run pytest --agent-url http://localhost:9009`.
  - Define the `LangGraphAgentExecutor` structure without complex logic.

## Sprint 0.5: LangGraph & MCP Integration

- **Goal**: Build the core "Brain-to-Arm" pipeline.
- **Tasks**:
  - Implement the LangGraph "Plan-Act-Learn" state machine.
  - Integrate an MCP client capable of dynamically registering and invoking external tools.
  - Connect `TaskUpdater` to stream node transitions in real-time.
  - Implement baseline Observation Masking algorithm to truncate long tool outputs.

## Sprint 1: Early Tracks (Mar 2 – Mar 22)

- **Goal**: Practice on Game Agent, Finance Agent, and Business Process Agent benchmarks.
- **Tasks**:
  - Implement In-Context Tool-Integration and orchestration for high-volume tool workflows (e.g., Business Process).
  - Refine memory management (Summarize-and-Forget) for long-duration simulations (Game track).
  - Configure specific fallback behaviors (Reflective Feedback) when the Green Agent rejects a financial transaction.

## Sprint 2: Self-Evolving Enhancements (Late March - Mid April)

- **Goal**: Fortify the agent against failure through self-correction.
- **Tasks**:
  - Implement Validator-Driven Feedback: automatic retries and strategy pivot on validation failure.
  - Optimize prompt engineering and token-saving windowing techniques.

## Sprint 3: Healthcare Agent Sprint (April 13 – May 5)

- **Goal**: Target the primary domain utilizing FHIR reasoning.
- **Tasks**:
  - Guarantee compliance with clinical communication benchmarks.
  - Test deep MCP interoperability with Green Agent FHIR tools.
  - Ensure Docker image size and stateless nature adheres precisely to strict competition guidelines.
  - Final GHCR release pipeline verification.
