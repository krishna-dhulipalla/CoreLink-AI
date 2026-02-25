# Project Plan Details

## Project Objective

Develop the "Purple Agent" for Phase 2 of the AgentX-AgentBeats Competition. The agent will serve as a generalized reasoning engine conforming to the A2A protocol and communicating with Green Agent benchmark evaluators.

## Technical Requirements

- **Framework**: `RDI-Foundation/agent-template`.
- **Intelligence**: LangGraph (Graph-based LLM orchestration) implementing the "Plan-Act-Learn" loop derived from arXiv:2601.12538.
- **Interoperability**: Benchmark-agnostic. Tools are instantiated at runtime via the Model Context Protocol (MCP) served by Green Agents.
- **Packaging**: Final artifact must be a stateless Docker image targeting `linux/amd64`, published automatically to the GitHub Container Registry (GHCR).

## Execution Strategy

1. **Infrastructure**: Leverage the existing Action workflows in `agent-template` for building and testing the Docker container. Ensure `pyproject.toml` is updated with `langgraph`, `langchain`, and relevant MCP SDK dependencies.
2. **Architecture**: As documented in `DESIGN.md`, separation of concerns is critical:
   - **State**: LangGraph `StateGraph` object to define typed context memory.
   - **Control**: Node-edge logic implementing Plan, Execute Tool, Reflect, Summarize.
3. **Observation Masking**: Implement custom LangChain `MessagesModifier` or node logic that periodically checks token limits and summarizes older `<Event>` context while keeping the current task loop active.
4. **Interface**: As documented in `A2A_INTERFACE_SPEC.md`, convert LangGraph event streams directly into A2A `event_queue` updates cleanly formatted to reduce latency.
5. **Timeline Execution**: Follow `MILESTONES.md`, ensuring stable "Hello World" deployments before injecting the sophisticated feedback loops necessary for Sprint 3's Healthcare FHIR benchmarks.
