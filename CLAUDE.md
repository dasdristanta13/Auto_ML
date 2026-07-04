# CLAUDE.md

This file gives Claude (and any AI coding agent) the persistent context needed to work correctly in this repository. Read this before making changes.

## Project Overview

This is an **agentic AutoML platform**. Users upload a dataset and describe a use case in natural language. The system (LangGraph-orchestrated LLM agents) plans and executes a full ML pipeline — profiling, cleaning, feature engineering, model selection, training, evaluation, and reporting — and returns a trained model + explanation.

**Non-negotiable architectural rule: raw data never enters an LLM context window.** LLMs operate only on statistical profiles, schemas, small redacted samples, and tool results. All data manipulation happens in deterministic Python code or sandboxed execution, never inline in a prompt or as pasted rows. If you are ever tempted to pass a DataFrame or CSV content into a prompt string, stop — use the profiling/tool layer instead.

## Tech Stack

- **Orchestration**: LangGraph (Python) — StateGraph with typed `PipelineState`
- **LLM**: Claude via Anthropic API (see `src/llm/client.py`) — do not hardcode model strings; read from `config/models.yaml`
- **ML backend**: scikit-learn, XGBoost, LightGBM (optionally AutoGluon) — the LLM configures these, it does not reimplement ML logic
- **Async jobs**: Celery + Redis (or Ray, depending on `config/runtime.yaml`) for training jobs
- **Sandbox**: Docker/gVisor for executing any LLM-generated code
- **Storage**: S3-compatible object storage for datasets/artifacts; Postgres for metadata, run history, model registry
- **Observability**: LangSmith (or equivalent) for full agent trace logging

## Repository Structure

```
/src
  /agents          # one file per LangGraph node (LLM-backed nodes)
  /graph            # StateGraph definition, edges, routing functions
  /profiling        # deterministic data profiling (non-LLM)
  /tools            # typed tool functions exposed to the LLM
  /sandbox          # code validation + isolated execution
  /training         # async training job dispatch + polling
  /pii              # PII detection/redaction utilities
  /api              # FastAPI backend (upload, confirm checkpoint, status, artifacts)
  /llm              # provider-agnostic LLM client (anthropic/openai/gemini/mock) + tracing
  /state.py         # PipelineState schema (source of truth)
/frontend           # static single-page UI served by the API (no build step)
/config
  models.yaml       # which provider+model per node (planning vs code-gen vs report)
  runtime.yaml       # sandbox limits, retry caps, token budgets
/tests
  /fixtures         # synthetic datasets covering edge cases (see Testing section)
run_server.py       # web entrypoint (API + frontend)
run_local.py        # CLI entrypoint (stdin human checkpoint)
```

## Core Design Rules (do not violate)

1. **State schema is the contract.** All node inputs/outputs go through `PipelineState` (`src/state.py`). Don't pass ad hoc dicts between nodes. If you need a new field, add it to the TypedDict/Pydantic model first.
2. **Structured output over free-form code.** Whenever an operation has a known pattern (imputation, encoding, scaling, resampling), the LLM must emit a structured JSON plan validated against a schema — not Python code. Free-form code-gen is a fallback path only, and must go through the sandbox validation pipeline in `src/sandbox/`.
3. **Retry caps are mandatory.** Every conditional edge that can loop back (validation failure, unsatisfactory metrics) must check `state["retry_count"] < MAX_RETRIES` (default 3, set in `config/runtime.yaml`). Never add a loop without a cap and a graceful fallback to the report node.
4. **Training is async, never inline.** `train_model` tool dispatches a job and returns a `run_id` immediately. Do not block an LLM call on a training job. Polling happens in a separate node/loop with backoff.
5. **PII redaction happens before profiling output is constructed**, not after. Any sample values, column names with detected PII, or free-text fields must be scrubbed in `src/pii/` before they can reach `src/agents/`.
6. **Sandbox everything untrusted.** Any code string that originated from an LLM response must pass through `src/sandbox/validate.py` (AST whitelist check) and a dry-run on a data slice before running on the full dataset. No exceptions, including "trusted" internal use.
7. **Log full agent reasoning traces.** Every LLM call's prompt, tool calls, and response must be logged with the associated `run_id` for debugging and auditability. Don't strip this out for "cleanliness."

## Conventions

- Python 3.11+, type hints required on all public functions, Pydantic models for anything crossing a node boundary.
- One LLM-backed agent per file in `/src/agents`, named `<purpose>_node.py`, exporting a single `def <purpose>_node(state: PipelineState) -> PipelineState`.
- Deterministic routing functions live in `/src/graph/routing.py`, not inline lambdas in the graph definition (keep the graph file declarative and readable).
- Tool functions use the `@tool` decorator, must have a docstring describing exactly what they return (this docstring is what the LLM sees), and must never return raw row-level data beyond a capped, explicitly-sized sample (≤ 20 rows, enforced in code, not by convention).
- Prompts live in `/src/agents/prompts/` as separate `.md` or `.jinja` files, not inline strings — this keeps them reviewable and versionable.

## Testing Expectations

- Any change to `src/graph/` or `src/state.py` requires running the full fixture suite in `/tests/fixtures`, which includes: highly imbalanced classification, high-cardinality categoricals, time-series data (leakage-prone), wide datasets (500+ columns), datasets with injected PII, and datasets with ambiguous/missing target columns.
- Any change to a tool function requires a unit test asserting it never returns more than the capped row/sample limit.
- Any change to sandbox validation requires a test with intentionally malicious/broken LLM-generated code (disallowed imports, infinite loops, wrong schema output) to confirm it's caught before execution.

## What NOT to Do

- Do not add a code path that sends a full dataset, or any column beyond a capped sample, into a prompt.
- Do not remove or bypass retry caps "to make an eval pass."
- Do not let a training call block synchronously inside an agent node.
- Do not hardcode credentials, API keys, or model version strings — use `config/`.
- Do not add new LLM-backed nodes without a corresponding entry in the observability/tracing config — every node's decisions must be auditable.

## When Extending the Pipeline

If adding a new pipeline stage (e.g., a time-series-specific subgraph, a new resampling strategy), follow this checklist:
1. Extend `PipelineState` if new fields are needed.
2. Add the node under `/src/agents/` (LLM-backed) or `/src/profiling/` (deterministic).
3. Add routing logic to `/src/graph/routing.py`, with an explicit retry/fallback path.
4. Add a fixture dataset under `/tests/fixtures` exercising the new path.
5. Update `config/models.yaml` if the new node needs a specific model tier (e.g., cheaper/faster model for simple routing decisions, stronger model for planning).

## Open Questions / Areas Needing Human Review

- Target metric disambiguation currently falls back to a human-in-loop checkpoint; do not silently auto-select a metric for ambiguous use cases.
- Leakage detection heuristics (`detect_target_leakage`) are best-effort, not guaranteed — flag this explicitly in any report generated for end users.
