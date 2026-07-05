# AI Assistant chat panel (per-run Q&A)

Date: 2026-07-05
Status: approved

## Context

A dashboard mockup was reviewed for feature ideas. Most of it depicts a
larger enterprise platform out of scope for this v1 (multi-project dataset
management, model registry, deployments, monitoring/alerts, billing) — see
PRD.md §1.5 and PRODUCT.md. Of the mockup's features, the AI Assistant chat
panel was selected as the first to build: it is self-contained, needs no new
persistence layer, and reuses infrastructure that already exists.

## Goal

Let a user ask natural-language questions about a specific run's results
(e.g. "why is Tenure most important?", "should we try more complex models?")
and get an answer grounded in that run's already-computed data, with a short
multi-turn conversation and dynamically generated suggested questions.

## Non-negotiable scope constraint

The assistant only ever receives the same already-redacted, already-computed
JSON the report node and frontend already consume: profile summary, EDA
insights, leakage flags, feature plan, training results (incl. tuning info),
best model, and the report narrative. It never receives the raw dataset and
is given no tools — this keeps CLAUDE.md's "raw data never enters an LLM
context window" rule true by construction, with no new review surface for
tool-calling on user-typed input.

## Architecture

**Not a graph node.** Chat is reactive/on-demand, not a pipeline stage, so it
is not added to `src/graph/build_graph.py`. It lives in
`src/agents/chat_node.py`, exporting:

```python
def answer_chat_question(
    run_id: str,
    context: dict[str, Any],   # see "Context payload" below
    history: list[dict[str, str]],  # [{"role": "user"|"assistant", "content": str}, ...]
    question: str,
) -> str
```

It calls `get_llm_client().generate(run_id=run_id, node="chat", ...)` with
`json_schema=None` (free-text answer). This still satisfies CLAUDE.md's
trace-logging rule — `log_llm_call` logs by node name regardless of whether
the node is part of a `StateGraph`. `config/models.yaml` gets a new `chat`
entry (cheap/fast tier, like `understand_usecase`). Chat calls share the
run's existing `budgets.max_llm_calls_per_run` counter — no new budget
config.

**Prompt.** New template `src/agents/prompts/chat.md`, following the existing
`render_prompt` convention (tokens filled from JSON-serialized state, never
inline f-strings). Instructs the model: answer only from the given context,
say "I don't have that information for this run" rather than guess, keep
answers short and non-technical, and never claim a heuristic (leakage
detection) as a guarantee.

**Context payload.** A new helper (e.g. in `src/insights/auto_insights.py` or
a small new `_chat_context` in `server.py`) builds a trimmed dict reusing
pieces of the existing `_run_summary` computation: `task_spec`,
`profile_summary`, `eda_report.insights`, `leakage_flags`, `feature_plan`
(steps + rationale only), `training_results` (metrics + tuning per
candidate), `best_model`, `report.narrative`. This is deliberately a subset
of `_run_summary` (skips event timelines, stage messages) to keep the prompt
compact.

## State

Conversation history is per-run API session state, not pipeline state:
`_runs[run_id]["chat_history"]: list[{"role", "content", "timestamp"}]`,
initialized to `[]` in `create_run`. Not added to `PipelineState` — chat
never flows through a `StateGraph`, so CLAUDE.md rule #1 ("all node
inputs/outputs go through PipelineState") doesn't apply; this is analogous
to the existing `entry["events"]` / `entry["cancel_requested"]` fields that
already live at the API-entry level, not in pipeline state.

Multi-turn: the last 3 exchanges (6 messages) from `chat_history` are passed
as `history` into `answer_chat_question` on each call — bounded so prompt
size doesn't grow unbounded over a long conversation.

## Availability gate

The chat endpoint (and the frontend tab) is only usable once the run's
report is ready: `status in ("completed", "failed")`. Before that, the API
returns 409 and the frontend tab shows a disabled placeholder ("Available
once your model is trained").

## Suggested questions

Deterministic, no LLM call — derived from the run's own `insights` list
(already computed by `generate_insights`) plus report/best_model presence:

- any insight with `category == "imbalance"` → "Why is my data imbalanced,
  and what was done about it?"
- any insight with `category == "leakage"` → "Is there a risk of target
  leakage in this model?"
- `best_model.feature_importance` non-empty → "Why is '{top feature}' the
  strongest driver?"
- `best_model` present and `task_spec.metric` set → "How can I improve
  {metric}?" (always included as a safe default)

Capped to 4, deduplicated; if no notable insights exist, fall back to a
fixed generic set (leakage question, improve-metric question, "why was this
model chosen?", "what are the caveats?").

## API changes (`src/api/server.py`)

- `POST /api/runs/{run_id}/chat` — body `{"question": str}` → `{"answer":
  str}`. Validates `status in ("completed", "failed")` (409 otherwise),
  appends both the user question and the answer to `chat_history` with
  timestamps, calls `answer_chat_question`.
- `GET /api/runs/{run_id}` (`_run_summary`) gains two new keys:
  `chat_history` (full list, so a page refresh restores the conversation)
  and `suggested_questions` (recomputed each call — cheap, deterministic).

## Frontend changes

- `frontend/index.html`: a third tab button `tab-chat-btn` ("Ask AI") beside
  `tab-report-btn` / `tab-test-btn` in the existing `tab-bar`, and a
  `tab-chat-panel` following the same hidden/active pattern as
  `tab-test-panel`. Contains: a scrollable message list, a row of suggested-
  question chips (populate the input on click, don't auto-submit), a text
  input + send button.
- `frontend/app.js`: extend `switchTab(name)` to a three-way toggle; a
  `renderChat(run)` that renders `chat_history` and disables the tab/shows a
  placeholder when the report isn't ready yet; a submit handler that POSTs
  to `/api/runs/{id}/chat`, optimistically appends the user message, then
  appends the assistant's reply (or an inline error) on response.
- `frontend/styles.css`: message bubbles and suggested-question chips styled
  from existing tokens (`--bg-surface-raised`, `--accent-primary-soft`,
  `--text-*`), no new color families — validated against `references/palette.md`
  if any new categorical color is introduced (none are anticipated; this is
  a two-role UI — user vs assistant — styled via ink/surface tokens, not
  series color).

## Mock mode

`src/llm/client.py`'s `_mock_response` gets a `chat` case returning a fixed
canned answer referencing the question, so `AUTOML_MOCK_LLM=1` runs can
exercise the full chat flow with no API keys.

## Testing (TDD)

- Unit: suggested-question generation — each insight category maps to its
  expected question, generic fallback when insights are empty, cap-at-4 with
  dedup.
- API (`AUTOML_MOCK_LLM=1`, FastAPI `TestClient`): 409 when chat is requested
  before the report is ready; a successful question round-trip populates
  `chat_history` with both messages; a second question sees the first
  exchange reflected in the (mocked) call's history argument; `GET
  /api/runs/{id}` includes `chat_history` and `suggested_questions`.
- `test_pipeline_smoke.py` is unaffected (chat is API-only, not part of any
  `StateGraph`).

## Out of scope (this spec)

- Tool-calling / live data access from chat (flagged as a future option
  during brainstorming, deferred).
- Cross-run chat ("compare this run to my last one") — no persistence layer
  for historical runs exists yet (separate future spec, see brainstorming
  discussion of "cross-run history & model comparison").
- Streaming token-by-token responses — a single request/response per
  question, matching every other LLM call in this codebase.
