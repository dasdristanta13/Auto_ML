# AI Assistant Chat Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user ask natural-language questions about one completed run's already-computed results (profile, insights, feature plan, training results, report) from a new "Ask AI" tab, with a short multi-turn conversation and dynamically generated suggested questions.

**Architecture:** A new non-graph LLM call (`src/agents/chat_node.py`) invoked directly by a new FastAPI endpoint, never as a `StateGraph` node. It only ever receives the same already-redacted, already-computed JSON the report/frontend already show — no raw dataset, no tools. Conversation history lives in the API's existing in-memory `_runs[run_id]` entry, not in `PipelineState`. Suggested questions are computed deterministically (no LLM call) from the run's existing insights.

**Tech Stack:** FastAPI (existing `src/api/server.py`), the existing provider-agnostic `src/llm/client.py`, vanilla JS/CSS frontend (no build step), pytest + FastAPI `TestClient`.

## Global Constraints

- The assistant must never receive the raw dataset or any tool access — only the trimmed, already-computed context described in the design spec (`docs/superpowers/specs/2026-07-05-ai-assistant-chat-design.md`).
- Chat is gated to `status in ("completed", "failed")` — enforced in the API (409 otherwise), not just the UI.
- Conversation history passed into each LLM call is capped to the last 3 exchanges (6 messages).
- Chat LLM calls use `node="chat"` and share the run's existing `budgets.max_llm_calls_per_run` counter — no new budget config.
- Suggested questions are capped at 4, deduplicated, with a fixed fallback set when no insights are notable.
- Follow existing repo conventions exactly: prompts live in `src/agents/prompts/*.md` (never inline strings), `render_prompt()` fills `{{TOKEN}}` placeholders, every LLM call goes through `get_llm_client().generate(...)` so it is trace-logged automatically.
- `AUTOML_MOCK_LLM=1` must fully exercise the new endpoint with no API keys/network (existing repo-wide convention).

---

### Task 1: Deterministic suggested-questions helper

**Files:**
- Modify: `src/insights/auto_insights.py`
- Test: `tests/test_chat_suggestions.py` (create)

**Interfaces:**
- Produces: `suggested_questions(insights: list[dict[str, Any]], task_spec: dict[str, Any], best_model: dict[str, Any]) -> list[str]` — used by Task 3's `_run_summary`/endpoint code.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chat_suggestions.py`:

```python
"""suggested_questions() derives chat-panel suggestion chips deterministically
(no LLM call) from a run's own insights/results — see
docs/superpowers/specs/2026-07-05-ai-assistant-chat-design.md."""

from __future__ import annotations

from src.insights.auto_insights import suggested_questions


def test_imbalance_insight_yields_imbalance_question():
    insights = [{"id": "class_imbalance", "category": "imbalance", "tone": "warning", "message": "..."}]
    qs = suggested_questions(insights, {"metric": "f1"}, {"candidate_name": "rf"})
    assert "Why is my data imbalanced, and what was done about it?" in qs


def test_leakage_insight_yields_leakage_question():
    insights = [{"id": "leakage_flag", "category": "leakage", "tone": "danger", "message": "..."}]
    qs = suggested_questions(insights, {}, {})
    assert "Is there a risk of target leakage in this model?" in qs


def test_top_feature_importance_yields_feature_question():
    best_model = {"feature_importance": [{"feature": "tenure_months", "importance": 0.4}]}
    qs = suggested_questions([], {}, best_model)
    assert "Why is 'tenure_months' the strongest driver?" in qs


def test_metric_present_yields_improve_metric_question():
    best_model = {"candidate_name": "rf"}
    qs = suggested_questions([], {"metric": "f1"}, best_model)
    assert "How can I improve f1?" in qs


def test_no_notable_signal_falls_back_to_generic_set():
    qs = suggested_questions([], {}, {})
    assert qs
    assert len(qs) <= 4


def test_never_exceeds_four_and_has_no_duplicates():
    insights = [{"category": "imbalance"}, {"category": "leakage"}]
    best_model = {
        "feature_importance": [{"feature": "x", "importance": 0.9}],
        "candidate_name": "rf",
    }
    qs = suggested_questions(insights, {"metric": "f1"}, best_model)
    assert len(qs) <= 4
    assert len(qs) == len(set(qs))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_chat_suggestions.py -v`
Expected: FAIL with `ImportError: cannot import name 'suggested_questions' from 'src.insights.auto_insights'`

- [ ] **Step 3: Implement `suggested_questions`**

Open `src/insights/auto_insights.py`. Add near the top, after the existing `_MIN_ROWS_FOR_CARDINALITY_CHECK = 20` constant:

```python
_MAX_SUGGESTED_QUESTIONS = 4
_FALLBACK_SUGGESTED_QUESTIONS = [
    "Is there a risk of target leakage in this model?",
    "What are the caveats I should know about?",
    "Why was this model chosen over the alternatives?",
]
```

Then add this function after `generate_insights` (at the end of the file):

```python
def suggested_questions(
    insights: list[dict[str, Any]], task_spec: dict[str, Any], best_model: dict[str, Any]
) -> list[str]:
    """Deterministic (no LLM call) prompts for the chat panel's suggestion
    chips, derived from what's actually notable in THIS run so they're
    relevant rather than generic. Capped at 4, deduplicated; falls back to a
    fixed generic set when nothing stands out."""
    questions: list[str] = []
    categories = {i.get("category") for i in insights}

    if "imbalance" in categories:
        questions.append("Why is my data imbalanced, and what was done about it?")
    if "leakage" in categories:
        questions.append("Is there a risk of target leakage in this model?")

    importance = best_model.get("feature_importance") or []
    if importance:
        questions.append(f"Why is '{importance[0]['feature']}' the strongest driver?")

    metric = task_spec.get("metric")
    if best_model.get("candidate_name") and metric:
        questions.append(f"How can I improve {metric}?")

    if not questions:
        questions = list(_FALLBACK_SUGGESTED_QUESTIONS)

    seen: set[str] = set()
    deduped: list[str] = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped[:_MAX_SUGGESTED_QUESTIONS]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_chat_suggestions.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/insights/auto_insights.py tests/test_chat_suggestions.py
git commit -m "feat: add deterministic suggested-questions helper for AI assistant chat"
```

---

### Task 2: Chat prompt template + `answer_chat_question`

**Files:**
- Create: `src/agents/prompts/chat.md`
- Create: `src/agents/chat_node.py`
- Test: `tests/test_chat_node.py` (create)

**Interfaces:**
- Consumes: `render_prompt(template_name: str, **tokens) -> str` from `src/agents/prompt_utils.py` (existing); `get_llm_client() -> LLMClient` and `LLMClient.generate(run_id, node, system_prompt, user_prompt, json_schema=None, retries=1) -> Any` from `src/llm/client.py` (existing).
- Produces: `answer_chat_question(run_id: str, context: dict[str, Any], history: list[dict[str, str]], question: str) -> str` — used by Task 3's chat endpoint.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chat_node.py`:

```python
"""answer_chat_question is NOT a StateGraph node (see
docs/superpowers/specs/2026-07-05-ai-assistant-chat-design.md) — it's called
directly by the chat API endpoint. These tests monkeypatch LLMClient.generate
the same way tests/test_pipeline_smoke.py does, so no API keys/network are
needed."""

from __future__ import annotations

from src.agents.chat_node import answer_chat_question
from src.llm.client import LLMClient


def test_answer_chat_question_calls_llm_with_question_as_user_prompt(monkeypatch):
    captured = {}

    def _fake_generate(self, run_id, node, system_prompt, user_prompt, json_schema=None, retries=1):
        captured.update(
            run_id=run_id, node=node, system_prompt=system_prompt,
            user_prompt=user_prompt, json_schema=json_schema,
        )
        return "the answer"

    monkeypatch.setattr(LLMClient, "generate", _fake_generate)

    result = answer_chat_question(
        run_id="r1",
        context={"task_spec": {"metric": "f1"}},
        history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        question="why is tenure important?",
    )

    assert result == "the answer"
    assert captured["run_id"] == "r1"
    assert captured["node"] == "chat"
    assert captured["user_prompt"] == "why is tenure important?"
    assert captured["json_schema"] is None
    assert "f1" in captured["system_prompt"]


def test_answer_chat_question_trims_history_to_last_three_exchanges(monkeypatch):
    captured = {}

    def _fake_generate(self, run_id, node, system_prompt, user_prompt, json_schema=None, retries=1):
        captured["system_prompt"] = system_prompt
        return "ok"

    monkeypatch.setattr(LLMClient, "generate", _fake_generate)

    long_history = [{"role": "user", "content": f"question {i}"} for i in range(10)]
    answer_chat_question(run_id="r1", context={}, history=long_history, question="latest?")

    # last 6 messages of a 10-message list are indices 4..9
    assert "question 9" in captured["system_prompt"]
    assert "question 4" in captured["system_prompt"]
    assert "question 3" not in captured["system_prompt"]
    assert "question 0" not in captured["system_prompt"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_chat_node.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.chat_node'`

- [ ] **Step 3: Write the prompt template**

Create `src/agents/prompts/chat.md`:

```
You are the AI Assistant embedded in an agentic AutoML platform, answering a
user's question about ONE specific run.

Only answer using the context below — the same already-computed,
already-redacted information already shown to the user elsewhere in the
product. You have no access to the raw dataset and no tools. If the answer
isn't determinable from this context, say so plainly rather than guessing.

Keep answers short (a few sentences), plain-language, and free of unexplained
ML jargon. Never present a heuristic (e.g. target-leakage detection) as a
guarantee.

## This run's context
{{RUN_CONTEXT_JSON}}

## Conversation so far (may be empty)
{{CHAT_HISTORY_JSON}}
```

- [ ] **Step 4: Implement `chat_node.py`**

Create `src/agents/chat_node.py`:

```python
"""LLM-backed, on-demand Q&A about ONE completed run's already-computed
results. NOT a StateGraph node — see
docs/superpowers/specs/2026-07-05-ai-assistant-chat-design.md. Invoked
directly by src/api/server.py's chat endpoint, only after the run's report
is ready. Only ever sees the same already-redacted, already-computed data
the report/frontend already show; no raw dataset access, no tools."""

from __future__ import annotations

from typing import Any

from src.agents.prompt_utils import render_prompt
from src.llm.client import get_llm_client

_MAX_HISTORY_MESSAGES = 6  # last 3 exchanges, so prompt size stays bounded


def answer_chat_question(
    run_id: str,
    context: dict[str, Any],
    history: list[dict[str, str]],
    question: str,
) -> str:
    system_prompt = render_prompt(
        "chat.md",
        RUN_CONTEXT_JSON=context,
        CHAT_HISTORY_JSON=history[-_MAX_HISTORY_MESSAGES:],
    )
    return get_llm_client().generate(
        run_id=run_id,
        node="chat",
        system_prompt=system_prompt,
        user_prompt=question,
        json_schema=None,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_chat_node.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/agents/prompts/chat.md src/agents/chat_node.py tests/test_chat_node.py
git commit -m "feat: add chat_node for AI assistant per-run Q&A"
```

---

### Task 3: API endpoint, chat state, and mock-mode wiring

**Files:**
- Modify: `src/api/server.py`
- Modify: `config/models.yaml`
- Modify: `src/llm/client.py`
- Test: `tests/test_api_chat.py` (create)

**Interfaces:**
- Consumes: `answer_chat_question(...)` from Task 2; `suggested_questions(...)` from Task 1; existing `generate_insights`, `_json_safe`, `_get_entry`, `_lock`, `_runs`.
- Produces: `POST /api/runs/{run_id}/chat` (`{"question": str}` → `{"answer": str}`, 409 before ready); `GET /api/runs/{run_id}` response gains `chat_history` and `suggested_questions` keys.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api_chat.py`:

```python
"""API integration tests for the AI Assistant chat endpoint (see
docs/superpowers/specs/2026-07-05-ai-assistant-chat-design.md). Runs the
whole intake -> confirm -> approve-features -> train flow through the real
HTTP layer with AUTOML_MOCK_LLM=1 so no API keys/network are needed."""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import server


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("AUTOML_MOCK_LLM", "1")
    return TestClient(server.app)


def _small_csv_bytes() -> bytes:
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "tenure_months": rng.normal(12, 5, n),
            "monthly_spend": rng.normal(60, 15, n),
            "churned": rng.choice([0, 1], n, p=[0.8, 0.2]),
        }
    )
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _wait_for_status(client, run_id, statuses, timeout=30.0):
    deadline = time.monotonic() + timeout
    run = None
    while time.monotonic() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in statuses:
            return run
        time.sleep(0.2)
    raise TimeoutError(f"run {run_id} never reached {statuses}, last status was {run['status'] if run else '?'}")


def _create_run(client) -> str:
    res = client.post(
        "/api/runs",
        files={"file": ("churn.csv", _small_csv_bytes(), "text/csv")},
        data={"description": "predict which customers will churn"},
    )
    assert res.status_code == 200
    return res.json()["run_id"]


def test_chat_before_report_ready_returns_409(client):
    run_id = _create_run(client)

    res = client.post(f"/api/runs/{run_id}/chat", json={"question": "why?"})

    assert res.status_code == 409


def test_chat_round_trip_after_report_ready(client):
    run_id = _create_run(client)
    _wait_for_status(client, run_id, {"awaiting_confirmation"})

    confirm = client.post(
        f"/api/runs/{run_id}/confirm",
        json={
            "target_column": "churned",
            "task_type": "classification",
            "metric": "f1",
            "cv_enabled": False,
            "cv_folds": 2,
            "tuning_enabled": False,
        },
    )
    assert confirm.status_code == 200
    _wait_for_status(client, run_id, {"awaiting_feature_approval", "failed"})

    approve = client.post(
        f"/api/runs/{run_id}/approve-features",
        json={"approved_step_indices": [], "resampling_enabled": False, "resampling_method": "none"},
    )
    assert approve.status_code == 200
    run = _wait_for_status(client, run_id, {"completed", "failed"})
    assert run["status"] == "completed", run.get("errors")

    assert run["suggested_questions"]
    assert run["chat_history"] == []

    first = client.post(f"/api/runs/{run_id}/chat", json={"question": "why was this model chosen?"})
    assert first.status_code == 200
    answer_1 = first.json()["answer"]
    assert answer_1

    second = client.post(f"/api/runs/{run_id}/chat", json={"question": "what about caveats?"})
    assert second.status_code == 200

    run = client.get(f"/api/runs/{run_id}").json()
    history = run["chat_history"]
    assert [h["role"] for h in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == "why was this model chosen?"
    assert history[1]["content"] == answer_1
    assert history[2]["content"] == "what about caveats?"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_api_chat.py -v`
Expected: FAIL — `test_chat_before_report_ready_returns_409` fails with a 404 (no `/chat` route exists yet, so FastAPI returns 404 Not Found, not the 409 the test asserts).

- [ ] **Step 3: Add the `chat` node to `config/models.yaml`**

Open `config/models.yaml`. After the `report:` block at the end of the file, add:

```yaml

  chat:
    provider: openai
    model: gpt-5-nano
    temperature: 0.2
    max_tokens: 1024
```

- [ ] **Step 4: Add the mock response for `node == "chat"`**

Open `src/llm/client.py`. In `_mock_response`, immediately before the final `raise ValueError(...)` line, add:

```python
    if node == "chat":
        return (
            "MOCK-MODE ANSWER (AUTOML_MOCK_LLM=1 — no real LLM was called). "
            "In a real run, this would answer your question using only this "
            "run's already-computed profile, insights, feature plan, "
            "training results, and report."
        )
```

- [ ] **Step 5: Wire the endpoint into `src/api/server.py`**

Add the import at the top, alongside the existing imports from `src.insights.auto_insights`:

Find:
```python
from src.insights.auto_insights import generate_insights
```

Replace with:
```python
from src.agents.chat_node import answer_chat_question
from src.insights.auto_insights import generate_insights, suggested_questions
```

Add `"chat_history": []` to the entry dict in `create_run`. Find:
```python
        _runs[run_id] = {
            "state": state,
            "status": "profiling",
            "events": [],
            "filename": file.filename,
            "created_at": time.time(),
            "finished_at": None,
            "cancel_requested": False,
        }
```

Replace with:
```python
        _runs[run_id] = {
            "state": state,
            "status": "profiling",
            "events": [],
            "filename": file.filename,
            "created_at": time.time(),
            "finished_at": None,
            "cancel_requested": False,
            "chat_history": [],
        }
```

Add a `_chat_context` helper just before `_run_summary`. Find:
```python
def _run_summary(run_id: str, entry: dict[str, Any]) -> dict[str, Any]:
```

Insert immediately before it:
```python
def _chat_context(state: PipelineState) -> dict[str, Any]:
    """Trimmed subset of _run_summary's already-redacted, already-computed
    fields for the chat prompt — skips event timelines/stage messages to
    keep the prompt compact (see the chat design spec)."""
    feature_plan = state.get("feature_plan") or {}
    return _json_safe(
        {
            "task_spec": state.get("task_spec"),
            "profile_summary": {
                "row_count": state.get("profile", {}).get("row_count"),
                "column_count": state.get("profile", {}).get("column_count"),
            },
            "eda_insights": (state.get("eda_report") or {}).get("insights", []),
            "leakage_flags": state.get("leakage_flags", []),
            "feature_plan_steps": [
                {"op": s.get("op"), "columns": s.get("columns"), "rationale": s.get("rationale")}
                for s in feature_plan.get("steps", [])
            ],
            "training_results": [
                {
                    "candidate_name": r.get("candidate_name"),
                    "status": r.get("status"),
                    "metrics": r.get("metrics"),
                    "tuning": r.get("tuning"),
                }
                for r in state.get("training_results", [])
            ],
            "best_model": state.get("best_model"),
            "report_narrative": (state.get("report") or {}).get("narrative"),
        }
    )
```

Update `_run_summary` to compute `insights` once and add the two new keys. Find:
```python
            "profile_columns": _profile_columns(state),
            "leakage_flags": state.get("leakage_flags", []),
            "feature_plan": state.get("feature_plan"),
            "training_results": state.get("training_results", []),
            "best_model": state.get("best_model"),
            "insights": generate_insights(state, stages_done),
            "report": state.get("report", {}).get("narrative"),
            "errors": state.get("errors", []),
        }
    )
```

Replace with:
```python
            "profile_columns": _profile_columns(state),
            "leakage_flags": state.get("leakage_flags", []),
            "feature_plan": state.get("feature_plan"),
            "training_results": state.get("training_results", []),
            "best_model": state.get("best_model"),
            "insights": insights,
            "report": state.get("report", {}).get("narrative"),
            "errors": state.get("errors", []),
            "chat_history": entry.get("chat_history", []),
            "suggested_questions": suggested_questions(
                insights, state.get("task_spec") or {}, state.get("best_model") or {}
            ),
        }
    )
```

And add the `insights` local variable right before the `return _json_safe(` line in the same function. Find:
```python
    events = _plain_language_events(state, stages_done)
    for event in events:
        event["timestamp"] = completed_at_by_node.get(event["stage"])

    return _json_safe(
```

Replace with:
```python
    events = _plain_language_events(state, stages_done)
    for event in events:
        event["timestamp"] = completed_at_by_node.get(event["stage"])

    insights = generate_insights(state, stages_done)

    return _json_safe(
```

Finally, add the endpoint itself. Find:
```python
@app.get("/api/runs/{run_id}/trace")
def get_trace(run_id: str) -> list[dict[str, Any]]:
    _get_entry(run_id)
    return _json_safe(read_trace(run_id))
```

Replace with:
```python
@app.get("/api/runs/{run_id}/trace")
def get_trace(run_id: str) -> list[dict[str, Any]]:
    _get_entry(run_id)
    return _json_safe(read_trace(run_id))


class ChatRequest(BaseModel):
    question: str


@app.post("/api/runs/{run_id}/chat")
def chat(run_id: str, body: ChatRequest) -> dict[str, Any]:
    """Answer a question about this run's already-computed results (see
    docs/superpowers/specs/2026-07-05-ai-assistant-chat-design.md). Only
    available once the report is ready — gated here, not just in the UI, so
    the contract holds regardless of client."""
    entry = _get_entry(run_id)
    with _lock:
        if entry["status"] not in ("completed", "failed"):
            raise HTTPException(status_code=409, detail=f"run is '{entry['status']}', not ready for questions yet")
        state = entry["state"]
        context = _chat_context(state)
        history_for_prompt = [{"role": h["role"], "content": h["content"]} for h in entry["chat_history"]]

    answer = answer_chat_question(run_id=run_id, context=context, history=history_for_prompt, question=body.question)

    with _lock:
        now = time.time()
        entry["chat_history"].append({"role": "user", "content": body.question, "timestamp": now})
        entry["chat_history"].append({"role": "assistant", "content": answer, "timestamp": now})

    return {"answer": answer}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_api_chat.py -v`
Expected: PASS (2 passed). The round-trip test takes a few seconds (it runs a real small training job with CV/tuning disabled).

- [ ] **Step 7: Run the full backend test suite to check for regressions**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: all tests pass (no regressions in `test_pipeline_smoke.py` or elsewhere).

- [ ] **Step 8: Commit**

```bash
git add src/api/server.py config/models.yaml src/llm/client.py tests/test_api_chat.py
git commit -m "feat: add AI assistant chat API endpoint"
```

---

### Task 4: Frontend chat tab

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/app.js`
- Modify: `frontend/styles.css`

**Interfaces:**
- Consumes: `GET /api/runs/{id}` response fields `chat_history: [{role, content, timestamp}]`, `suggested_questions: string[]`, `status` (from Task 3); `POST /api/runs/{id}/chat` (`{question}` → `{answer}`).

- [ ] **Step 1: Add the tab button and panel to `frontend/index.html`**

Find:
```html
        <div class="tab-bar" role="tablist">
          <button class="tab-btn active" id="tab-report-btn" type="button" role="tab" aria-selected="true">Report</button>
          <button class="tab-btn" id="tab-test-btn" type="button" role="tab" aria-selected="false">Test the model</button>
        </div>
```

Replace with:
```html
        <div class="tab-bar" role="tablist">
          <button class="tab-btn active" id="tab-report-btn" type="button" role="tab" aria-selected="true">Report</button>
          <button class="tab-btn" id="tab-test-btn" type="button" role="tab" aria-selected="false">Test the model</button>
          <button class="tab-btn" id="tab-chat-btn" type="button" role="tab" aria-selected="false">Ask AI</button>
        </div>
```

Find:
```html
        <div id="tab-test-panel" class="hidden" role="tabpanel">
          <p class="muted small">Runs the saved model locally against values you enter here; nothing leaves this machine.</p>
          <form id="predict-form" class="predict-grid"></form>
          <div class="btn-row">
            <button type="submit" form="predict-form" class="btn primary">Predict</button>
          </div>
          <div id="predict-result" class="predict-result hidden"></div>
        </div>
      </div>
```

Replace with:
```html
        <div id="tab-test-panel" class="hidden" role="tabpanel">
          <p class="muted small">Runs the saved model locally against values you enter here; nothing leaves this machine.</p>
          <form id="predict-form" class="predict-grid"></form>
          <div class="btn-row">
            <button type="submit" form="predict-form" class="btn primary">Predict</button>
          </div>
          <div id="predict-result" class="predict-result hidden"></div>
        </div>

        <div id="tab-chat-panel" class="hidden" role="tabpanel">
          <p class="muted small" id="chat-placeholder">Available once your model is trained.</p>
          <div class="chat-thread hidden" id="chat-thread"></div>
          <div class="chat-suggestions hidden" id="chat-suggestions"></div>
          <form id="chat-form" class="chat-form hidden">
            <input type="text" id="chat-input" placeholder="Ask about this run's results…" autocomplete="off" />
            <button type="submit" class="btn primary">Send</button>
          </form>
        </div>
      </div>
```

- [ ] **Step 2: Extend `switchTab` to a three-way toggle in `frontend/app.js`**

Find:
```javascript
function switchTab(name) {
  const isReport = name === "report";
  $("tab-report-btn").classList.toggle("active", isReport);
  $("tab-report-btn").setAttribute("aria-selected", String(isReport));
  $("tab-test-btn").classList.toggle("active", !isReport);
  $("tab-test-btn").setAttribute("aria-selected", String(!isReport));
  $("tab-report-panel").classList.toggle("hidden", !isReport);
  $("tab-test-panel").classList.toggle("hidden", isReport);
  if (!isReport && lastRun) loadPredictTab(lastRun);
}
$("tab-report-btn").addEventListener("click", () => switchTab("report"));
$("tab-test-btn").addEventListener("click", () => switchTab("test"));
```

Replace with:
```javascript
function switchTab(name) {
  for (const tab of ["report", "test", "chat"]) {
    const isActive = tab === name;
    $(`tab-${tab}-btn`).classList.toggle("active", isActive);
    $(`tab-${tab}-btn`).setAttribute("aria-selected", String(isActive));
    $(`tab-${tab}-panel`).classList.toggle("hidden", !isActive);
  }
  if (name === "test" && lastRun) loadPredictTab(lastRun);
}
$("tab-report-btn").addEventListener("click", () => switchTab("report"));
$("tab-test-btn").addEventListener("click", () => switchTab("test"));
$("tab-chat-btn").addEventListener("click", () => switchTab("chat"));
```

- [ ] **Step 3: Add `renderChat` and the submit handler to `frontend/app.js`**

Find:
```javascript
/* ================= feature importance ================= */
```

Insert immediately before it:
```javascript
/* ================= AI assistant chat ================= */

function renderChat(run) {
  const ready = ["completed", "failed"].includes(run.status);
  $("chat-placeholder").classList.toggle("hidden", ready);
  $("chat-thread").classList.toggle("hidden", !ready);
  $("chat-suggestions").classList.toggle("hidden", !ready);
  $("chat-form").classList.toggle("hidden", !ready);
  if (!ready) return;

  const history = run.chat_history || [];
  $("chat-thread").innerHTML = history.length
    ? history
        .map(
          (m) => `
      <div class="chat-msg chat-${m.role}">
        <span class="chat-role">${m.role === "user" ? "You" : "Assistant"}</span>
        <p>${escapeHtml(m.content)}</p>
      </div>`
        )
        .join("")
    : `<p class="muted small">Ask anything about this run's data, decisions, or results.</p>`;
  $("chat-thread").scrollTop = $("chat-thread").scrollHeight;

  const suggestions = run.suggested_questions || [];
  $("chat-suggestions").innerHTML = suggestions
    .map((q) => `<button type="button" class="suggestion-chip">${escapeHtml(q)}</button>`)
    .join("");
  $("chat-suggestions").querySelectorAll(".suggestion-chip").forEach((chip, i) => {
    chip.addEventListener("click", () => {
      $("chat-input").value = suggestions[i];
      $("chat-input").focus();
    });
  });
}

$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("chat-input");
  const question = input.value.trim();
  if (!question || !currentRunId) return;
  input.value = "";
  const submitBtn = e.target.querySelector("button[type=submit]");
  submitBtn.disabled = true;
  try {
    const res = await fetch(`/api/runs/${currentRunId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    await poll();
  } catch (err) {
    alert("Could not get an answer: " + err.message);
  } finally {
    submitBtn.disabled = false;
  }
});

/* ================= feature importance ================= */
```

- [ ] **Step 4: Call `renderChat` from the main `render` function**

Find:
```javascript
  renderReport(run);
  renderCaveats(run);
  renderErrors(run);
}
```

Replace with:
```javascript
  renderReport(run);
  renderChat(run);
  renderCaveats(run);
  renderErrors(run);
}
```

- [ ] **Step 5: Reset the chat input when starting a new run**

Find:
```javascript
  predictFormLoadedFor = null;
  $("predict-result").classList.add("hidden");
  switchTab("report");
```

Replace with:
```javascript
  predictFormLoadedFor = null;
  $("predict-result").classList.add("hidden");
  $("chat-input").value = "";
  switchTab("report");
```

- [ ] **Step 6: Add chat styles to `frontend/styles.css`**

Find:
```css
.trace-details { margin-top: var(--sp-3); font-size: var(--text-sm); }
.trace-details summary { cursor: pointer; color: var(--text-secondary); font-weight: 650; }
.trace-body { font-family: var(--font-mono); font-size: var(--text-xs); white-space: pre-wrap; background: var(--bg-base); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: var(--sp-3); margin-top: var(--sp-2); max-height: 320px; overflow-y: auto; }
```

Insert immediately after it:
```css
/* ================= AI assistant chat tab ================= */
.chat-thread { display: flex; flex-direction: column; gap: var(--sp-2); max-height: 360px; overflow-y: auto; margin-bottom: var(--sp-3); }
.chat-msg { padding: 8px 12px; border-radius: var(--radius-sm); font-size: var(--text-sm); max-width: 85%; }
.chat-msg p { margin: 2px 0 0; white-space: pre-wrap; }
.chat-msg .chat-role { font-size: var(--text-xs); font-weight: 650; color: var(--text-secondary); }
.chat-msg.chat-user { align-self: flex-end; background: var(--accent-primary-soft); color: var(--text-primary); }
.chat-msg.chat-assistant { align-self: flex-start; background: var(--bg-surface-raised); border: 1px solid var(--border-subtle); }
.chat-suggestions { display: flex; flex-wrap: wrap; gap: var(--sp-2); margin-bottom: var(--sp-3); }
.suggestion-chip {
  font: inherit; font-size: var(--text-xs); font-weight: 650; padding: 4px 10px; border-radius: 999px;
  background: var(--bg-surface-raised); color: var(--text-secondary); border: 1px solid var(--border-subtle);
  cursor: pointer; transition: color 0.15s ease, border-color 0.15s ease;
}
.suggestion-chip:hover { color: var(--accent-primary); border-color: var(--accent-primary); }
.chat-form { display: flex; gap: var(--sp-2); }
.chat-form input[type="text"] {
  flex: 1; font: inherit; font-size: var(--text-sm); color: var(--text-primary);
  background: var(--bg-surface-raised); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 8px 12px;
}
```

- [ ] **Step 7: Syntax-check the JS**

Run: `node --check frontend/app.js`
Expected: no output (success)

- [ ] **Step 8: Manually verify in the browser**

Run: `AUTOML_MOCK_LLM=1 .venv/Scripts/python run_server.py` (background), then open `http://127.0.0.1:8000`.
- Upload a small CSV, confirm the task spec, approve the feature plan, wait for the run to complete.
- Click the "Ask AI" tab: before completion it should show the placeholder text; after completion it should show the message thread placeholder, suggestion chips, and an enabled input.
- Click a suggestion chip — it should populate the input (not auto-submit). Submit it — the mock answer should appear in the thread, and a second question should show both exchanges after a refresh.
- Confirm dark mode still reads correctly (toggle the theme switch).

- [ ] **Step 9: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: add Ask AI chat tab to the frontend"
```

---

### Task 5: Full regression pass

**Files:** none (verification only)

- [ ] **Step 1: Run the full backend test suite**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: all tests pass, no regressions.

- [ ] **Step 2: Update the design spec's status if anything changed during implementation**

If any implementation detail diverged from `docs/superpowers/specs/2026-07-05-ai-assistant-chat-design.md` (e.g. a different file for `_chat_context`), update the spec to match reality.

- [ ] **Step 3: Commit any final cleanup**

```bash
git add -A
git commit -m "chore: final cleanup for AI assistant chat feature"
```

(Skip this commit if there is nothing to add.)
