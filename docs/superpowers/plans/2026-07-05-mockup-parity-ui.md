# Mockup-Parity UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the frontend into a mockup-style home page and run dashboard (two-column with a right rail), add real-data-backed Data Quality and Class Distribution panels, and ship the AI Assistant chat as the dashboard's right-rail panel.

**Architecture:** Small additive backend changes (run-list scores, profile quality block, the already-planned chat endpoint) feed a restructured no-build vanilla JS frontend. Every widget renders only real pipeline data and hides itself when its data is absent. Chat backend comes verbatim from the existing approved plan `docs/superpowers/plans/2026-07-05-ai-assistant-chat.md` (Tasks 1–3); its Task 4 (tab UI) is superseded by Task 6 here (right-rail panel).

**Tech Stack:** FastAPI (`src/api/server.py`), pandas profiling (`src/profiling/profile.py`), vanilla JS/CSS frontend (no build step), pytest + FastAPI `TestClient`.

**Spec:** `docs/superpowers/specs/2026-07-05-mockup-parity-ui-design.md`

## Global Constraints

- No fabricated metrics or placeholder enterprise chrome (no cost savings, credits, resource usage, system health, notifications, monitoring). Every rendered number must come from the API.
- Raw data never enters an LLM context window; the quality block and class-distribution data are aggregates already computed by deterministic profiling.
- No new color families: reuse existing CSS tokens (`--accent-*`, `--cat-*`). Donuts re-tint on theme toggle (existing `applyTheme` pattern).
- Both themes (light/dark) must stay correct for every new element.
- All cards keep the existing hide-until-data pattern (`classList.add("hidden")` when data is absent).
- Commit messages: plain conventional commits, **no Co-Authored-By trailer**.
- Python: 3.11+, type hints on public functions. Frontend: no framework, no build step.
- Run tests with `.venv/Scripts/python -m pytest ...` from the worktree root.

---

### Task 1: `GET /api/runs` scores + `top_values` in profile columns

**Files:**
- Modify: `src/api/server.py` (functions `_profile_columns` ~line 138, `list_runs` ~line 301)
- Test: `tests/test_api_run_listing.py` (create)

**Interfaces:**
- Produces: `GET /api/runs` items gain `best_score: float | null` and `metric: str | null` (used by Task 4's home view). `GET /api/runs/{id}`'s `profile_columns[*]` gain `top_values: dict[str, int] | null` (used by Task 5's class-distribution panel). Internal helper `_best_score(entry: dict[str, Any]) -> float | None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api_run_listing.py`:

```python
"""GET /api/runs must expose best_score/metric so the home view can show
scores without fetching every run's detail, and _profile_columns must pass
top_values through for the class-distribution panel (see
docs/superpowers/specs/2026-07-05-mockup-parity-ui-design.md)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from src.api import server


def _entry(state: dict, status: str) -> dict:
    now = time.time()
    return {
        "state": state,
        "status": status,
        "events": [],
        "filename": "churn.csv",
        "created_at": now,
        "finished_at": now if status in ("completed", "failed") else None,
        "cancel_requested": False,
    }


def test_list_runs_includes_best_score_and_metric(monkeypatch):
    client = TestClient(server.app)
    state = {
        "use_case_description": "predict churn",
        "task_spec": {"metric": "f1"},
        "best_model": {"metrics": {"f1": 0.8421}},
    }
    monkeypatch.setitem(server._runs, "fake-run", _entry(state, "completed"))

    runs = client.get("/api/runs").json()
    fake = next(r for r in runs if r["run_id"] == "fake-run")
    assert fake["best_score"] == 0.8421
    assert fake["metric"] == "f1"


def test_list_runs_scores_null_before_completion(monkeypatch):
    client = TestClient(server.app)
    state = {"use_case_description": "predict churn"}
    monkeypatch.setitem(server._runs, "fake-run-2", _entry(state, "profiling"))

    runs = client.get("/api/runs").json()
    fake = next(r for r in runs if r["run_id"] == "fake-run-2")
    assert fake["best_score"] is None
    assert fake["metric"] is None


def test_profile_columns_include_top_values():
    state = {
        "profile": {
            "columns": {
                "churned": {
                    "dtype": "int64",
                    "null_rate": 0.0,
                    "n_unique": 2,
                    "is_pii": False,
                    "top_values": {"0": 90, "1": 10},
                },
                "tenure": {
                    "dtype": "float64",
                    "null_rate": 0.0,
                    "n_unique": 87,
                    "is_pii": False,
                },
            }
        }
    }
    cols = {c["name"]: c for c in server._profile_columns(state)}
    assert cols["churned"]["top_values"] == {"0": 90, "1": 10}
    assert cols["tenure"]["top_values"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_api_run_listing.py -v`
Expected: FAIL — `KeyError: 'best_score'` (and `KeyError: 'top_values'` for the third test).

- [ ] **Step 3: Implement**

In `src/api/server.py`, find:

```python
def _profile_columns(state: PipelineState) -> list[dict[str, Any]]:
    columns = state.get("profile", {}).get("columns", {})
    return [
        {
            "name": name,
            "dtype": info.get("dtype"),
            "null_rate": info.get("null_rate"),
            "n_unique": info.get("n_unique"),
            "is_pii": info.get("is_pii", False),
        }
        for name, info in columns.items()
    ]
```

Replace with:

```python
def _profile_columns(state: PipelineState) -> list[dict[str, Any]]:
    columns = state.get("profile", {}).get("columns", {})
    return [
        {
            "name": name,
            "dtype": info.get("dtype"),
            "null_rate": info.get("null_rate"),
            "n_unique": info.get("n_unique"),
            "is_pii": info.get("is_pii", False),
            # aggregate counts only (already PII-redacted upstream) — feeds
            # the class-distribution panel for the confirmed target column
            "top_values": info.get("top_values"),
        }
        for name, info in columns.items()
    ]


def _best_score(entry: dict[str, Any]) -> float | None:
    """Best model's score on the run's own success metric, for the run list."""
    state = entry["state"]
    metric = (state.get("task_spec") or {}).get("metric")
    metrics = (state.get("best_model") or {}).get("metrics") or {}
    if metric and metric in metrics:
        return round(float(metrics[metric]), 4)
    return None
```

Then find (inside `list_runs`):

```python
            {
                "run_id": run_id,
                "filename": entry["filename"],
                "status": entry["status"],
                "created_at": entry["created_at"],
                "description": entry["state"].get("use_case_description"),
            }
```

Replace with:

```python
            {
                "run_id": run_id,
                "filename": entry["filename"],
                "status": entry["status"],
                "created_at": entry["created_at"],
                "description": entry["state"].get("use_case_description"),
                "best_score": _best_score(entry),
                "metric": (entry["state"].get("task_spec") or {}).get("metric"),
            }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_api_run_listing.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/api/server.py tests/test_api_run_listing.py
git commit -m "feat: expose best_score/metric in run list and top_values in profile columns"
```

---

### Task 2: Deterministic data-quality block in the profile

**Files:**
- Modify: `src/profiling/profile.py` (end of `profile_dataset`, ~line 160)
- Modify: `src/api/server.py` (`_run_summary`'s `profile_summary` dict, ~line 256)
- Test: `tests/test_profile_quality.py` (create)

**Interfaces:**
- Produces: `profile_dataset(df)["quality"]` = `{"completeness": float, "duplicate_row_count": int, "duplicate_row_rate": float, "uniqueness": float, "overall": float}` (all 0–1 except the count). `GET /api/runs/{id}`'s `profile_summary` gains `quality` with the same shape (used by Task 5's stat card + quality panel).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_profile_quality.py`:

```python
"""profile_dataset must expose a deterministic quality block (completeness /
duplicates / uniqueness) for the dashboard's data-quality panel. Aggregates
only — no raw values (CLAUDE.md rule)."""

from __future__ import annotations

import pandas as pd

from src.profiling.profile import profile_dataset


def test_quality_block_reports_duplicates_and_completeness():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 1, None],  # null rate 0.2
            "b": ["x", "y", "z", "x", "y"],  # rows 0 and 3 duplicate each other
        }
    )
    quality = profile_dataset(df)["quality"]
    assert quality["duplicate_row_count"] == 1
    assert quality["duplicate_row_rate"] == 0.2
    assert quality["uniqueness"] == 0.8
    assert quality["completeness"] == 0.9  # 1 - mean null rate (0.2 + 0.0) / 2
    assert quality["overall"] == round((0.9 + 0.8) / 2, 4)


def test_quality_block_perfect_dataset():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    quality = profile_dataset(df)["quality"]
    assert quality["duplicate_row_count"] == 0
    assert quality["uniqueness"] == 1.0
    assert quality["completeness"] == 1.0
    assert quality["overall"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_profile_quality.py -v`
Expected: FAIL with `KeyError: 'quality'`

- [ ] **Step 3: Implement the quality block**

In `src/profiling/profile.py`, find (end of `profile_dataset`):

```python
    profile["sample_rows"] = redacted.head(MAX_SAMPLE_ROWS).to_dict(orient="records")
    return profile
```

Replace with:

```python
    # deterministic quality aggregates for the dashboard — shape-level
    # statistics only, no raw values, so safe alongside the rest of the profile
    null_rates = [float(df[col].isna().mean()) for col in df.columns]
    completeness = 1.0 - (sum(null_rates) / len(null_rates) if null_rates else 0.0)
    duplicate_row_count = int(df.duplicated().sum())
    duplicate_row_rate = duplicate_row_count / len(df) if len(df) else 0.0
    uniqueness = 1.0 - duplicate_row_rate
    profile["quality"] = {
        "completeness": round(completeness, 4),
        "duplicate_row_count": duplicate_row_count,
        "duplicate_row_rate": round(duplicate_row_rate, 4),
        "uniqueness": round(uniqueness, 4),
        "overall": round((completeness + uniqueness) / 2, 4),
    }

    profile["sample_rows"] = redacted.head(MAX_SAMPLE_ROWS).to_dict(orient="records")
    return profile
```

- [ ] **Step 4: Expose it in `_run_summary`**

In `src/api/server.py`, find:

```python
            "profile_summary": {
                "row_count": state.get("profile", {}).get("row_count"),
                "column_count": state.get("profile", {}).get("column_count"),
                "pii_columns_detected": state.get("profile", {})
                .get("pii_report", {})
                .get("pii_columns_detected"),
            },
```

Replace with:

```python
            "profile_summary": {
                "row_count": state.get("profile", {}).get("row_count"),
                "column_count": state.get("profile", {}).get("column_count"),
                "pii_columns_detected": state.get("profile", {})
                .get("pii_report", {})
                .get("pii_columns_detected"),
                "quality": state.get("profile", {}).get("quality"),
            },
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_profile_quality.py tests/test_profile_wide.py -v`
Expected: PASS (quality tests plus no regression in the wide-profile tests)

- [ ] **Step 6: Commit**

```bash
git add src/profiling/profile.py src/api/server.py tests/test_profile_quality.py
git commit -m "feat: add deterministic data-quality block to dataset profile"
```

---

### Task 3: AI Assistant chat backend (existing approved plan, Tasks 1–3)

**Files:** exactly those listed in `docs/superpowers/plans/2026-07-05-ai-assistant-chat.md` Tasks 1–3:
- Modify: `src/insights/auto_insights.py`, `src/api/server.py`, `config/models.yaml`, `src/llm/client.py`
- Create: `src/agents/prompts/chat.md`, `src/agents/chat_node.py`
- Test: `tests/test_chat_suggestions.py`, `tests/test_chat_node.py`, `tests/test_api_chat.py` (create all three)

**Interfaces:**
- Produces (consumed by Task 6's panel): `POST /api/runs/{run_id}/chat` with body `{"question": str}` → `{"answer": str}`, 409 unless `status in ("completed", "failed")`; `GET /api/runs/{run_id}` gains `chat_history: [{"role": "user"|"assistant", "content": str, "timestamp": float}]` and `suggested_questions: string[]`.

- [ ] **Step 1: Execute Tasks 1–3 of `docs/superpowers/plans/2026-07-05-ai-assistant-chat.md` exactly as written**

That plan contains complete code, tests, and expected outputs for each step — follow its checkboxes for Task 1 (suggested-questions helper), Task 2 (chat prompt + `answer_chat_question`), and Task 3 (API endpoint, chat state, mock-mode wiring), including its per-task commits (drop any Co-Authored-By trailer from its commit templates).

**Do NOT execute its Task 4 or Task 5** — the tab UI is superseded by Task 6 of this plan, and the regression pass happens in Task 7 here.

Note: this plan's Tasks 1–2 already modified `src/api/server.py`, but they touched `_profile_columns`/`list_runs`/`profile_summary` only — every find/replace anchor in the chat plan's Task 3 still matches verbatim.

- [ ] **Step 2: Verify the chat backend test suite passes**

Run: `.venv/Scripts/python -m pytest tests/test_chat_suggestions.py tests/test_chat_node.py tests/test_api_chat.py -v`
Expected: PASS (all tests)

---

### Task 4: Home view (hero + trust chips + stats + recent projects + feature grid)

**Files:**
- Modify: `frontend/index.html` (the `#intake-view` block)
- Modify: `frontend/app.js` (`loadRecentRuns`, new `renderHome`/`renderHomePipeline`)
- Modify: `frontend/styles.css` (append home styles)

**Interfaces:**
- Consumes: `GET /api/runs` items `{run_id, filename, status, created_at, description, best_score, metric}` (Task 1); `GET /api/runs/{id}` fields `stages_done`, `stage_timeline`, `filename` (existing); existing JS helpers `$`, `escapeHtml`, `relativeTime`, `formatDuration`, `openRun`, `ICONS`, `STAGES`.
- Produces: nothing consumed by later tasks (self-contained view).

- [ ] **Step 1: Replace the intake view in `frontend/index.html`**

Find (the whole current intake `<main>`):

```html
    <!-- ============ intake ============ -->
    <main class="content" id="intake-view">
      <section class="card intake-card">
        <div class="intake-hero">
          <h2>What do you want to predict or solve?</h2>
          <p class="muted">Describe your goal in plain language, then attach the dataset. Nothing runs until you've confirmed what we understood.</p>
        </div>
        <form id="new-run-form">
          <label class="usecase-field field">
            <span class="visually-hidden">Prediction goal</span>
            <textarea id="description" class="usecase-input" rows="2" required
              placeholder="e.g. Predict which customers will churn next month"></textarea>
          </label>
          <label class="dropzone" id="dropzone">
            <input type="file" id="file-input" accept=".csv" hidden />
            <span id="dropzone-label"><strong>Drop a CSV here</strong> or click to browse</span>
          </label>
          <div class="estimate-row" id="estimate-row"></div>
          <div class="btn-row">
            <button type="submit" class="btn primary" id="submit-btn" disabled>Run pipeline</button>
          </div>
        </form>
      </section>
    </main>
```

Replace with:

```html
    <!-- ============ home / intake ============ -->
    <main class="content home-content" id="intake-view">
      <section class="home-hero">
        <div class="hero-eyebrow" aria-hidden="true">
          <span>AI-Powered</span><span class="hero-eyebrow-dot"></span>
          <span>Agentic</span><span class="hero-eyebrow-dot"></span>
          <span>Automated</span>
        </div>
        <h2 class="hero-title">Build Better Models.<br /><span class="hero-accent">Automatically.</span></h2>
        <p class="hero-sub">Describe your goal in plain language and attach a dataset. LLM agents plan and run the full pipeline — profiling, feature engineering, training, evaluation — and explain every decision. Nothing runs until you've confirmed what they understood.</p>
      </section>

      <section class="card intake-card">
        <div class="intake-hero">
          <h2>What do you want to predict or solve?</h2>
          <p class="muted">Nothing runs until you've confirmed what we understood.</p>
        </div>
        <form id="new-run-form">
          <label class="usecase-field field">
            <span class="visually-hidden">Prediction goal</span>
            <textarea id="description" class="usecase-input" rows="2" required
              placeholder="e.g. Predict which customers will churn next month"></textarea>
          </label>
          <label class="dropzone" id="dropzone">
            <input type="file" id="file-input" accept=".csv" hidden />
            <span id="dropzone-label"><strong>Drop a CSV here</strong> or click to browse</span>
          </label>
          <div class="estimate-row" id="estimate-row"></div>
          <div class="btn-row">
            <button type="submit" class="btn primary" id="submit-btn" disabled>Run pipeline</button>
          </div>
        </form>
      </section>

      <div class="trust-row" role="list">
        <span class="trust-chip" role="listitem"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/></svg>No coding required</span>
        <span class="trust-chip" role="listitem"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/></svg>Runs locally</span>
        <span class="trust-chip" role="listitem"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-3.6 8-10V5l-8-3-8 3v7c0 6.4 8 10 8 10Z"/></svg>Raw data never reaches the LLM</span>
        <span class="trust-chip" role="listitem"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z"/><path d="M14 2v6h6"/></svg>Full audit trace</span>
      </div>

      <section class="home-stats hidden" id="home-stats"></section>

      <section class="home-band hidden" id="home-band">
        <div class="card home-projects">
          <div class="card-head"><h3>Recent Projects</h3></div>
          <div class="home-projects-list" id="home-projects-list"></div>
        </div>
        <div class="card home-pipeline hidden" id="home-pipeline">
          <div class="card-head"><h3>Pipeline in Progress</h3><span class="muted small" id="home-pipeline-sub"></span></div>
          <ol class="mini-rail" id="home-pipeline-rail"></ol>
        </div>
      </section>

      <section class="home-features">
        <h2 class="home-features-title">An End-to-End AutoML Pipeline</h2>
        <p class="muted home-features-sub">From data understanding to a downloadable model — automated, explained, and checkpointed by you.</p>
        <div class="feature-grid">
          <div class="feature-card">
            <span class="feature-icon violet"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg></span>
            <h3>Smart Data Understanding</h3>
            <p>Deterministic profiling, PII redaction, and leakage checks before any AI-facing step. Raw rows never reach the LLM.</p>
          </div>
          <div class="feature-card">
            <span class="feature-icon green"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 21v-7m0-4V3m8 18v-9m0-4V3m8 18v-5m0-4V3M1 14h6M9 8h6m2 8h6"/></svg></span>
            <h3>Intelligent Feature Engineering</h3>
            <p>EDA-grounded transformation plans — imputation, encoding, scaling — that you review and approve before they run.</p>
          </div>
          <div class="feature-card">
            <span class="feature-icon blue"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/></svg></span>
            <h3>Model Selection &amp; Tuning</h3>
            <p>Candidate models suited to your task, k-fold cross-validation, and Optuna hyperparameter search per candidate.</p>
          </div>
          <div class="feature-card">
            <span class="feature-icon amber"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18h6M10 22h4M12 2a6 6 0 0 0-4 10.5c.6.6 1 1.4 1 2.5h6c0-1.1.4-1.9 1-2.5A6 6 0 0 0 12 2Z"/></svg></span>
            <h3>Explainable Insights</h3>
            <p>Feature importance, auto insights, plain-language reports, and explicit caveats for every run.</p>
          </div>
          <div class="feature-card">
            <span class="feature-icon violet"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="9" cy="8" r="4"/><path d="M2 21c0-3.9 3.1-7 7-7 1.4 0 2.7.4 3.8 1.1"/><path d="m15 18 2 2 4-4"/></svg></span>
            <h3>Human Checkpoints</h3>
            <p>The pipeline pauses for your confirmation of the task spec and your approval of the feature plan — you stay in control.</p>
          </div>
        </div>
      </section>
    </main>
```

- [ ] **Step 2: Wire `renderHome` into `frontend/app.js`**

Find:

```javascript
async function loadRecentRuns() {
  const box = $("nav-runs");
  try {
    const runs = await (await fetch("/api/runs")).json();
    if (!runs.length) {
      box.innerHTML = `<span class="nav-runs-empty">No runs yet</span>`;
      return;
    }
    box.innerHTML = "";
    for (const run of runs.slice(0, 6)) {
      const el = document.createElement("button");
      el.type = "button";
      el.className = "nav-run";
      el.title = `${run.filename} — ${run.description || ""}`;
      el.innerHTML = `<span class="dot ${run.status}"></span>${escapeHtml(run.filename)}`;
      el.onclick = () => openRun(run.run_id);
      box.appendChild(el);
    }
  } catch {
    box.innerHTML = `<span class="nav-runs-empty">No runs yet</span>`;
  }
}
```

Replace with:

```javascript
async function loadRecentRuns() {
  const box = $("nav-runs");
  let runs = [];
  try {
    runs = await (await fetch("/api/runs")).json();
  } catch {
    box.innerHTML = `<span class="nav-runs-empty">No runs yet</span>`;
    renderHome([]);
    return;
  }
  if (!runs.length) {
    box.innerHTML = `<span class="nav-runs-empty">No runs yet</span>`;
  } else {
    box.innerHTML = "";
    for (const run of runs.slice(0, 6)) {
      const el = document.createElement("button");
      el.type = "button";
      el.className = "nav-run";
      el.title = `${run.filename} — ${run.description || ""}`;
      el.innerHTML = `<span class="dot ${run.status}"></span>${escapeHtml(run.filename)}`;
      el.onclick = () => openRun(run.run_id);
      box.appendChild(el);
    }
  }
  renderHome(runs);
}

/* ================= home view ================= */

const ACTIVE_STATUSES = ["profiling", "running", "awaiting_confirmation", "awaiting_feature_approval"];
/* cross-run "best" is only meaningful for higher-is-better metrics */
const HIGHER_IS_BETTER = new Set(["f1", "accuracy", "roc_auc", "r2"]);
let homePipelineRunId = null;

function renderHome(runs) {
  const active = runs.filter((r) => ACTIVE_STATUSES.includes(r.status));
  let bestRun = null;
  for (const r of runs) {
    if (r.status !== "completed" || r.best_score == null || !HIGHER_IS_BETTER.has(r.metric)) continue;
    if (!bestRun || r.best_score > bestRun.best_score) bestRun = r;
  }

  const statsBox = $("home-stats");
  statsBox.classList.toggle("hidden", !runs.length);
  if (runs.length) {
    statsBox.innerHTML = `
      <div class="home-stat"><div class="home-stat-value">${runs.length}</div><div class="home-stat-label">Total experiments</div></div>
      <div class="home-stat"><div class="home-stat-value">${bestRun ? bestRun.best_score.toFixed(3) : "—"}</div><div class="home-stat-label">${bestRun ? `Best ${escapeHtml(bestRun.metric)} score` : "Best score"}</div></div>
      <div class="home-stat"><div class="home-stat-value">${active.length}</div><div class="home-stat-label">Active run${active.length === 1 ? "" : "s"}</div></div>`;
  }

  $("home-band").classList.toggle("hidden", !runs.length);
  if (runs.length) {
    $("home-projects-list").innerHTML = runs
      .slice(0, 5)
      .map(
        (r) => `
      <button type="button" class="home-project" data-run-id="${r.run_id}">
        <span class="home-project-main">
          <span class="home-project-name">${escapeHtml(r.filename)}</span>
          <span class="home-project-desc">${escapeHtml(r.description || "")}</span>
        </span>
        <span class="home-project-meta">
          ${r.best_score != null ? `<span class="home-project-score mono">${escapeHtml(r.metric)}: ${r.best_score.toFixed(3)}</span>` : ""}
          <span class="status-badge ${r.status}">${r.status.replaceAll("_", " ")}</span>
          <span class="home-project-time">${relativeTime(r.created_at)}</span>
        </span>
      </button>`
      )
      .join("");
    $("home-projects-list").querySelectorAll(".home-project").forEach((el) => {
      el.addEventListener("click", () => openRun(el.dataset.runId));
    });
  }
  renderHomePipeline(active[0] || null);
}

async function renderHomePipeline(activeRun) {
  const card = $("home-pipeline");
  if (!activeRun) {
    card.classList.add("hidden");
    homePipelineRunId = null;
    return;
  }
  homePipelineRunId = activeRun.run_id;
  let run;
  try {
    run = await (await fetch(`/api/runs/${activeRun.run_id}`)).json();
  } catch {
    return;
  }
  if (homePipelineRunId !== activeRun.run_id) return; // superseded by a newer refresh
  card.classList.remove("hidden");
  $("home-pipeline-sub").textContent = run.filename;

  const done = new Set(run.stages_done || []);
  const durations = {};
  for (const rec of run.stage_timeline || []) durations[rec.node] = rec.duration_seconds;
  let activeAssigned = false;
  $("home-pipeline-rail").innerHTML = STAGES.map((stage) => {
    const stageDone = stage.node === "poll_training" ? done.has("evaluate") : done.has(stage.node);
    let cls = "pending";
    if (stageDone) cls = "done";
    else if (!activeAssigned) {
      cls = "active";
      activeAssigned = true;
    }
    const duration = durations[stage.node];
    return `<li class="mini-stage ${cls}">
      <span class="mini-stage-dot">${cls === "done" ? ICONS.check : ""}</span>
      <span class="mini-stage-label">${stage.label}</span>
      <span class="mini-stage-time mono">${cls === "done" && duration != null ? formatDuration(duration) : cls === "active" ? "running" : ""}</span>
    </li>`;
  }).join("");
}
```

- [ ] **Step 3: Append home styles to `frontend/styles.css`**

Append at the end of the file (before the responsive-shell media query is fine, but end-of-file keeps it simple — the media query at the end only touches `.app`/`.sidebar`/`.page-header`/`.content` so ordering does not conflict):

```css
/* ================= home view ================= */

.home-content { max-width: 1100px; margin: 0 auto; width: 100%; }

.home-hero { text-align: center; padding: var(--sp-4) 0 var(--sp-2); }
.hero-eyebrow {
  display: inline-flex; align-items: center; gap: 10px;
  font-size: var(--text-xs); font-weight: 650; color: var(--accent-primary);
  background: var(--accent-primary-soft); border-radius: 999px; padding: 5px 14px;
  margin-bottom: var(--sp-3);
}
.hero-eyebrow-dot { width: 4px; height: 4px; border-radius: 50%; background: currentColor; opacity: 0.6; }
.hero-title {
  font-size: clamp(30px, 4.5vw, 46px); line-height: 1.08; letter-spacing: -0.02em;
  font-weight: 800;
}
.hero-accent {
  background: linear-gradient(90deg, var(--accent-primary), var(--accent-primary-strong));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
:root[data-theme="dark"] .hero-accent {
  background: linear-gradient(90deg, var(--accent-primary), #c4b5fd);
  -webkit-background-clip: text; background-clip: text;
}
.hero-sub {
  color: var(--text-secondary); font-size: var(--text-sm);
  max-width: 560px; margin: var(--sp-3) auto 0;
}

.trust-row { display: flex; flex-wrap: wrap; gap: var(--sp-2); justify-content: center; }
.trust-chip {
  display: inline-flex; align-items: center; gap: 7px;
  font-size: var(--text-xs); font-weight: 600; color: var(--text-secondary);
  background: var(--bg-surface); border: 1px solid var(--border-subtle);
  border-radius: 999px; padding: 6px 13px;
}
.trust-chip .icon { color: var(--accent-primary); }

.home-stats {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--sp-3);
}
.home-stat {
  background: var(--bg-surface); border: 1px solid var(--border-subtle);
  border-radius: var(--radius); box-shadow: var(--shadow);
  padding: var(--sp-3); text-align: center;
}
.home-stat-value { font-family: var(--font-display); font-size: var(--text-xl); font-weight: 750; }
.home-stat-label { font-size: var(--text-xs); color: var(--text-secondary); font-weight: 600; }
@media (max-width: 640px) { .home-stats { grid-template-columns: 1fr; } }

.home-band { display: grid; grid-template-columns: 1.5fr 1fr; gap: var(--sp-3); align-items: start; }
@media (max-width: 900px) { .home-band { grid-template-columns: 1fr; } }

.home-projects-list { display: grid; gap: 6px; }
.home-project {
  display: flex; justify-content: space-between; align-items: center; gap: var(--sp-2);
  font: inherit; text-align: left; width: 100%; cursor: pointer;
  background: var(--bg-surface-raised); border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm); padding: 10px 12px; color: var(--text-primary);
}
.home-project:hover { border-color: var(--accent-primary); }
.home-project-main { min-width: 0; }
.home-project-name { display: block; font-weight: 650; font-size: var(--text-sm); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.home-project-desc { display: block; font-size: var(--text-xs); color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 320px; }
.home-project-meta { display: flex; align-items: center; gap: var(--sp-2); flex-shrink: 0; flex-wrap: wrap; justify-content: flex-end; }
.home-project-score { font-size: var(--text-xs); color: var(--accent-primary); font-weight: 650; }
.home-project-time { font-size: var(--text-xs); color: var(--text-secondary); font-family: var(--font-mono); }

.mini-rail { list-style: none; padding: 0; display: grid; gap: 2px; }
.mini-stage { display: flex; align-items: center; gap: 10px; padding: 6px 0; font-size: var(--text-sm); }
.mini-stage-dot {
  width: 22px; height: 22px; border-radius: 50%; flex-shrink: 0;
  border: 2px solid var(--border-subtle); background: var(--bg-surface);
  display: inline-flex; align-items: center; justify-content: center;
  color: var(--accent-success);
}
.mini-stage-dot .icon { width: 12px; height: 12px; }
.mini-stage.done .mini-stage-dot { border-color: var(--accent-success); }
.mini-stage.active .mini-stage-dot { border-color: var(--accent-primary); animation: stage-pulse 1.4s ease-in-out infinite; }
.mini-stage-label { color: var(--text-secondary); }
.mini-stage.done .mini-stage-label, .mini-stage.active .mini-stage-label { color: var(--text-primary); font-weight: 600; }
.mini-stage.active .mini-stage-label { color: var(--accent-primary); }
.mini-stage-time { margin-left: auto; font-size: var(--text-xs); color: var(--text-secondary); }

.home-features { text-align: center; padding-top: var(--sp-3); }
.home-features-title { font-size: var(--text-lg); }
.home-features-sub { font-size: var(--text-sm); margin: 4px 0 var(--sp-3); }
.feature-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--sp-3); text-align: left;
}
.feature-card {
  background: var(--bg-surface); border: 1px solid var(--border-subtle);
  border-radius: var(--radius); box-shadow: var(--shadow); padding: var(--sp-3);
}
.feature-card h3 { font-size: var(--text-sm); margin: var(--sp-2) 0 4px; }
.feature-card p { font-size: var(--text-xs); color: var(--text-secondary); }
.feature-icon {
  width: 38px; height: 38px; border-radius: 10px;
  display: inline-flex; align-items: center; justify-content: center;
}
.feature-icon .icon { width: 19px; height: 19px; }
.feature-icon.violet { background: var(--accent-primary-soft); color: var(--accent-primary); }
.feature-icon.green { background: var(--accent-success-soft); color: var(--accent-success); }
.feature-icon.amber { background: var(--accent-warning-soft); color: var(--accent-warning); }
.feature-icon.blue { background: var(--bg-surface-raised); color: var(--cat-2); border: 1px solid var(--border-subtle); }
```

- [ ] **Step 4: Syntax-check the JS**

Run: `node --check frontend/app.js`
Expected: no output (success)

- [ ] **Step 5: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: redesign intake view as a mockup-style home page"
```

---

### Task 5: Dashboard two-column layout + Data Quality and Class Distribution panels

**Files:**
- Modify: `frontend/index.html` (the `#run-view` block)
- Modify: `frontend/app.js` (`render`, `renderStatCards`, `applyTheme`; new `renderQuality`, `renderClassDistribution`)
- Modify: `frontend/styles.css` (run layout + quality styles; extend the `#donut` rule)

**Interfaces:**
- Consumes: `profile_summary.quality` (Task 2), `profile_columns[*].top_values` (Task 1), existing run fields (`task_spec`, `stages_done`, `resampling_plan`).
- Produces: `#assistant-card` placeholder markup filled in by Task 6; `renderClassDistribution(run)` / `renderQuality(run)` called from `render`.

- [ ] **Step 1: Restructure `frontend/index.html`'s run view**

Four targeted edits:

**(a)** Find:

```html
    <main class="content hidden" id="run-view">

      <!-- stat cards -->
      <section class="stat-row" id="stat-cards"></section>

      <!-- live pipeline -->
```

Replace with:

```html
    <main class="content hidden" id="run-view">

      <!-- stat cards -->
      <section class="stat-row" id="stat-cards"></section>

      <div class="run-layout">
      <div class="run-main">

      <!-- live pipeline -->
```

**(b)** Add the two new panels and drop the activity card from the dashboard grid. Find:

```html
        <div class="card hidden" id="dataset-card">
          <div class="card-head"><h3>Dataset summary</h3><span class="muted small" id="dataset-sub"></span></div>
          <div class="donut-wrap">
            <svg id="donut" viewBox="0 0 120 120" role="img" aria-label="Feature type breakdown"></svg>
            <div class="donut-center" id="donut-center"></div>
            <ul class="donut-legend" id="donut-legend"></ul>
          </div>
          <div class="chips" id="dataset-chips"></div>
        </div>
```

Replace with:

```html
        <div class="card hidden" id="dataset-card">
          <div class="card-head"><h3>Dataset summary</h3><span class="muted small" id="dataset-sub"></span></div>
          <div class="donut-wrap">
            <svg id="donut" viewBox="0 0 120 120" role="img" aria-label="Feature type breakdown"></svg>
            <div class="donut-center" id="donut-center"></div>
            <ul class="donut-legend" id="donut-legend"></ul>
          </div>
          <div class="chips" id="dataset-chips"></div>
        </div>

        <div class="card hidden" id="classdist-card">
          <div class="card-head"><h3>Class distribution</h3><span class="muted small" id="classdist-sub"></span></div>
          <div class="donut-wrap">
            <svg id="classdist-donut" viewBox="0 0 120 120" role="img" aria-label="Target class distribution"></svg>
            <div class="donut-center" id="classdist-center"></div>
            <ul class="donut-legend" id="classdist-legend"></ul>
          </div>
          <div class="chips" id="classdist-chips"></div>
        </div>

        <div class="card hidden" id="quality-card">
          <div class="card-head"><h3>Data quality overview</h3><span class="muted small" id="quality-sub"></span></div>
          <div class="quality-wrap">
            <div class="quality-ring" id="quality-ring"></div>
            <div class="quality-bars" id="quality-bars"></div>
          </div>
        </div>
```

**(c)** Find:

```html
        <div class="card hidden" id="activity-card">
          <div class="card-head"><h3>Recent activity</h3></div>
          <ul class="activity-list" id="activity-list"></ul>
        </div>
      </section>
```

Replace with:

```html
      </section>
```

**(d)** Close the main column and add the right rail. Find:

```html
      <!-- errors -->
      <div class="card error-card hidden" id="error-card">
        <h3>Issues encountered</h3>
        <ul class="callout-list" id="error-list"></ul>
      </div>
    </main>
```

Replace with:

```html
      <!-- errors -->
      <div class="card error-card hidden" id="error-card">
        <h3>Issues encountered</h3>
        <ul class="callout-list" id="error-list"></ul>
      </div>

      </div><!-- /run-main -->

      <aside class="run-rail">
        <div class="card assistant-card" id="assistant-card">
          <div class="card-head">
            <h3><svg class="icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2 14 9 21 12 14 15 12 22 10 15 3 12 10 9Z"/></svg> AI Assistant</h3>
          </div>
          <p class="muted small" id="chat-placeholder">Available once your model is trained.</p>
          <div class="chat-thread hidden" id="chat-thread"></div>
          <div class="chat-suggestions hidden" id="chat-suggestions"></div>
          <form id="chat-form" class="chat-form hidden">
            <input type="text" id="chat-input" placeholder="Ask about this run…" autocomplete="off" />
            <button type="submit" class="btn primary" id="chat-send-btn" aria-label="Send question">
              <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m22 2-11 11"/><path d="M22 2 15 22l-4-9-9-4 20-7Z"/></svg>
            </button>
          </form>
          <p class="chat-error hidden" id="chat-error" role="alert"></p>
        </div>

        <div class="card hidden" id="activity-card">
          <div class="card-head"><h3>Recent activity</h3></div>
          <ul class="activity-list" id="activity-list"></ul>
        </div>
      </aside>
      </div><!-- /run-layout -->
    </main>
```

(The chat elements are inert until Task 6 wires them up; they are all `hidden` by default except the placeholder, which is honest at any pipeline stage.)

- [ ] **Step 2: Add the new renderers to `frontend/app.js`**

Find (end of `renderDatasetSummary`):

```javascript
  $("dataset-chips").innerHTML = `
    <span class="chip detected">${ICONS.check} worst null rate: ${(worstNull * 100).toFixed(1)}%</span>
    ${targetCol ? `<span class="chip detected">${ICONS.check} target: <span class="mono">${escapeHtml(targetCol)}</span></span>` : ""}
    ${piiCount ? `<span class="chip flagged" title="Redacted from every AI-facing step">${ICONS.warning} ${piiCount} PII column(s)</span>` : ""}`;
}
```

Replace with:

```javascript
  $("dataset-chips").innerHTML = `
    <span class="chip detected">${ICONS.check} worst null rate: ${(worstNull * 100).toFixed(1)}%</span>
    ${targetCol ? `<span class="chip detected">${ICONS.check} target: <span class="mono">${escapeHtml(targetCol)}</span></span>` : ""}
    ${piiCount ? `<span class="chip flagged" title="Redacted from every AI-facing step">${ICONS.warning} ${piiCount} PII column(s)</span>` : ""}`;
}

/* ================= class distribution (classification targets) ================= */

function renderClassDistribution(run) {
  const card = $("classdist-card");
  const spec = run.task_spec || {};
  const confirmed = (run.stages_done || []).includes("confirm");
  const target = (run.profile_columns || []).find((c) => c.name === spec.target_column);
  const entries = target && target.top_values ? Object.entries(target.top_values).sort((a, b) => b[1] - a[1]) : [];
  if (spec.task_type !== "classification" || !confirmed || entries.length < 2) {
    card.classList.add("hidden");
    return;
  }
  card.classList.remove("hidden");
  $("classdist-sub").textContent = `target: ${spec.target_column}`;

  const total = entries.reduce((acc, [, n]) => acc + n, 0);
  const styles = getComputedStyle(document.documentElement);
  const palette = DONUT_KEYS.map((k) => styles.getPropertyValue(k).trim());
  const R = 44, C = 2 * Math.PI * R;
  const gapPx = entries.length > 1 ? 3 : 0;
  let offset = 0;
  let svg = "";
  entries.forEach(([, count], i) => {
    const frac = count / total;
    const len = Math.max(frac * C - gapPx, 1);
    svg += `<circle cx="60" cy="60" r="${R}" fill="none" stroke="${palette[i % palette.length]}"
      stroke-width="14" stroke-linecap="butt"
      stroke-dasharray="${len} ${C - len}" stroke-dashoffset="${-offset}"
      transform="rotate(-90 60 60)"/>`;
    offset += frac * C;
  });
  $("classdist-donut").innerHTML = svg;
  $("classdist-center").innerHTML = `${entries.length}<small>classes</small>`;

  $("classdist-legend").innerHTML = entries
    .map(
      ([label, count], i) => `
      <li><span class="swatch" style="background:${palette[i % palette.length]}"></span>
      ${escapeHtml(label)}<span class="count">${count.toLocaleString()} (${((count / total) * 100).toFixed(1)}%)</span></li>`
    )
    .join("");

  const covered = target.n_unique <= entries.length; // top_values holds every class
  const majority = entries[0][1];
  const minority = entries[entries.length - 1][1];
  const ratio = covered && minority > 0 ? majority / minority : null;
  const plan = run.resampling_plan || {};
  $("classdist-chips").innerHTML = `
    ${ratio != null ? `<span class="chip ${ratio >= 3 ? "flagged" : "detected"}">${ratio >= 3 ? ICONS.warning : ICONS.check} imbalance ratio ${ratio.toFixed(1)} : 1</span>` : ""}
    ${!covered ? `<span class="chip detected">top ${entries.length} of ${target.n_unique} classes shown</span>` : ""}
    ${plan.enabled ? `<span class="chip detected">${ICONS.check} ${escapeHtml(String(plan.method || "").replaceAll("_", " "))} applied during training</span>` : ""}`;
}

/* ================= data quality overview ================= */

function renderQuality(run) {
  const quality = (run.profile_summary || {}).quality;
  const card = $("quality-card");
  if (!quality) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  const overallPct = Math.round(quality.overall * 100);
  $("quality-sub").textContent = `${quality.duplicate_row_count.toLocaleString()} duplicate row(s)`;

  const R = 34, C = 2 * Math.PI * R;
  $("quality-ring").innerHTML = `
    <svg viewBox="0 0 80 80" role="img" aria-label="Overall data quality ${overallPct}%">
      <circle cx="40" cy="40" r="${R}" fill="none" stroke="var(--border-subtle)" stroke-width="8"/>
      <circle cx="40" cy="40" r="${R}" fill="none" stroke="var(--accent-success)" stroke-width="8" stroke-linecap="round"
        stroke-dasharray="${((overallPct / 100) * C).toFixed(2)} ${C.toFixed(2)}" transform="rotate(-90 40 40)"/>
    </svg>
    <div class="quality-ring-label"><strong>${overallPct}%</strong><small>overall</small></div>`;

  const dims = [
    { label: "Completeness", value: quality.completeness, hint: "share of cells that are not null" },
    { label: "Uniqueness", value: quality.uniqueness, hint: "share of rows that are not exact duplicates" },
  ];
  $("quality-bars").innerHTML = dims
    .map(
      (d) => `
      <div class="quality-row" title="${d.hint}">
        <span class="quality-name">${d.label}</span>
        <span class="fi-track"><span class="fi-fill quality-fill" style="width:${(d.value * 100).toFixed(1)}%"></span></span>
        <span class="quality-value mono">${Math.round(d.value * 100)}%</span>
      </div>`
    )
    .join("");
}
```

- [ ] **Step 3: Call the new renderers and add the stat card**

**(a)** Find (in `render`):

```javascript
  renderDatasetSummary(run);
  renderInsights(run);
```

Replace with:

```javascript
  renderDatasetSummary(run);
  renderClassDistribution(run);
  renderQuality(run);
  renderInsights(run);
```

**(b)** Find (in `renderStatCards`):

```javascript
  const insights = run.insights || [];
  if (insights.length) {
    cards.push({
      icon: "bulb", tint: "violet", label: "Auto insights",
      value: String(insights.length),
      sub: "generated from your data",
    });
  }
```

Replace with:

```javascript
  const quality = summary.quality;
  if (quality) {
    cards.push({
      icon: "shield", tint: "green", label: "Data quality",
      value: `${Math.round(quality.overall * 100)}%`,
      sub: "overall score",
    });
  }
  const insights = run.insights || [];
  if (insights.length) {
    cards.push({
      icon: "bulb", tint: "violet", label: "Auto insights",
      value: String(insights.length),
      sub: "generated from your data",
    });
  }
```

**(c)** Re-tint the new donut on theme change. Find (in `applyTheme`):

```javascript
  if (lastRun) renderDatasetSummary(lastRun); // re-tint donut for the new surface
```

Replace with:

```javascript
  if (lastRun) {
    renderDatasetSummary(lastRun); // re-tint donuts for the new surface
    renderClassDistribution(lastRun);
  }
```

- [ ] **Step 4: Add layout + panel styles to `frontend/styles.css`**

**(a)** Find:

```css
.donut-wrap { display: flex; align-items: center; gap: var(--sp-4); flex-wrap: wrap; position: relative; }
#donut { width: 150px; height: 150px; flex-shrink: 0; }
```

Replace with:

```css
.donut-wrap { display: flex; align-items: center; gap: var(--sp-4); flex-wrap: wrap; position: relative; }
#donut, #classdist-donut { width: 150px; height: 150px; flex-shrink: 0; }
```

**(b)** Append at the end of the file:

```css
/* ================= run layout: main column + right rail ================= */

.run-layout { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: var(--sp-3); align-items: start; }
.run-main { display: grid; gap: var(--sp-3); min-width: 0; }
.run-rail { display: grid; gap: var(--sp-3); min-width: 0; }
@media (max-width: 1100px) { .run-layout { grid-template-columns: 1fr; } }

/* ================= data quality overview ================= */

.quality-wrap { display: flex; align-items: center; gap: var(--sp-4); flex-wrap: wrap; }
.quality-ring { position: relative; width: 100px; height: 100px; flex-shrink: 0; }
.quality-ring svg { width: 100px; height: 100px; }
.quality-ring-label {
  position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center; pointer-events: none;
  font-family: var(--font-display); line-height: 1.1;
}
.quality-ring-label strong { font-size: var(--text-lg); }
.quality-ring-label small { font-size: var(--text-xs); color: var(--text-secondary); font-family: var(--font-body); }
.quality-bars { flex: 1; min-width: 180px; display: grid; gap: var(--sp-2); }
.quality-row { display: grid; grid-template-columns: minmax(90px, 120px) 1fr 44px; align-items: center; gap: var(--sp-2); font-size: var(--text-sm); }
.quality-name { font-size: var(--text-xs); font-weight: 600; color: var(--text-secondary); }
.quality-fill { background: var(--accent-success); }
.quality-value { font-size: var(--text-xs); text-align: right; }
```

- [ ] **Step 5: Syntax-check and eyeball**

Run: `node --check frontend/app.js`
Expected: no output (success)

- [ ] **Step 6: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: two-column run dashboard with data-quality and class-distribution panels"
```

---

### Task 6: AI Assistant right-rail panel (frontend)

**Files:**
- Modify: `frontend/app.js` (new `renderChat` + submit handler, `render` call, `openRun` reset)
- Modify: `frontend/styles.css` (chat styles)

(The panel's HTML was added in Task 5 step 1(d).)

**Interfaces:**
- Consumes: `run.chat_history`, `run.suggested_questions`, `POST /api/runs/{id}/chat` (Task 3); `#assistant-card` markup (Task 5); existing helpers `$`, `escapeHtml`, `poll`, `lastRun`, `currentRunId`.

- [ ] **Step 1: Add `renderChat` and the submit handler to `frontend/app.js`**

Find:

```javascript
/* ================= feature importance ================= */
```

Insert immediately before it:

```javascript
/* ================= AI assistant panel ================= */

let chatPendingQuestion = null;

function renderChat(run) {
  const ready = ["completed", "failed"].includes(run.status);
  $("chat-placeholder").classList.toggle("hidden", ready);
  $("chat-thread").classList.toggle("hidden", !ready);
  $("chat-suggestions").classList.toggle("hidden", !ready);
  $("chat-form").classList.toggle("hidden", !ready);
  if (!ready) return;

  const history = run.chat_history || [];
  const bubbles = history.map(
    (m) => `
      <div class="chat-msg chat-${m.role}">
        <span class="chat-role">${m.role === "user" ? "You" : "Assistant"}</span>
        <p>${escapeHtml(m.content)}</p>
      </div>`
  );
  if (chatPendingQuestion != null) {
    bubbles.push(`
      <div class="chat-msg chat-user">
        <span class="chat-role">You</span>
        <p>${escapeHtml(chatPendingQuestion)}</p>
      </div>`);
    bubbles.push(`
      <div class="chat-msg chat-assistant chat-thinking">
        <span class="chat-role">Assistant</span>
        <p>Thinking…</p>
      </div>`);
  }
  $("chat-thread").innerHTML = bubbles.length
    ? bubbles.join("")
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
  if (!question || !currentRunId || chatPendingQuestion != null) return;
  input.value = "";
  $("chat-error").classList.add("hidden");
  chatPendingQuestion = question;
  if (lastRun) renderChat(lastRun); // show the question + thinking bubble immediately
  $("chat-send-btn").disabled = true;
  try {
    const res = await fetch(`/api/runs/${currentRunId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    chatPendingQuestion = null;
    await poll(); // re-fetch: chat_history now contains both messages
  } catch (err) {
    chatPendingQuestion = null;
    if (lastRun) renderChat(lastRun);
    $("chat-error").textContent = "Could not get an answer: " + err.message;
    $("chat-error").classList.remove("hidden");
  } finally {
    $("chat-send-btn").disabled = false;
  }
});

/* ================= feature importance ================= */
```

- [ ] **Step 2: Call `renderChat` from `render` and reset chat state in `openRun`**

**(a)** Find:

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

**(b)** Find (in `openRun`):

```javascript
  predictFormLoadedFor = null;
  $("predict-result").classList.add("hidden");
  switchTab("report");
```

Replace with:

```javascript
  predictFormLoadedFor = null;
  $("predict-result").classList.add("hidden");
  chatPendingQuestion = null;
  $("chat-input").value = "";
  $("chat-error").classList.add("hidden");
  switchTab("report");
```

- [ ] **Step 3: Add chat styles to `frontend/styles.css`**

Append at the end of the file:

```css
/* ================= AI assistant panel ================= */

.chat-thread { display: flex; flex-direction: column; gap: var(--sp-2); max-height: 340px; overflow-y: auto; margin-bottom: var(--sp-3); }
.chat-msg { padding: 8px 12px; border-radius: var(--radius-sm); font-size: var(--text-sm); max-width: 88%; }
.chat-msg p { margin: 2px 0 0; white-space: pre-wrap; overflow-wrap: break-word; }
.chat-msg .chat-role { font-size: var(--text-xs); font-weight: 650; color: var(--text-secondary); }
.chat-msg.chat-user { align-self: flex-end; background: var(--accent-primary-soft); color: var(--text-primary); }
.chat-msg.chat-assistant { align-self: flex-start; background: var(--bg-surface-raised); border: 1px solid var(--border-subtle); }
.chat-msg.chat-thinking p { color: var(--text-secondary); font-style: italic; }
.chat-suggestions { display: flex; flex-wrap: wrap; gap: var(--sp-2); margin-bottom: var(--sp-3); }
.suggestion-chip {
  font: inherit; font-size: var(--text-xs); font-weight: 650; padding: 4px 10px; border-radius: 999px;
  background: var(--bg-surface-raised); color: var(--text-secondary); border: 1px solid var(--border-subtle);
  cursor: pointer; transition: color 0.15s ease, border-color 0.15s ease; text-align: left;
}
.suggestion-chip:hover { color: var(--accent-primary); border-color: var(--accent-primary); }
.chat-form { display: flex; gap: var(--sp-2); }
.chat-form input[type="text"] { flex: 1; min-width: 0; font-size: var(--text-sm); }
.chat-form .btn { padding: 9px 12px; flex-shrink: 0; }
.chat-error { color: var(--accent-danger); font-size: var(--text-xs); margin-top: var(--sp-2); }
```

- [ ] **Step 4: Syntax-check the JS**

Run: `node --check frontend/app.js`
Expected: no output (success)

- [ ] **Step 5: Commit**

```bash
git add frontend/app.js frontend/styles.css
git commit -m "feat: AI assistant chat panel in the dashboard right rail"
```

---

### Task 7: Full regression + manual browser verification

**Files:** none (verification only; spec update if reality diverged)

- [ ] **Step 1: Run the full backend test suite**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: all tests pass, no regressions.

- [ ] **Step 2: Manual mock-mode browser pass**

Start the server from the worktree with the mock LLM (PowerShell): `$env:AUTOML_MOCK_LLM = "1"; .venv/Scripts/python run_server.py` (background), open `http://127.0.0.1:8000`. Verify:

1. **First launch (no runs):** hero, intake card, trust chips, and feature grid render; stats strip and Recent Projects band are hidden; no console errors.
2. **Start a run** (small CSV, e.g. a churn-style file with a binary target): home's stats strip and Recent Projects appear on the next 4s refresh; Pipeline in Progress mini-rail shows stages advancing.
3. **Dashboard:** stat cards include Data Quality; pipeline rail, dataset summary, and (after confirming a classification task) the Class Distribution donut with imbalance chip render; Data Quality panel shows the ring + completeness/uniqueness bars; Recent Activity sits in the right rail.
4. **Checkpoints still work:** confirm task spec, approve feature plan, run completes.
5. **AI Assistant:** placeholder before completion; after completion, suggested-question chips populate the input (no auto-submit); submitting shows the question + "Thinking…" bubble, then the mock answer; a page refresh restores the conversation; a second question appends correctly.
6. **Both themes:** toggle dark mode — donuts re-tint, hero gradient readable, chat bubbles/quality bars legible.
7. **Responsive:** narrow the window below ~1100px — the rail stacks under the main column; below ~900px the sidebar collapses (existing behavior intact).

- [ ] **Step 3: Update the design spec if anything diverged**

If any implementation detail diverged from `docs/superpowers/specs/2026-07-05-mockup-parity-ui-design.md`, update the spec to match reality and commit:

```bash
git add -A
git commit -m "docs: align mockup-parity spec with implementation"
```

(Skip if nothing diverged.)
