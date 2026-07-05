# Mockup-parity home + run dashboard (real data only)

Date: 2026-07-05
Status: approved

## Context

Two reference mockups were reviewed: a marketing landing page and an
enterprise project dashboard. The user chose (a) a polished **in-app home**
rather than a separate marketing page, (b) **real-data mockup parity** for the
run dashboard — no fabricated metrics or placeholder enterprise chrome, and
(c) executing the already-approved AI Assistant chat plan
(`2026-07-05-ai-assistant-chat-design.md`) with its UI adapted from an
"Ask AI" tab to the mockup's right-rail assistant panel.

Explicitly out of scope because the local single-run build cannot back them
with real data: cost savings, credits/usage, resource usage, system health,
notifications, monitoring/deployments, experiment trend across historical
runs, trusted-by logos, testimonials, pricing.

## Goal

Make the existing no-build vanilla JS frontend look and feel like the
mockups while keeping every widget fed by real backend data, and ship the
AI Assistant chat as the dashboard's right-rail panel.

## 1. Home view (replaces the current intake screen)

The current `#intake-view` becomes a scrollable home page:

- **Hero**: eyebrow chips ("AI-Powered · Agentic · Automated"), headline
  "Build Better Models. **Automatically.**" (accent word in violet), subcopy
  describing the actual product, and the *working* intake form (goal
  textarea + CSV dropzone + estimate row + Run pipeline button) directly
  below — the hero is the create-project flow, not a fake CTA.
- **Trust chips** (real product properties, replacing mockup logos):
  "No coding required", "Runs locally", "Raw data never reaches the LLM",
  "Full audit trace".
- **Stats strip** from real runs: Total experiments (`GET /api/runs`
  length), Best score across completed runs (max over new `best_score`
  field), Active runs (status not in completed/failed/cancelled). Hidden
  when there are no runs yet (first-launch experience stays clean).
- **Two-column band** (hidden when empty):
  - *Recent Projects*: up to 5 runs — filename/description, status pill,
    score when completed, relative time; click opens the run.
  - *Pipeline in Progress*: a mini vertical stage rail for the newest
    active run (stage name + done/running/pending state + per-stage
    duration from `stage_timeline`); hidden when nothing is running.
- **"How it works" feature grid**: five static cards describing what the
  pipeline actually does — Smart Data Understanding (profiling, PII
  redaction), Feature Engineering (EDA-grounded plan you approve),
  Model Selection & Tuning (candidates, k-fold CV, Optuna),
  Explainable Insights (feature importance, auto insights, caveats),
  Human Checkpoints (confirm task spec, approve feature plan).

### API change

`GET /api/runs` items gain `best_score: float | null` and
`metric: str | null` (from `state["best_model"]` / `task_spec`), so the home
can show scores without N extra requests.

## 2. Run dashboard

- **Stat cards** extended to six, all real: Dataset (rows × cols), Best
  Model (name + metric + score), Candidates trained, Runtime, Auto Insights
  count, **Data Quality** (see below). Cards keep the existing
  hide-until-data pattern.
- **Data quality**: computed deterministically in
  `src/profiling/profile.py`, exposed as `profile["quality"]`:
  - `completeness` = 1 − mean null rate across columns
  - `duplicate_row_rate` = duplicated full rows / row count (new profiled
    field; requires reading it during `profile_dataset`, no LLM exposure
    concerns — it is an aggregate)
  - `uniqueness` = 1 − duplicate_row_rate
  - `overall` = mean of the dimensions, 0–100 scale in the UI
  A "Data Quality Overview" panel renders the dimension bars + overall
  ring. Only computable dimensions are shown (no invented "validity" or
  "timeliness" scores).
- **Pipeline rail** restyled to the mockup: circular stage icons joined by
  a progress line, stage label + state + duration under each; the live
  training banner (message, progress bar, ETA, per-candidate tuning lines)
  restyled to sit inside the rail card. Data is unchanged
  (`stages_done`, `stage_timeline`, existing train-progress fields).
- **Class Distribution panel** (classification runs, post-confirm): donut
  of the target column's `top_values` (already in `profile_columns` via the
  profile), imbalance ratio, and the resampling suggestion/decision line.
  Hidden for regression/forecasting or when `top_values` is absent.
- **Layout**: two-column dashboard — main column (pipeline, checkpoints,
  dataset summary, class distribution, model comparison, tuning trend,
  feature importance, data quality, report card) and a right rail
  (AI Assistant panel on top, Recent Activity below). Collapses to one
  column under ~1100px.
- Existing checkpoint cards, report/test tabs, caveats, and error cards
  keep their behavior; they are restyled only.

## 3. AI Assistant (delta vs the chat design spec)

Backend is implemented exactly per
`docs/superpowers/plans/2026-07-05-ai-assistant-chat.md` Tasks 1–3
(suggested-questions helper, `chat_node.py` + prompt, API endpoint +
mock-mode + `chat_history`/`suggested_questions` in `_run_summary`).

UI delta (supersedes the plan's Task 4): instead of a third tab, a
persistent right-rail card "AI Assistant":

- Before the run completes: dimmed placeholder "Available once your model
  is trained."
- After `status in (completed, failed)`: suggested-question chips (populate
  the input on click, don't auto-submit), scrollable message thread
  (user right-aligned soft-violet bubbles, assistant left-aligned surface
  bubbles), input + send button, inline "thinking" indicator while a
  request is in flight, inline error message on failure (no `alert()`).
- Chat renders from `run.chat_history` on every poll so refresh restores
  the conversation.

## Non-negotiables carried over

- No raw data in prompts (chat context is the already-redacted computed
  subset; quality metrics are aggregates).
- No new color families without dataviz palette validation; reuse existing
  tokens (`--accent-*`, `--cat-*`). Class-distribution donut uses the
  existing categorical palette.
- Both themes (light/dark) must stay correct; donuts re-tint on theme
  change like the existing dataset donut.

## Testing

- pytest: `profile_dataset` quality block (known-duplicates fixture ⇒ exact
  rates; no-duplicate case ⇒ uniqueness 1.0); `list_runs` includes
  `best_score`/`metric` (null before completion, populated after); the chat
  plan's own tests (Tasks 1–3) as written.
- Frontend: `node --check frontend/app.js`; manual mock-mode
  (`AUTOML_MOCK_LLM=1`) browser pass covering first-launch empty home,
  active-run home, full run to completion, chat round-trip, both themes.
- Full `pytest tests/ -q` regression (graph/state untouched, so the
  fixture-suite trigger in CLAUDE.md does not fire, but the full suite runs
  anyway).
