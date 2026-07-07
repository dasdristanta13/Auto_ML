# Model Summary Overview Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the run-results page from its current 2-tab (Report / Test the model) layout into a 7-tab layout (Overview / Data / Pipeline / Models Compared / Explainability / Artifacts / Logs), with a new Overview tab that surfaces the champion model, a condensed run journey, a condensed leaderboard, model-selection rationale, and pipeline-actions summary — all wired to data the app already computes and serializes today.

**Architecture:** Frontend-only change across `frontend/index.html`, `frontend/app.js`, `frontend/styles.css` (vanilla JS, no build step, no bundler). No backend/API changes — every widget reads fields already present in the `GET /api/runs/{id}` payload (`best_model`, `training_results`, `task_spec`, `report`, `stage_timeline`, `stages_done`, `eda_report`, `feature_plan`, `resampling_applied`, `feature_selection_result`, `caveats` via `report`, `suggested_questions`, `chat_history`, `events`). The existing `render(run)` dispatch function (called on every 1500ms poll tick) gains new render-function calls following its existing unconditional-re-render pattern — no diffing is introduced, matching the current codebase style.

**Tech Stack:** Vanilla JS (ES2020+), hand-authored HTML, CSS custom properties (existing light/dark theme tokens in `styles.css`). No test runner exists for the frontend (confirmed: no `*.test.js` anywhere in the repo; all `/tests` are Python/pytest for the backend). Verification for every task in this plan is manual, via a local run of the app in a browser — this matches how every prior frontend feature in this repo's git history (Column Summary, Correlations, Missing Values, Outliers sub-tabs, etc.) was verified.

## Global Constraints

- No backend/API changes in this plan — if a task seems to need one, stop and flag it rather than inventing an endpoint (per `docs/superpowers/specs/2026-07-07-model-summary-overview-design.md`).
- Do not render the Model Trust Score gauge, Business Recommendations card, or SHAP beeswarm chart — they are explicitly out of scope (separate follow-up specs B/C/D).
- "Deploy Model", "Generate API", "Schedule Retraining", "Monitor Model", "View SHAP Report" must render disabled with `title="Not available in this local build"`, matching the existing `.nav-item.disabled` convention in `frontend/index.html`.
- Reuse existing CSS custom properties (`--accent-primary`, `--bg-surface`, etc.) and existing class vocabulary (`.card`, `.card-head`, `.chip`, `.stat-card`, `.callout-list`, `.tab-bar`/`.tab-btn`) — no new color palette.
- Every new render function is a plain function taking `run` (the parsed JSON from `GET /api/runs/{id}`) and writing into the DOM by ID, following the exact style of the existing `renderFeatureImportance`, `renderCaveats`, etc. in `frontend/app.js`.
- After every task, run the app locally (`python run_server.py` or equivalent existing entrypoint) and manually click through the affected UI — do not mark a task done on code-reads-correctly alone.

---

## Task 1: Generic 7-tab system + relocate existing widgets

**Files:**
- Modify: `frontend/index.html:178-427` (run-view main content: stat cards, live-pipeline card, dash-grid, report-card)
- Modify: `frontend/app.js:994` (`openRun`'s `switchTab("report")` call)
- Modify: `frontend/app.js:2038-2047` (delete old `switchTab`, replace with generic `switchRunTab`)
- Modify: `frontend/styles.css:673` (`.run-layout` — add a `.no-rail` variant)

**Interfaces:**
- Produces: `RUN_TABS` array (`["overview", "pipeline", "models", "explainability", "artifacts", "logs"]`), `switchRunTab(name)` function, `#run-layout` and `#run-rail` element IDs, `#tab-overview-panel` … `#tab-logs-panel` container IDs that every later task appends into. `#test-model-btn` button (functional, toggles `#tab-test-panel`).
- Consumes: existing `openDatasetDetail(runId)` (already defined, `app.js:290`), existing `lastRun` global (`app.js`, set in `render()`), existing `loadPredictTab(run)` (`app.js:2051`).

### Step 1: Add `id` attributes needed for tab/rail toggling

In `frontend/index.html`, find:

```html
      <div class="run-layout">
      <div class="run-main">
```

Replace with:

```html
      <div class="run-layout" id="run-layout">
      <div class="run-main">
```

Find:

```html
      <aside class="run-rail">
```

Replace with:

```html
      <aside class="run-rail" id="run-rail">
```

### Step 2: Replace the "Live pipeline progress" card with a Pipeline tab panel

In `frontend/index.html`, the run-view currently opens with (lines 178-218):

```html
    <main class="content hidden" id="run-view">

      <!-- stat cards -->
      <section class="stat-row" id="stat-cards"></section>

      <div class="run-layout">
      <div class="run-main">

      <!-- new experiment on this dataset (multi-experiment spec) -->
      <div class="card checkpoint hidden" id="rerun-card">
        ...
      </div>

      <!-- live pipeline -->
      <section class="card">
        <div class="card-head">
          <h3>Live pipeline progress</h3>
          <span class="muted small" id="pipeline-sub"></span>
        </div>
        <ol class="stage-tracker" id="stage-tracker"></ol>
        <div class="train-progress hidden" id="train-progress">
          <div class="train-progress-text" id="train-progress-text"></div>
          <div class="train-progress-track"><div class="train-progress-fill" id="train-progress-fill"></div></div>
          <div class="tuning-progress-list" id="tuning-progress-list"></div>
        </div>
      </section>
```

Insert a new 7-tab bar immediately after `<section class="stat-row" id="stat-cards"></section>` and before `<div class="run-layout" id="run-layout">`, and move the "live pipeline" `<section>` out of the always-visible flow and into a new `#tab-pipeline-panel` (built in Step 4, after the tab bar). For this step, just add the tab bar and wrap `run-main`'s first real content in `#tab-overview-panel`:

```html
      <!-- stat cards -->
      <section class="stat-row" id="stat-cards"></section>

      <div class="tab-bar run-tab-bar" role="tablist">
        <button class="tab-btn active" id="tab-overview-btn" type="button" role="tab" aria-selected="true">Overview</button>
        <button class="tab-btn" id="tab-data-btn" type="button" role="tab" aria-selected="false">Data</button>
        <button class="tab-btn" id="tab-pipeline-btn" type="button" role="tab" aria-selected="false">Pipeline</button>
        <button class="tab-btn" id="tab-models-btn" type="button" role="tab" aria-selected="false">Models Compared</button>
        <button class="tab-btn" id="tab-explainability-btn" type="button" role="tab" aria-selected="false">Explainability</button>
        <button class="tab-btn" id="tab-artifacts-btn" type="button" role="tab" aria-selected="false">Artifacts</button>
        <button class="tab-btn" id="tab-logs-btn" type="button" role="tab" aria-selected="false">Logs</button>
      </div>

      <div class="run-layout" id="run-layout">
      <div class="run-main">

      <div id="tab-overview-panel">

      <!-- new experiment on this dataset (multi-experiment spec) -->
      <div class="card checkpoint hidden" id="rerun-card">
        ...
      </div>

      <button type="button" class="btn ghost" id="test-model-btn">Test this model</button>
      <div id="tab-test-panel-slot"></div>

      </div><!-- /tab-overview-panel -->

      <div id="tab-pipeline-panel" class="hidden">
        <!-- live pipeline -->
        <section class="card">
          <div class="card-head">
            <h3>Pipeline stages</h3>
            <span class="muted small" id="pipeline-sub"></span>
          </div>
          <ol class="stage-tracker" id="stage-tracker"></ol>
          <div class="train-progress hidden" id="train-progress">
            <div class="train-progress-text" id="train-progress-text"></div>
            <div class="train-progress-track"><div class="train-progress-fill" id="train-progress-fill"></div></div>
            <div class="tuning-progress-list" id="tuning-progress-list"></div>
          </div>
        </section>
      </div><!-- /tab-pipeline-panel -->
```

(The `...` is the existing `#rerun-card` content, unchanged — do not retype it, just leave it in place inside the new `#tab-overview-panel` wrapper div.) Note `#tab-test-panel-slot` is an empty placeholder div — Step 6 moves the real `#tab-test-panel` markup into it.

### Step 3: Split the dash-grid into Models Compared / Explainability / Overview leftovers

Currently (`index.html:331-396`):

```html
      <!-- dashboard grid -->
      <section class="dash-grid">
        <div class="card hidden" id="insights-card"> ... </div>
        <div class="card hidden" id="dataset-card"> ... </div>
        <div class="card hidden" id="classdist-card"> ... </div>
        <div class="card hidden" id="quality-card"> ... </div>

        <div class="card hidden" id="results-card"> ... </div>

        <div class="card hidden" id="tuning-card"> ... </div>

        <div class="card hidden" id="fi-card"> ... </div>

      </section>
```

Split this into two groups. `#fi-card` ("Top Drivers") stays in Overview per the spec's left-column list (item 6) — it is **not** duplicated into Explainability; the Explainability tab gets only the future-SHAP note for this pass, so there is exactly one `#fi-list` element in the DOM. `insights-card`/`dataset-card`/`classdist-card`/`quality-card`/`fi-card` all stay inside `#tab-overview-panel` (`#tab-overview-panel` does **not** close yet — it stays open through Step 4, since `caveats-card`/`error-card`/the AI-summary content still need to land inside it/the rail). Concretely:

1. Leave the `</div><!-- /tab-overview-panel -->` placement alone for now — it moves to the very end, after Step 4.
2. Change the dash-grid's surrounding markup to pull `results-card`/`tuning-card` out into a new Models Compared tab, and `fi-card` into its own section within Overview (not the dash-grid, so it can sit full-width as its own card):

```html
      <!-- dashboard grid -->
      <section class="dash-grid">
        <div class="card hidden" id="insights-card"> ... </div>
        <div class="card hidden" id="dataset-card"> ... </div>
        <div class="card hidden" id="classdist-card"> ... </div>
        <div class="card hidden" id="quality-card"> ... </div>
      </section>

      <div class="card hidden" id="fi-card"> ... </div>

      <div id="tab-models-panel" class="hidden">
        <section class="dash-grid">
          <div class="card hidden" id="results-card"> ... </div>
          <div class="card hidden" id="tuning-card"> ... </div>
        </section>
      </div><!-- /tab-models-panel -->

      <div id="tab-explainability-panel" class="hidden">
        <p class="muted small">SHAP-based impact analysis isn't available for this run yet — see the "Top Drivers" card on the Overview tab for current feature importance. A future update will add per-prediction explanation charts here.</p>
      </div><!-- /tab-explainability-panel -->
```

(Keep every card's internal markup — `card-head`, `#results-table`, `#fi-list`, etc. — byte-for-byte unchanged; only the wrapping `<section class="dash-grid">`/tab-panel divs move. Note `#tab-models-panel` and `#tab-explainability-panel` are inserted here textually, but physically they should end up **after** `#tab-overview-panel` closes in the final file — do this edit as two separate moves: (a) delete `results-card`/`tuning-card` from the dash-grid and relocate them into a new `#tab-models-panel` div placed immediately after `#tab-overview-panel`'s eventual closing tag from Step 4; (b) delete `fi-card` from its old spot and reinsert it inside `#tab-overview-panel`, directly after the dash-grid `</section>`; (c) add the new `#tab-explainability-panel` div, with only the muted note, right after `#tab-models-panel`.)

### Step 4: Move `report-card`'s contents into Artifacts / Logs tabs, delete the old 2-tab bar

Currently (`index.html:398-427`):

```html
      <!-- report + try-the-model, tabbed -->
      <div class="card hidden" id="report-card">
        <div class="tab-bar" role="tablist">
          <button class="tab-btn active" id="tab-report-btn" type="button" role="tab" aria-selected="true">Report</button>
          <button class="tab-btn" id="tab-test-btn" type="button" role="tab" aria-selected="false">Test the model</button>
        </div>

        <div id="tab-report-panel" role="tabpanel">
          <div id="report-lede" class="report-lede hidden"></div>
          <div id="report-body" class="report-body"></div>
          <div class="result-actions">
            <a class="btn primary" id="download-btn" href="#" download>Download best model</a>
            <a class="btn ghost" id="download-script-btn" href="#" download>Download training script</a>
            <button class="btn ghost" id="trace-toggle-btn" type="button">View LLM audit trace</button>
          </div>
          <details class="trace-details hidden" id="trace-details">
            <summary>Raw LLM trace (technical)</summary>
            <div class="trace-body" id="trace-body"></div>
          </details>
        </div>

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

Replace with (`#tab-test-panel` moves into the `#tab-test-panel-slot` placeholder from Step 2; downloads move to Artifacts; trace moves to Logs; `report-lede`/`report-body` move into the right rail, not Overview's main column — see below):

```html
      <div id="tab-artifacts-panel" class="hidden">
        <section class="card">
          <div class="card-head"><h3>Artifacts</h3></div>
          <div class="result-actions">
            <a class="btn primary" id="download-btn" href="#" download>Download best model</a>
            <a class="btn ghost" id="download-script-btn" href="#" download>Download training script</a>
          </div>
        </section>
      </div><!-- /tab-artifacts-panel -->

      <div id="tab-logs-panel" class="hidden">
        <section class="card">
          <div class="card-head"><h3>Pipeline events</h3></div>
          <ul class="reasoning-log" id="logs-tab-events"></ul>
        </section>
        <section class="card">
          <div class="card-head"><h3>LLM audit trace</h3></div>
          <button class="btn ghost" id="trace-toggle-btn" type="button">View LLM audit trace</button>
          <details class="trace-details hidden" id="trace-details">
            <summary>Raw LLM trace (technical)</summary>
            <div class="trace-body" id="trace-body"></div>
          </details>
        </section>
      </div><!-- /tab-logs-panel -->
```

Place these two new panels immediately after `#tab-explainability-panel` (from Step 3).

Inside `#tab-test-panel-slot` (from Step 2), place:

```html
      <div id="tab-test-panel" class="hidden" role="tabpanel">
        <p class="muted small">Runs the saved model locally against values you enter here; nothing leaves this machine.</p>
        <form id="predict-form" class="predict-grid"></form>
        <div class="btn-row">
          <button type="submit" form="predict-form" class="btn primary">Predict</button>
        </div>
        <div id="predict-result" class="predict-result hidden"></div>
      </div>
```

Now close out `#tab-overview-panel` properly. Directly inside it, right after the `#fi-card` div added in Step 3, insert the caveats/error cards (unchanged, just relocated from their old spot right after the old `report-card` block) and then close the panel:

```html
      <!-- caveats — unmissable, not fine print -->
      <div class="card caveats-card hidden" id="caveats-card">
        <h3> ... </h3>
        <ul class="callout-list" id="caveats-list"></ul>
      </div>

      <!-- errors -->
      <div class="card error-card hidden" id="error-card">
        <h3>Issues encountered</h3>
        <ul class="callout-list" id="error-list"></ul>
      </div>

      </div><!-- /tab-overview-panel -->
```

(Keep `caveats-card`'s and `error-card`'s internal markup exactly as it already is in the file — only their position moves, from after the old `report-card` to here, and the stray extra `</div>` that used to close `run-main`'s section around them should **not** be duplicated; there is exactly one `</div><!-- /tab-overview-panel -->` in the whole file after this move.)

Finally, `report-lede`/`report-body` do **not** go in Overview's main column — the approved spec places "AI Summary" in the **right rail**. Find `<aside class="run-rail" id="run-rail">` (from Step 1) and insert as its first children:

```html
      <aside class="run-rail" id="run-rail">
        <div id="report-lede" class="report-lede hidden"></div>
        <div id="report-body" class="report-body"></div>
```

(Task 8 Step 1 wraps these two bare divs in a proper `.ai-summary-card` — for this task they're just relocated, unstyled, which is fine since Task 8 runs before this plan is considered done.)

### Step 5: `frontend/app.js` — generic `switchRunTab`, replacing the old 2-way `switchTab`

Find (`app.js:2038-2047`):

```js
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

```js
const RUN_TABS = ["overview", "pipeline", "models", "explainability", "artifacts", "logs"];

function switchRunTab(name) {
  for (const tab of RUN_TABS) {
    const isActive = tab === name;
    $(`tab-${tab}-btn`).classList.toggle("active", isActive);
    $(`tab-${tab}-btn`).setAttribute("aria-selected", String(isActive));
    $(`tab-${tab}-panel`).classList.toggle("hidden", !isActive);
  }
  $("run-rail").classList.toggle("hidden", name !== "overview");
  $("run-layout").classList.toggle("no-rail", name !== "overview");
}
for (const tab of RUN_TABS) {
  $(`tab-${tab}-btn`).addEventListener("click", () => switchRunTab(tab));
}
$("tab-data-btn").addEventListener("click", () => {
  if (lastRun) openDatasetDetail(lastRun.source_run_id || lastRun.run_id);
});

function toggleTestModelPanel() {
  const panel = $("tab-test-panel");
  const wasHidden = panel.classList.contains("hidden");
  panel.classList.toggle("hidden");
  if (wasHidden && lastRun) loadPredictTab(lastRun);
}
$("test-model-btn").addEventListener("click", toggleTestModelPanel);
```

### Step 6: Fix the two remaining references to the old `switchTab`

Find (`app.js:994`, inside `openRun`):

```js
  switchTab("report");
```

Replace with:

```js
  switchRunTab("overview");
  $("tab-test-panel").classList.add("hidden");
```

Find (`app.js:2024`, inside `renderReport`):

```js
  if ($("tab-test-btn").classList.contains("active")) loadPredictTab(run);
```

Replace with:

```js
  if (!$("tab-test-panel").classList.contains("hidden")) loadPredictTab(run);
```

### Step 7: Populate the new Logs-tab events list

Add a small shared helper and call it from `render()`. Find (`app.js:1354-1364`, inside `renderReasoningRail`):

```js
  const events = run.events || [];
  $("reasoning-log").innerHTML = events.length
    ? [...events]
        .reverse()
        .map(
          (e) => `<li>${ICONS.check}<span>${escapeHtml(e.message)}</span>${
            e.timestamp ? `<span class="reasoning-log-time">${new Date(e.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>` : ""
          }</li>`
        )
        .join("")
    : `<li class="muted small">No stages completed yet.</li>`;
}
```

Replace with:

```js
  renderEventsLog($("reasoning-log"), run.events || []);
}

function renderEventsLog(container, events) {
  container.innerHTML = events.length
    ? [...events]
        .reverse()
        .map(
          (e) => `<li>${ICONS.check}<span>${escapeHtml(e.message)}</span>${
            e.timestamp ? `<span class="reasoning-log-time">${new Date(e.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>` : ""
          }</li>`
        )
        .join("")
    : `<li class="muted small">No stages completed yet.</li>`;
}

function renderLogsTab(run) {
  renderEventsLog($("logs-tab-events"), run.events || []);
}
```

Then find (`app.js:1057`, inside `render()`):

```js
  renderReasoningRail(run);
```

Add a line right after it:

```js
  renderReasoningRail(run);
  renderLogsTab(run);
```

### Step 8: `.run-layout.no-rail` CSS

In `frontend/styles.css`, find (line 673):

```css
.run-layout { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: var(--sp-3); align-items: start; }
```

Add right after it:

```css
.run-layout.no-rail { grid-template-columns: 1fr; }
.run-tab-bar { margin-bottom: var(--sp-3); }
```

### Step 9: Manual verification

Run the app locally, open a completed run. Confirm:
- 7 tabs render, "Overview" active by default.
- Clicking "Data" navigates to the Dataset Detail page for this run's dataset (not a blank panel).
- Clicking "Pipeline" shows the stage tracker; clicking "Models Compared" shows the results table + tuning chart; clicking "Explainability" shows the feature-importance bars + the SHAP not-yet-available note; clicking "Artifacts" shows the two download links; clicking "Logs" shows the events list and the (collapsed) trace toggle.
- "Test this model" button toggles the predict form open/closed and it still submits correctly.
- The right rail (AI Assistant chat, activity feed) is visible on Overview and hidden on every other tab, with Overview's main column expanding to full width on the other tabs.
- Nothing in the existing Report content (lede/body) is missing — it's now inside Overview, unstyled but present.

### Step 10: Commit

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "$(cat <<'EOF'
feat: restructure run view into 7-tab layout (Overview/Data/Pipeline/Models Compared/Explainability/Artifacts/Logs)

Relocates existing widgets (stage tracker, results table, tuning chart,
feature importance, downloads, LLM trace) into their new tabs without
changing their rendering logic; Data tab redirects to the existing
Dataset Detail page instead of duplicating its markup/IDs.
EOF
)"
```

---

## Task 2: Breadcrumb

**Files:**
- Modify: `frontend/index.html` (insert above `<header class="page-header">`, `index.html:61`)
- Modify: `frontend/app.js` (add `renderBreadcrumb(run)`, call from `render()`; show/hide in `showRunView`/`showIntakeView`)

**Interfaces:**
- Consumes: `run.filename`, `run.source_run_id`, `run.run_id` (all confirmed present, `app.js:1035-1038`).
- Produces: `renderBreadcrumb(run)`, called from `render()`.

### Step 1: Markup

In `frontend/index.html`, find:

```html
    <header class="page-header">
```

Insert immediately before it:

```html
    <div class="breadcrumb hidden" id="run-breadcrumb">
      <button type="button" class="breadcrumb-link" id="run-breadcrumb-datasets">Datasets</button>
      <span class="breadcrumb-sep">/</span>
      <span id="run-breadcrumb-name"></span>
    </div>
```

### Step 2: Render function

In `frontend/app.js`, add near `renderHeaderTags` (find that function; add this one right after its closing brace):

```js
function renderBreadcrumb(run) {
  $("run-breadcrumb").classList.remove("hidden");
  $("run-breadcrumb-name").textContent = run.filename;
}
```

Wire the click handler once, near the other top-level `addEventListener` calls (e.g. right after the `$("cancel-btn")` handler, `app.js:971-976`):

```js
$("run-breadcrumb-datasets").addEventListener("click", () => {
  if (lastRun) openDatasetDetail(lastRun.source_run_id || lastRun.run_id);
});
```

### Step 3: Call it from `render()` and hide it elsewhere

Find (`app.js:1043`):

```js
  renderHeaderTags(run);
```

Add right after:

```js
  renderHeaderTags(run);
  renderBreadcrumb(run);
```

Find `showIntakeView()`'s reset block (`app.js:200-208`, look for `$("header-tags").classList.add("hidden");`). Add right after it:

```js
  $("run-breadcrumb").classList.add("hidden");
```

Do the same in `showDatasetsView()` and `showDatasetDetailView()` (both already reset `$("header-tags")` around `app.js:239` and `app.js:287` — add `$("run-breadcrumb").classList.add("hidden");` next to each).

### Step 4: Manual verification

Open a completed run: breadcrumb reads "Datasets / {filename}". Click it: navigates to that dataset's detail page. Navigate to the dashboard/datasets/dataset-detail views: breadcrumb is hidden (no stale text left over from a previous run).

### Step 5: Commit

```bash
git add frontend/index.html frontend/app.js
git commit -m "feat: add Datasets breadcrumb above the run header"
```

---

## Task 3: Champion banner

**Files:**
- Modify: `frontend/index.html` (`#tab-overview-panel`, insert as its first child)
- Modify: `frontend/app.js` (new `renderChampionBanner(run)`, called from `render()`)
- Modify: `frontend/styles.css` (new `.champion-banner` rules)

**Interfaces:**
- Consumes: `run.best_model` (`{candidate_name, metrics, duration_seconds}`), `run.task_spec.metric`, `run.training_results` (array), `run.cv_config`/`run.best_model.cv_metrics` for fold count, `run.report` (truthy check reused from existing `export-report-btn` toggle logic at `app.js:1042`).
- Produces: `renderChampionBanner(run)`, `#champion-banner` markup, moves the existing `#test-model-btn` (Task 1 Step 2) inside the banner's action row.

### Step 1: Markup

In `frontend/index.html`, inside `#tab-overview-panel` (the div opened in Task 1 Step 2), as the very first line after `<div id="tab-overview-panel">`, insert:

```html
        <section class="card champion-banner hidden" id="champion-banner">
          <span class="champion-trophy">${ICONS.trophy}</span>
          <div class="champion-banner-body">
            <div class="champion-banner-label">Champion Model Selected</div>
            <h2 id="champion-banner-name"></h2>
            <div class="champion-banner-stats" id="champion-banner-stats"></div>
          </div>
          <div class="champion-banner-actions">
            <button type="button" class="btn ghost" id="champion-compare-btn">Compare Models</button>
            <button type="button" class="btn ghost hidden" id="champion-download-btn">Download Report</button>
            <span class="btn ghost disabled" title="Not available in this local build">Deploy Model</span>
          </div>
        </section>
```

(Note: `${ICONS.trophy}` is literal placeholder text here because this is static HTML, not a template — replace it with the actual inline `<svg>` markup already used for the trophy icon elsewhere. Find the trophy `<svg>` markup in `frontend/index.html`'s existing icon usages — e.g. search for `stroke-width="2"><path d="M8 21h8m-4-4v4M7 4h10v6a5 5 0 0 1-10 0V4Z"` — and paste that literal `<svg>` tag in place of `${ICONS.trophy}` above, matching how every other static icon in `index.html` is written inline. If no exact match exists in `index.html` yet, copy the `trophy` SVG path data from `ICONS.trophy` in `app.js:23` into a hand-written `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 21h8m-4-4v4M7 4h10v6a5 5 0 0 1-10 0V4Z"/><path d="M7 6H4a2 2 0 0 0 2 5m11-5h3a2 2 0 0 1-2 5"/></svg>` element.)

Then move the existing `#test-model-btn` button (added in Task 1 Step 2, currently a sibling right after `#rerun-card`) so it sits inside `.champion-banner-actions`, after the `Deploy Model` span:

```html
            <button type="button" class="btn ghost disabled" title="Not available in this local build">Deploy Model</button>
            <button type="button" class="btn primary" id="test-model-btn">Test this model</button>
```

(Delete the old standalone `<button ... id="test-model-btn">Test this model</button>` line from where Task 1 Step 2 placed it; it now lives here instead. `#tab-test-panel-slot` stays where it was.)

### Step 2: Render function

In `frontend/app.js`, add near `renderStatCards` (right before its `function renderStatCards(run) {` line):

```js
function renderChampionBanner(run) {
  const best = run.best_model || {};
  const card = $("champion-banner");
  if (!best.candidate_name) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  $("champion-banner-name").textContent = best.candidate_name;

  const metric = (run.task_spec || {}).metric;
  const results = run.training_results || [];
  const lowerIsBetter = !!(best.tuning || {}).lower_is_better;
  const sorted = [...results]
    .filter((r) => r.status === "succeeded" && metric && r.metrics && metric in r.metrics)
    .sort((a, b) => (lowerIsBetter ? a.metrics[metric] - b.metrics[metric] : b.metrics[metric] - a.metrics[metric]));
  const runnerUp = sorted.find((r) => r.run_id !== best.run_id);

  const stats = [];
  if (metric && best.metrics && metric in best.metrics) {
    const bestScore = Number(best.metrics[metric]);
    let deltaText = "";
    if (runnerUp) {
      const delta = lowerIsBetter ? runnerUp.metrics[metric] - bestScore : bestScore - runnerUp.metrics[metric];
      deltaText = ` <span class="champion-delta">${delta >= 0 ? "+" : ""}${delta.toFixed(3)} vs next best</span>`;
    }
    stats.push(`<div><span class="champion-stat-label">${escapeHtml(metric.toUpperCase())}</span><strong>${bestScore.toFixed(3)}</strong>${deltaText}</div>`);
  }
  if (best.duration_seconds != null) {
    stats.push(`<div><span class="champion-stat-label">Training Time</span><strong>${formatDuration(best.duration_seconds)}</strong></div>`);
  }
  if (best.cv_folds) {
    stats.push(`<div><span class="champion-stat-label">Cross Validation</span><strong>${best.cv_folds} Fold${best.resampling_applied ? ` + ${best.resampling_applied.replaceAll("_", " ")}` : ""}</strong></div>`);
  }
  $("champion-banner-stats").innerHTML = stats.join("");

  $("champion-download-btn").classList.toggle("hidden", !run.report);
}

$("champion-compare-btn").addEventListener("click", () => switchRunTab("models"));
$("champion-download-btn").addEventListener("click", (e) => {
  e.preventDefault();
  $("export-report-btn").click();
});
```

(`formatDuration` and `escapeHtml` already exist in `app.js` — reuse them, do not redefine. `#champion-download-btn` has no real `href` of its own — the existing `#export-report-btn` click handler (`app.js:1380-1388`) builds the report as a client-side Blob download rather than fetching a URL, so the banner button just delegates to that existing handler rather than duplicating the Blob logic.)

### Step 3: Call it from `render()`

Find (`app.js:1055`):

```js
  renderStatCards(run);
```

Add right before it:

```js
  renderChampionBanner(run);
  renderStatCards(run);
```

### Step 4: CSS

In `frontend/styles.css`, add near the other run-layout rules (after the `.run-layout.no-rail` rule added in Task 1 Step 8):

```css
.champion-banner {
  display: flex; align-items: center; gap: var(--sp-4);
  background: var(--accent-primary); color: #fff;
}
.champion-banner h2, .champion-banner-label { color: #fff; }
.champion-banner-label { font-size: var(--text-xs); font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.85; }
.champion-trophy { flex-shrink: 0; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; background: rgba(255,255,255,0.15); border-radius: 999px; }
.champion-trophy .icon { width: 26px; height: 26px; }
.champion-banner-body { flex: 1; min-width: 0; }
.champion-banner-stats { display: flex; gap: var(--sp-4); flex-wrap: wrap; margin-top: var(--sp-2); font-size: var(--text-sm); }
.champion-banner-stats strong { display: block; font-size: var(--text-lg); font-family: var(--font-display); }
.champion-stat-label { display: block; font-size: var(--text-xs); opacity: 0.85; }
.champion-delta { font-size: var(--text-xs); opacity: 0.9; }
.champion-banner-actions { display: flex; flex-direction: column; gap: var(--sp-2); flex-shrink: 0; }
.champion-banner-actions .btn.ghost { background: rgba(255,255,255,0.15); color: #fff; border-color: rgba(255,255,255,0.3); }
.champion-banner-actions .btn.disabled { opacity: 0.5; pointer-events: none; }
```

### Step 5: Manual verification

Open a completed run with ≥2 candidates: banner shows champion name, primary metric with a "+X vs next best" delta, training time, CV info. "Compare Models" switches to the Models Compared tab. "Test this model" still opens the predict form. "Deploy Model" is visibly disabled. Open a run with only 1 candidate: no delta text shown, nothing throws.

### Step 6: Commit

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: add champion model banner to Overview tab"
```

---

## Task 4: Journey of This Run (condensed timeline)

**Files:**
- Modify: `frontend/index.html` (insert after `#champion-banner` in `#tab-overview-panel`)
- Modify: `frontend/app.js` (new `renderJourneyCondensed(run)`, called from `render()`)
- Modify: `frontend/styles.css` (new `.journey-condensed` rules)

**Interfaces:**
- Consumes: `run.stage_timeline` (array of `{node, duration_seconds}` — confirmed via `_stage_timeline` in `src/api/server.py`), `run.stages_done` (array of node names), existing `STAGES` array (`app.js:30-44`) for node grouping, `run.best_model.candidate_name`.
- Produces: `renderJourneyCondensed(run)`.

### Step 1: Markup

In `frontend/index.html`, after the `</section>` closing `#champion-banner` (Task 3 Step 1), insert:

```html
        <section class="card">
          <div class="card-head"><h3>Journey of This Run</h3></div>
          <ol class="journey-condensed" id="journey-condensed"></ol>
          <button type="button" class="btn ghost" id="journey-view-pipeline-btn">View Full Pipeline</button>
        </section>
```

### Step 2: Render function

In `frontend/app.js`, add:

```js
const JOURNEY_GROUPS = [
  { label: "Data Received", nodes: ["profile"] },
  { label: "Data Inspection", nodes: ["leakage_check", "eda"] },
  { label: "Feature Engineering", nodes: ["feature_engineering", "apply_feature_plan"] },
  { label: "Model Search", nodes: ["model_selection", "dispatch_training", "poll_training"] },
  { label: "Evaluation", nodes: ["evaluate"] },
  { label: "Champion Selected", nodes: ["report"] },
];

function renderJourneyCondensed(run) {
  const done = new Set(run.stages_done || []);
  const durations = {};
  for (const rec of run.stage_timeline || []) durations[rec.node] = rec.duration_seconds;
  const best = run.best_model || {};

  $("journey-condensed").innerHTML = JOURNEY_GROUPS.map((group, i) => {
    const realNodes = group.nodes.map((n) => (n === "poll_training" ? "evaluate" : n));
    const isDone = group.nodes.every((n) => done.has(n === "poll_training" ? "evaluate" : n));
    const lastNode = group.nodes[group.nodes.length - 1];
    const duration = durations[lastNode === "poll_training" ? "evaluate" : lastNode];
    const sub = group.label === "Champion Selected" && best.candidate_name ? best.candidate_name : "";
    return `
      <li class="${isDone ? "done" : "pending"}">
        <span class="journey-num">${isDone ? ICONS.check : i + 1}</span>
        <span class="journey-label">${i + 1}. ${escapeHtml(group.label)}</span>
        ${sub ? `<span class="muted small">${escapeHtml(sub)}</span>` : ""}
        ${isDone && duration != null ? `<span class="journey-time">${formatDuration(duration)}</span>` : ""}
      </li>`;
  }).join("");
}

$("journey-view-pipeline-btn").addEventListener("click", () => switchRunTab("pipeline"));
```

### Step 3: Call it from `render()`

Find the line added in Task 3 Step 3 (`renderChampionBanner(run);`). Add right after it:

```js
  renderChampionBanner(run);
  renderJourneyCondensed(run);
  renderStatCards(run);
```

### Step 4: CSS

```css
.journey-condensed { list-style: none; padding: 0; display: grid; gap: var(--sp-2); margin-bottom: var(--sp-3); }
.journey-condensed li { display: flex; align-items: center; gap: var(--sp-3); padding: 8px 0; }
.journey-condensed li.pending { opacity: 0.5; }
.journey-num {
  width: 22px; height: 22px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: var(--text-xs); font-weight: 700;
  background: var(--accent-success-soft); color: var(--accent-success);
}
.journey-condensed li.pending .journey-num { background: var(--bg-surface-raised); color: var(--text-secondary); }
.journey-num .icon { width: 13px; height: 13px; }
.journey-label { font-weight: 600; flex: 1; }
.journey-time { font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-secondary); }
```

### Step 5: Manual verification

On a completed run, all 6 rows show green checks with durations and the champion's name under "Champion Selected". On an in-progress run (mid-training), later rows show pending/dimmed with no crash. "View Full Pipeline" switches to the Pipeline tab.

### Step 6: Commit

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: add condensed Journey of This Run timeline to Overview tab"
```

---

## Task 5: Condensed Model Leaderboard + explainability-star heuristic

**Files:**
- Modify: `frontend/index.html` (insert after `#journey-condensed` section)
- Modify: `frontend/app.js` (new `MODEL_EXPLAINABILITY_STARS`, `renderLeaderboardCondensed(run)`)
- Modify: `frontend/styles.css` (new `.leaderboard-condensed` / `.stars` rules)

**Interfaces:**
- Consumes: `run.training_results`, `run.task_spec.metric`, `run.best_model.run_id`, each result's `candidate_name` and `duration_seconds`.
- Produces: `renderLeaderboardCondensed(run)`, `explainabilityStars(candidateName)`.

**Important — no backend change available:** `CandidateModel.estimator`/`.library` (`src/state.py:60-65`) are never serialized to the frontend — confirmed by reading `_run_summary()` in `src/api/server.py:365-430`: it returns `training_results` (raw `TrainingResult` dicts, which only have `candidate_name`/`metrics`/`duration_seconds`/etc., no `estimator`) and does **not** include `candidate_models` at all. Per this plan's Global Constraints (no backend/API changes), the star rating below is keyed off `candidate_name` (an LLM-assigned free-text label, e.g. "Logistic Regression", "Random Forest", "XGBoost") via case-insensitive keyword matching, not the estimator class name. This is a fuzzier heuristic than matching on `estimator` would be, and is documented as such in the UI tooltip.

### Step 1: Markup

```html
        <section class="card">
          <div class="card-head">
            <h3>Model Leaderboard</h3>
            <span class="muted small" id="leaderboard-condensed-sub"></span>
          </div>
          <div class="table-scroll"><table class="results-table" id="leaderboard-condensed-table"></table></div>
          <button type="button" class="btn ghost hidden" id="leaderboard-view-all-btn">View all models</button>
        </section>
```

### Step 2: Render function

```js
// Keyed by keyword found in the LLM-assigned candidate_name (case-insensitive),
// checked in order — first match wins. Falls back to 3 stars when nothing
// matches. This is a fuzzy, best-effort heuristic (candidate_name is free text,
// not an enum), not a per-run measurement.
const EXPLAINABILITY_KEYWORD_STARS = [
  { keywords: ["logistic", "linear", "ridge", "lasso", "elastic net", "elasticnet"], stars: 5 },
  { keywords: ["decision tree", "k-nearest", "knn", "naive bayes"], stars: 4 },
  { keywords: ["random forest", "extra trees", "extratrees"], stars: 3 },
  { keywords: ["gradient boost", "xgboost", "xgb", "lightgbm", "lgbm", "catboost"], stars: 2 },
];

function explainabilityStars(candidateName) {
  const name = (candidateName || "").toLowerCase();
  const match = EXPLAINABILITY_KEYWORD_STARS.find((entry) => entry.keywords.some((k) => name.includes(k)));
  const n = match ? match.stars : 3;
  return "★".repeat(n) + "☆".repeat(5 - n);
}

function renderLeaderboardCondensed(run) {
  const results = run.training_results || [];
  if (!results.length) { $("leaderboard-condensed-table").innerHTML = ""; $("leaderboard-view-all-btn").classList.add("hidden"); return; }

  const metric = (run.task_spec || {}).metric;
  const bestId = (run.best_model || {}).run_id;
  const metricNames = [...new Set(results.flatMap((r) => Object.keys(r.metrics || {})))];
  const primary = metric && metricNames.includes(metric) ? metric : metricNames[0];
  const secondary = metricNames.find((m) => m !== primary);

  const showAll = results.length <= 6;
  const champion = results.find((r) => r.run_id === bestId);
  const others = results.filter((r) => r.run_id !== bestId);
  const shown = showAll ? results : [champion, ...others.slice(0, 5)].filter(Boolean);

  $("leaderboard-condensed-sub").textContent = primary ? `ranked by ${primary}` : "";
  $("leaderboard-view-all-btn").classList.toggle("hidden", showAll);

  let html = `<tr><th>Model</th>${primary ? `<th>${escapeHtml(primary)}</th>` : ""}${secondary ? `<th>${escapeHtml(secondary)}</th>` : ""}<th>Training Time</th><th>Explainability</th><th>Champion</th></tr>`;
  for (const r of shown) {
    const isBest = r.run_id === bestId;
    html += `<tr class="${isBest ? "best" : ""}">
      <td>${escapeHtml(r.candidate_name)}</td>
      ${primary ? `<td class="num">${r.metrics && primary in r.metrics ? Number(r.metrics[primary]).toFixed(3) : "—"}</td>` : ""}
      ${secondary ? `<td class="num">${r.metrics && secondary in r.metrics ? Number(r.metrics[secondary]).toFixed(3) : "—"}</td>` : ""}
      <td>${r.duration_seconds != null ? formatDuration(r.duration_seconds) : "—"}</td>
      <td class="stars" title="Approximate rating based on the model's name, not a per-run measurement">${explainabilityStars(r.candidate_name)}</td>
      <td>${isBest ? `<span class="winner-tag">★ CHAMPION</span>` : ""}</td>
    </tr>`;
  }
  $("leaderboard-condensed-table").innerHTML = html;
}

$("leaderboard-view-all-btn").addEventListener("click", () => switchRunTab("models"));
```

### Step 3: Call it from `render()`

Add after `renderJourneyCondensed(run);`:

```js
  renderJourneyCondensed(run);
  renderLeaderboardCondensed(run);
```

### Step 4: CSS

```css
.stars { color: var(--accent-warning); letter-spacing: 1px; }
```

### Step 5: Manual verification

Run with 3+ candidates: leaderboard shows champion pinned/tagged, stars column populated based on keyword matches in each candidate's name, unrecognized names default to 3 stars without erroring. Run with 8+ candidates: only 6 rows shown (champion + top 5), "View all models" visible and switches to Models Compared tab.

### Step 6: Commit

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: add condensed Model Leaderboard with explainability-star heuristic to Overview tab"
```

---

## Task 6: Why This Model? / Why Others Not Selected

**Files:**
- Modify: `frontend/index.html` (insert after leaderboard section)
- Modify: `frontend/app.js` (new `renderModelRationale(run)`)
- Modify: `frontend/styles.css` (new `.rationale-grid` rules)

**Interfaces:**
- Consumes: `run.training_results`, `run.best_model`, `run.task_spec.metric`.
- Produces: `renderModelRationale(run)`.

### Step 1: Markup

```html
        <section class="card">
          <div class="rationale-grid">
            <div>
              <h3>Why This Model?</h3>
              <ul class="callout-list" id="why-this-model-list"></ul>
            </div>
            <div>
              <h3>Why Other Models Were Not Selected?</h3>
              <div class="table-scroll"><table class="results-table" id="why-others-table"></table></div>
            </div>
          </div>
        </section>
```

### Step 2: Render function

```js
function renderModelRationale(run) {
  const best = run.best_model || {};
  const results = run.training_results || [];
  if (!best.candidate_name || results.length < 2) {
    $("why-this-model-list").innerHTML = "";
    $("why-others-table").innerHTML = "";
    return;
  }
  const metric = (run.task_spec || {}).metric;
  const lowerIsBetter = !!(best.tuning || {}).lower_is_better;
  const bestScore = metric && best.metrics ? Number(best.metrics[metric]) : null;

  const whyThis = [];
  if (bestScore != null) whyThis.push(`Highest ${escapeHtml(metric)} (${bestScore.toFixed(3)}) among all candidates`);
  const bestCv = best.cv_metrics && metric && best.cv_metrics[metric];
  if (bestCv) whyThis.push(`Stable performance across folds (CV std ${bestCv.std.toFixed(3)})`);
  const fastestId = [...results].sort((a, b) => (a.duration_seconds ?? Infinity) - (b.duration_seconds ?? Infinity))[0]?.run_id;
  if (fastestId === best.run_id) whyThis.push("Fastest training time among all candidates");
  if ((best.tuning || {}).enabled) whyThis.push(`Hyperparameters tuned over ${best.tuning.trials_done} trial(s)`);
  $("why-this-model-list").innerHTML = whyThis.map((t) => `<li>${ICONS.check}<span>${t}</span></li>`).join("") ||
    `<li class="muted small">No further rationale available for this run.</li>`;

  const others = results.filter((r) => r.run_id !== best.run_id && r.status === "succeeded");
  let html = `<tr><th>Model</th><th>Reason</th><th>Impact</th></tr>`;
  for (const r of others) {
    const delta = metric && r.metrics && bestScore != null && metric in r.metrics
      ? (lowerIsBetter ? r.metrics[metric] - bestScore : bestScore - r.metrics[metric])
      : null;
    const durRatio = best.duration_seconds && r.duration_seconds ? r.duration_seconds / best.duration_seconds : null;
    let impact = "Marginal gain";
    if (durRatio != null && durRatio > 2) impact = "High Cost";
    else if (durRatio != null && durRatio > 1.3) impact = "Medium Cost";
    // delta >= 0 always means "worse than champion", but whether that means a
    // higher or lower raw metric value depends on lowerIsBetter — don't
    // conflate delta's sign with the raw-value comparison word.
    const reason = delta != null
      ? `${(lowerIsBetter === (delta >= 0)) ? "Higher" : "Lower"} ${escapeHtml(metric)} (${Math.abs(delta).toFixed(3)} difference)${durRatio != null && durRatio > 1.3 ? " and slower training" : ""}`
      : "Did not outperform the champion";
    html += `<tr><td>${escapeHtml(r.candidate_name)}</td><td>${reason}</td><td><span class="chip flagged">${impact}</span></td></tr>`;
  }
  $("why-others-table").innerHTML = html;
}
```

### Step 3: Call it from `render()`

```js
  renderLeaderboardCondensed(run);
  renderModelRationale(run);
```

### Step 4: CSS

```css
.rationale-grid { display: grid; grid-template-columns: 1fr 1fr; gap: var(--sp-4); }
@media (max-width: 900px) { .rationale-grid { grid-template-columns: 1fr; } }
```

### Step 5: Manual verification

Run with ≥2 succeeded candidates: "Why This Model" lists concrete, real bullet points (no fabricated text); "Why Others" table has one row per non-champion candidate with a delta-based reason and a heuristic Impact chip. Run with exactly 1 candidate: both panels render an empty/graceful state, no crash.

### Step 6: Commit

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: add deterministic Why-This-Model / Why-Others-Not-Selected panels to Overview tab"
```

---

## Task 7: "What AI Did During This Run" checklist

**Files:**
- Modify: `frontend/index.html` (insert after rationale section, before the existing `dash-grid`/report-lede content)
- Modify: `frontend/app.js` (new `renderPipelineActions(run)`)
- Modify: `frontend/styles.css` (reuse `.callout-list`, no new rules needed)

**Interfaces:**
- Consumes: `run.feature_plan.steps` (`FeatureStep[]`, `src/state.py:31-52` — each has `op`, `columns`, `rationale`), `run.training_results[].resampling_applied` (per-candidate field, `TrainingResult`, `src/state.py:119` — **not** a top-level `run` field; there is no top-level `resampling_applied` in the API response, only `resampling_plan`/`resampling_suggestion` — mirror the existing lookup pattern used in `renderResults()`, `app.js:1757`: `results.find((r) => r.resampling_applied)`), `run.feature_selection` (confirmed top-level field name in `_run_summary()`, `src/api/server.py:404` — **not** `run.feature_selection_result`).
- Produces: `renderPipelineActions(run)`.

### Step 1: Markup

```html
        <div class="card hidden" id="pipeline-actions-card">
          <div class="card-head"><h3>What AI Did During This Run</h3></div>
          <ul class="callout-list" id="pipeline-actions-list"></ul>
        </div>
```

### Step 2: Render function

```js
const FEATURE_OP_LABELS = {
  impute: "Imputed missing values",
  encode: "Encoded categorical values",
  scale: "Standardized numerical features",
  bin: "Binned continuous values",
  datetime_decompose: "Decomposed datetime columns",
  drop: "Removed columns",
  custom_code: "Applied a custom transformation",
};

function renderPipelineActions(run) {
  const steps = ((run.feature_plan || {}).steps) || [];
  const card = $("pipeline-actions-card");
  const items = [];

  for (const step of steps) {
    const label = FEATURE_OP_LABELS[step.op] || step.op;
    const cols = (step.columns || []).slice(0, 3).join(", ") + (step.columns && step.columns.length > 3 ? ", …" : "");
    items.push(`<li>${ICONS.check}<span><strong>${escapeHtml(label)}</strong>${cols ? ` — ${escapeHtml(cols)}` : ""}</span></li>`);
  }
  const resamplingApplied = (run.training_results || []).find((r) => r.resampling_applied)?.resampling_applied;
  if (resamplingApplied) {
    items.push(`<li>${ICONS.check}<span><strong>Applied ${escapeHtml(resamplingApplied.replaceAll("_", " "))}</strong> to correct class imbalance</span></li>`);
  }
  const fs = run.feature_selection;
  if (fs && fs.n_features_selected != null) {
    items.push(`<li>${ICONS.check}<span><strong>Feature selection</strong> kept ${fs.n_features_selected} of ${fs.n_features_total} features</span></li>`);
  }

  if (!items.length) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  $("pipeline-actions-list").innerHTML = items.join("");
}
```

### Step 3: Call it from `render()`

```js
  renderModelRationale(run);
  renderPipelineActions(run);
```

### Step 4: Manual verification

Run with a non-trivial feature plan (imputation + encoding + scaling), resampling enabled, and feature selection enabled: checklist shows one line per action with real column names, capped at 3 names + ellipsis for wide plans. Run with an empty feature plan: card hides entirely (no empty card shown).

### Step 5: Commit

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: add What-AI-Did-During-This-Run checklist to Overview tab"
```

---

## Task 8: Right rail restyle — Next Steps grid + AI Summary / Ask AI polish

**Files:**
- Modify: `frontend/index.html` (`#run-rail` aside: add Next Steps grid; restyle `#report-lede`/`#assistant-card` wrapper markup)
- Modify: `frontend/app.js` (new `renderNextSteps` — mostly static, no new data needed; minor tweak to where `report-lede` is read)
- Modify: `frontend/styles.css` (new `.next-steps-grid`, `.ai-summary-card`, chat pinned-input rules)

**Interfaces:**
- Consumes: nothing new — reuses `run.report` (already read by existing `renderReport`), existing `#assistant-card`/`#chat-thread`/`#chat-suggestions` (unchanged logic).
- Produces: static Next Steps button grid wired to `switchRunTab`/existing buttons; visual restyle only for AI Summary and Ask AI.

### Step 1: Wrap `report-lede`/`report-body` as an "AI Summary" card

In `frontend/index.html`, find the relocated (Task 1 Step 4) block, now the first children of `<aside class="run-rail" id="run-rail">`:

```html
      <aside class="run-rail" id="run-rail">
        <div id="report-lede" class="report-lede hidden"></div>
        <div id="report-body" class="report-body"></div>
```

Replace with:

```html
      <aside class="run-rail" id="run-rail">
        <div class="card hidden ai-summary-card" id="ai-summary-card">
          <div class="card-head"><h3>${ICONS.sparkle} AI Summary</h3></div>
          <div id="report-lede" class="report-lede hidden"></div>
          <div id="report-body" class="report-body"></div>
        </div>
```

(As in Task 3 Step 1, replace the literal `${ICONS.sparkle}` placeholder with the actual inline sparkle `<svg>` — copy it from `ICONS.sparkle` in `app.js:25`: `<svg class="icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2 14 9 21 12 14 15 12 22 10 15 3 12 10 9Z"/></svg>`.)

**Note:** Task 1's implementer found that the brief's original description of this step didn't account for the fact that `#report-card` (and the `$("report-card")` reference inside `renderReport`) had to be removed *during Task 1 itself* — leaving it would have thrown on every run load. So `renderReport` currently (as of Task 1's commit) looks like this, with no card-hiding at all:

```js
function renderReport(run) {
  if (!run.report) return;

  const lines = run.report.split("\n").filter((l) => l.trim());
```

Now that `#ai-summary-card` exists (from Step 1 above), restore proper show/hide behavior keyed on it:

```js
function renderReport(run) {
  const card = $("ai-summary-card");
  if (!run.report) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  const lines = run.report.split("\n").filter((l) => l.trim());
```

(Leave the rest of `renderReport` — the `report-lede`/`report-body`/download-link logic — unchanged; only add the `card` show/hide lines shown above.)

### Step 2: Next Steps grid markup

In `frontend/index.html`, inside `<aside class="run-rail" id="run-rail">`, insert a new card right before `#assistant-card`:

```html
        <div class="card" id="next-steps-card">
          <div class="card-head"><h3>Next Steps</h3></div>
          <div class="next-steps-grid">
            <button type="button" class="next-step-btn" id="nextstep-compare-btn">Compare Models</button>
            <button type="button" class="next-step-btn" id="nextstep-artifacts-btn">Download Artifacts</button>
            <button type="button" class="next-step-btn" id="nextstep-share-btn">Share Report</button>
            <span class="next-step-btn disabled" title="Not available in this local build">View SHAP Report</span>
            <span class="next-step-btn disabled" title="Not available in this local build">Deploy Model</span>
            <span class="next-step-btn disabled" title="Not available in this local build">Generate API</span>
            <span class="next-step-btn disabled" title="Not available in this local build">Schedule Retraining</span>
            <span class="next-step-btn disabled" title="Not available in this local build">Monitor Model</span>
          </div>
        </div>
```

### Step 3: Wire the three real Next Steps buttons

In `frontend/app.js`, add near the other top-level listeners:

```js
$("nextstep-compare-btn").addEventListener("click", () => switchRunTab("models"));
$("nextstep-artifacts-btn").addEventListener("click", () => switchRunTab("artifacts"));
$("nextstep-share-btn").addEventListener("click", () => $("share-btn").click());
```

### Step 4: CSS

```css
.next-steps-grid { display: grid; grid-template-columns: 1fr 1fr; gap: var(--sp-2); }
.next-step-btn {
  font: inherit; font-size: var(--text-xs); font-weight: 600; text-align: center;
  padding: var(--sp-2); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);
  background: var(--bg-surface-raised); color: var(--text-primary); cursor: pointer;
}
.next-step-btn:hover:not(.disabled) { border-color: var(--accent-primary); color: var(--accent-primary); }
.next-step-btn.disabled { opacity: 0.5; cursor: not-allowed; }
.ai-summary-card { background: var(--accent-primary-soft); }
.chat-form { position: sticky; bottom: 0; background: var(--bg-surface); padding-top: var(--sp-2); }
.chat-suggestions .suggestion-chip { border-radius: 999px; }
```

### Step 5: Manual verification

Right rail shows "Next Steps" grid above the AI Assistant chat; the three real buttons (Compare Models, Download Artifacts, Share Report) work; the five disabled ones are visibly inert with the standard "Not available" tooltip. AI Summary card shows the same narrative text as before, just restyled. Chat panel unchanged functionally (send a question, get a response) but the input row now sits pinned at the bottom of the card.

### Step 6: Commit

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: restyle AI Summary card and add Next Steps grid to Overview right rail"
```

---

## Task 9: Full manual verification pass

**Files:** none (verification only)

- [ ] **Step 1: Start the app locally**

Run: whatever the existing local entrypoint is (`python run_server.py` per the repo root, or the project's documented dev command).

- [ ] **Step 2: Walk through a completed run**

Open a run that finished successfully with ≥3 candidates, tuning enabled, resampling enabled, and feature selection enabled. Click through all 7 tabs. Confirm every widget from Tasks 1-8 renders with real numbers, no console errors (check browser dev tools console).

- [ ] **Step 3: Walk through an in-progress run**

Start a new run and watch it live. Confirm Overview still renders sensibly with partial data (fewer candidates, no champion yet) — no thrown exceptions during the 1500ms poll cycle. Confirm the right rail's live "reasoning rail" behavior (separate from the new Logs tab) still works as before.

- [ ] **Step 4: Walk through a failed run**

Open (or induce, e.g. via a fixture that trips a retry cap) a run with `status: "failed"`. Confirm the champion banner hides gracefully if there's no best model, the leaderboard/rationale panels degrade gracefully, and the existing `#error-card` still shows.

- [ ] **Step 5: Dark mode pass**

Toggle dark mode (existing `#theme-toggle`). Re-check champion banner, journey timeline, leaderboard, next-steps grid all remain legible (no hardcoded light-only colors were introduced — everything above uses `var(--...)` tokens).

- [ ] **Step 6: Regression check on unrelated views**

Confirm the Datasets list view, Dataset Detail view, and the intake/new-experiment flow are all unaffected (breadcrumb hidden appropriately, no leftover `run-view` tab bar bleeding into other views).

- [ ] **Step 7: Final commit (if any fixes were needed)**

```bash
git add -A
git commit -m "fix: address issues found in Model Summary Overview manual verification pass"
```

(Only run this if Steps 2-6 surfaced fixes — if everything passed clean, there's nothing to commit here.)
