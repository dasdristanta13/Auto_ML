# Experiments Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new "Experiments" tab to the run view — a visual, dashboard-style summary (KPI cards, bar chart, multi-series trial-trend chart, a consolidated experiments table, distribution donuts, and a best-experiment side panel) adapted from a reference AutoML.ai screenshot to this app's real per-run data model.

**Architecture:** `frontend/` is a static, no-build vanilla-JS/HTML/CSS app (`frontend/index.html`, `frontend/app.js`, `frontend/styles.css`) served directly by the FastAPI backend and polled every 1.5s via `GET /api/runs/{id}`. All new UI is derived from the already-in-memory `run` object — no new backend endpoints. Charts are hand-built inline SVG/CSS (no charting library), following the same pattern already used for the existing donut and tuning-trend charts.

**Tech Stack:** Vanilla JS (global scope, no modules, no bundler), plain CSS custom properties for theming (light/dark), inline SVG for charts.

## Global Constraints

- Raw data never enters an LLM context window — not relevant here (pure frontend rendering of already-fetched run JSON), but do not add any new code path that sends dataset rows into a prompt.
- No new backend endpoints or `PipelineState` fields — everything needed already exists on the run JSON returned by `GET /api/runs/{run_id}`.
- No per-candidate drill-down route/modal — the reference image's row-level actions menu is intentionally dropped (see spec's Non-goals).
- Categorical color assignment must use the dataviz skill's fixed-order, six-checks-validated palette — never eyeball or cycle hues. This plan's Task 3 pre-validates the palette extension needed (see below); later tasks consume the resulting CSS variables by name.
- No JS test runner exists in this repo (`frontend/` has no `package.json`/test framework — confirmed via search). Verification for every task is manual: load the app via the `run` skill, open browser devtools, and either (a) call the new render function directly against a hand-built fixture `run` object pasted into the console, or (b) drive a real run to the relevant state. Exact fixture objects and exact expected DOM assertions are given in each task — this is not a placeholder for "test it somehow."
- Reference spec: `docs/superpowers/plans/2026-07-08-experiments-tab-design.md`.

---

## Palette pre-work (context for Task 3 and Task 6 — not a standalone task)

The existing categorical palette (`--cat-1..4` in `frontend/styles.css:30-33` light / `:79-82` dark) has only 4 slots, validated together per the file's header comment. The new multi-series trend chart (Task 4) and "By Model" donut (Task 6) need up to 8 named identities (typical run has 5-10 candidates; the 9th+ folds into a shared "Other" muted color per the dataviz skill's non-negotiable rule — never generate a 9th hue). I ran the dataviz skill's validator to extend the existing 4 slots to 8 without touching slots 1-4 (already used elsewhere: dataset-summary donut, class-distribution donut, tuning chart lines):

```
node <dataviz-skill>/scripts/validate_palette.js "#7c3aed,#0284c7,#059669,#d97706,#dc2626,#db2777,#0d9488,#ea580c" --mode light --surface "#ffffff"
→ ALL CHECKS PASS (worst adjacent CVD ΔE 14.5)

node <dataviz-skill>/scripts/validate_palette.js "#8b5cf6,#0284c7,#059669,#d97706,#ef4444,#ec4899,#0d9488,#ea580c" --mode dark --surface "#1a1d24"
→ ALL CHECKS PASS (worst adjacent CVD ΔE 9.4 — floor band, legal with secondary encoding: this plan's charts always pair color with a legend + direct label on the champion, satisfying that requirement)
```

Final validated slots (Task 3 adds these as new CSS variables — do not invent different hex values):

| Slot | Hue | Light | Dark |
|---|---|---|---|
| `--cat-5` | red | `#dc2626` | `#ef4444` |
| `--cat-6` | magenta | `#db2777` | `#ec4899` |
| `--cat-7` | teal | `#0d9488` | `#0d9488` |
| `--cat-8` | orange | `#ea580c` | `#ea580c` |

---

## Task 1: Tab shell — HTML, RUN_TABS wiring, and empty panel

**Files:**
- Modify: `frontend/index.html:189-196` (tab bar), `frontend/index.html:494-495` (insert new panel after `tab-models-panel`, before `tab-explainability-panel`)
- Modify: `frontend/app.js:2376` (`RUN_TABS`), `frontend/app.js:1096` (add call in `render(run)`)

**Interfaces:**
- Produces: `renderExperimentsTab(run)` — orchestrator function, called from `render(run)`. Tasks 2-7 each add one call inside this function's body and one section of markup inside `#tab-experiments-panel`. Consumes: nothing yet (stub only in this task).

- [ ] **Step 1: Add the tab button**

In `frontend/index.html`, in the `.run-tab-bar` block, insert a new button between the Pipeline and Models Compared buttons:

```html
        <button class="tab-btn" id="tab-pipeline-btn" type="button" role="tab" aria-selected="false">Pipeline</button>
        <button class="tab-btn" id="tab-experiments-btn" type="button" role="tab" aria-selected="false">Experiments</button>
        <button class="tab-btn" id="tab-models-btn" type="button" role="tab" aria-selected="false">Models Compared</button>
```

- [ ] **Step 2: Add the empty panel**

Immediately after the closing `</div><!-- /tab-models-panel -->` line (`frontend/index.html:495`) and before `<div id="tab-explainability-panel" class="hidden">`, insert:

```html
      <div id="tab-experiments-panel" class="hidden">
        <div class="experiments-layout">
          <div class="experiments-main">
            <section class="stat-row" id="exp-stat-cards"></section>
          </div>
          <aside class="card" id="exp-best-panel"></aside>
        </div>
      </div><!-- /tab-experiments-panel -->
```

(This is deliberately minimal — Tasks 2-7 fill in `exp-stat-cards` content and add the bar chart / trend chart / table / donuts markup inside `.experiments-main`, and the side-panel content inside `#exp-best-panel`.)

- [ ] **Step 3: Wire the tab into `RUN_TABS` and `render()`**

In `frontend/app.js:2376`, change:

```javascript
const RUN_TABS = ["overview", "pipeline", "models", "explainability", "artifacts", "logs"];
```

to:

```javascript
const RUN_TABS = ["overview", "pipeline", "experiments", "models", "explainability", "artifacts", "logs"];
```

In `frontend/app.js`, immediately after the `renderTuningTrend(run);` line (`app.js:1096`), add:

```javascript
  renderExperimentsTab(run);
```

Then define the stub function near the other tab-panel render functions (add just above `/* ================= results table ================= */` at `app.js:2011`):

```javascript
/* ================= experiments tab ================= */

function renderExperimentsTab(run) {
  // Tasks 2-7 fill this in: stat cards, bar chart, trend chart, table, donuts, side panel.
}
```

- [ ] **Step 4: Manual verification**

Start the app with the `run` skill. Open the app in a browser, open any run (or start a new one — the tab must appear even before training finishes, since it's a plain tab like the others). Confirm:
- An "Experiments" tab button appears between "Pipeline" and "Models Compared".
- Clicking it shows an (empty) panel and hides the others; clicking any other tab hides it again.
- No console errors on load or on tab switch.

- [ ] **Step 5: Commit**

```bash
git add frontend/index.html frontend/app.js
git commit -m "feat: add empty Experiments tab shell to the run view"
```

---

## Task 2: KPI stat cards

**Files:**
- Modify: `frontend/app.js` (add `renderExperimentsStatCards`, call it from `renderExperimentsTab`)

**Interfaces:**
- Consumes: `run.training_results` (`TrainingResult[]`: `candidate_name`, `duration_seconds`, `cv_metrics`, `metrics`, `run_id`, `tuning.trials_done`), `run.best_model`, `run.task_spec.metric`.
- Produces: fills `#exp-stat-cards` (already in DOM from Task 1).

- [ ] **Step 1: Write the fixture and expected behavior (manual test, no runner — see Global Constraints)**

Fixture to paste in devtools console (reused by Tasks 2-7 — keep it around in a scratch file, e.g. `C:\Users\ASUS\AppData\Local\Temp\claude\...\scratchpad\exp-fixture.js`, not committed):

```javascript
const FIXTURE_RUN = {
  run_id: "r1",
  created_at: 1737300000, // fixed epoch for reproducible "Start Time" checks
  task_spec: { metric: "roc_auc", task_type: "classification", target_column: "churned" },
  cv_config: { enabled: true, requested_folds: 5 },
  resampling_plan: { enabled: true, method: "smote" },
  best_model: {
    run_id: "c1", candidate_name: "Logistic Regression", status: "succeeded",
    metrics: { roc_auc: 0.992, accuracy: 0.968 },
    cv_metrics: { roc_auc: { mean: 0.992, std: 0.003 }, accuracy: { mean: 0.968, std: 0.006 } },
    duration_seconds: 18, cv_folds: 5, resampling_applied: "smote",
    tuning: { enabled: true, trials_total: 15, trials_done: 15, metric: "roc_auc", lower_is_better: false,
      history: Array.from({ length: 15 }, (_, i) => ({ trial: i, score: 0.94 + i * 0.0035, best_score: 0.94 + i * 0.0035 })) },
  },
  training_results: [
    { run_id: "c1", candidate_name: "Logistic Regression", status: "succeeded",
      metrics: { roc_auc: 0.992, accuracy: 0.968 }, cv_metrics: { roc_auc: { mean: 0.992, std: 0.003 } },
      duration_seconds: 18, cv_folds: 5, resampling_applied: "smote",
      tuning: { enabled: true, trials_total: 15, trials_done: 15, metric: "roc_auc", lower_is_better: false,
        history: Array.from({ length: 15 }, (_, i) => ({ trial: i, score: 0.94 + i * 0.0035, best_score: 0.94 + i * 0.0035 })) } },
    { run_id: "c2", candidate_name: "CatBoost", status: "succeeded",
      metrics: { roc_auc: 0.889, accuracy: 0.956 }, cv_metrics: { roc_auc: { mean: 0.889, std: 0.004 } },
      duration_seconds: 132, cv_folds: 5, resampling_applied: "smote",
      tuning: { enabled: true, trials_total: 20, trials_done: 20, metric: "roc_auc", lower_is_better: false,
        history: Array.from({ length: 20 }, (_, i) => ({ trial: i, score: 0.85 + i * 0.002, best_score: 0.85 + i * 0.002 })) } },
    { run_id: "c3", candidate_name: "XGBoost", status: "succeeded",
      metrics: { roc_auc: 0.888, accuracy: 0.952 }, cv_metrics: { roc_auc: { mean: 0.888, std: 0.005 } },
      duration_seconds: 258, cv_folds: 5, resampling_applied: "smote",
      tuning: { enabled: false, trials_total: 0, trials_done: 0, history: [] } },
    { run_id: "c4", candidate_name: "Decision Tree", status: "failed", error: "ValueError: input contains NaN",
      metrics: {}, cv_metrics: {}, duration_seconds: 4, cv_folds: 0,
      tuning: { enabled: false, trials_total: 0, trials_done: 0, history: [] } },
  ],
};
```

- [ ] **Step 2: Implement `renderExperimentsStatCards`**

Add to `frontend/app.js` inside `renderExperimentsTab`:

```javascript
function renderExperimentsTab(run) {
  renderExperimentsStatCards(run);
}

function renderExperimentsStatCards(run) {
  const results = run.training_results || [];
  const best = run.best_model || {};
  const metric = (run.task_spec || {}).metric;
  const totalTrials = results.reduce((sum, r) => sum + (r.tuning?.trials_done || 1), 0);
  const avgDuration = results.length
    ? results.reduce((sum, r) => sum + (r.duration_seconds || 0), 0) / results.length
    : 0;
  const totalDuration = results.reduce((sum, r) => sum + (r.duration_seconds || 0), 0);
  const bestCv = metric && best.cv_metrics && best.cv_metrics[metric];
  const bestScore = bestCv ? bestCv.mean : (metric && best.metrics && metric in best.metrics ? Number(best.metrics[metric]) : null);

  const cards = [
    { icon: "layers", tint: "violet", label: "Total Experiments", value: String(totalTrials), sub: "trials across all candidates" },
    { icon: "grid", tint: "violet", label: "Models Evaluated", value: String(results.length), sub: "unique candidates" },
    { icon: "trophy", tint: "amber", label: `Best ${metric ? metric.toUpperCase() : "Score"}${bestCv ? " (CV)" : ""}`,
      value: bestScore != null ? bestScore.toFixed(3) : "—", sub: escapeHtml(best.candidate_name || "—") },
    { icon: "clock", tint: "green", label: "Avg. Training Time", value: formatDuration(avgDuration), sub: "per candidate" },
    { icon: "cpu", tint: "green", label: "Total Compute Time", value: formatDuration(totalDuration), sub: "total wall time" },
  ];

  $("exp-stat-cards").innerHTML = cards
    .map(
      (c) => `
      <div class="stat-card">
        <span class="stat-icon ${c.tint}">${ICONS[c.icon]}</span>
        <div class="stat-body">
          <div class="stat-label">${escapeHtml(c.label)}</div>
          <div class="stat-value">${c.value}</div>
          <div class="stat-sub">${c.sub}</div>
        </div>
      </div>`
    )
    .join("");
}
```

- [ ] **Step 3: Manual verification**

In the browser devtools console (app open, any run loaded so `$` and helpers exist), paste the `FIXTURE_RUN` object from Step 1, then run:

```javascript
renderExperimentsTab(FIXTURE_RUN);
document.querySelectorAll("#exp-stat-cards .stat-card").length
```

Expected: `5`. Then check values:

```javascript
[...document.querySelectorAll("#exp-stat-cards .stat-value")].map(e => e.textContent)
```

Expected: `["58", "4", "0.992", "1m 43s", "6m 52s"]` (58 = 15+20+1+1 trials; avg duration = (18+132+258+4)/4 = 103s = "1m 43s"; total duration = 412s = "6m 52s").

- [ ] **Step 4: Commit**

```bash
git add frontend/app.js
git commit -m "feat: add KPI stat cards to the Experiments tab"
```

---

## Task 3: Palette extension + Model Performance bar chart

**Files:**
- Modify: `frontend/styles.css:29-34` (light `--cat-5..8`), `:78-83` (dark `--cat-5..8`), append new `.exp-bar-*` rules
- Modify: `frontend/app.js` (add `renderExperimentsBarChart`, call from `renderExperimentsTab`)
- Modify: `frontend/index.html` (add bar chart card markup inside `.experiments-main`)

**Interfaces:**
- Consumes: same `run.training_results` / `run.best_model` / `run.task_spec.metric` shape as Task 2.
- Produces: fills a new `#exp-bar-chart` container. Does not yet need `--cat-5..8` (bar chart is binary champion/other) — those variables are added here because Task 3 is where the palette pre-work (see above) lands in the codebase, and Task 4 (next) consumes them immediately.

- [ ] **Step 1: Add the validated palette variables**

In `frontend/styles.css`, in the light `:root` block, change:

```css
  --cat-1: #7c3aed;
  --cat-2: #0284c7;
  --cat-3: #059669;
  --cat-4: #d97706;
```

to:

```css
  --cat-1: #7c3aed;
  --cat-2: #0284c7;
  --cat-3: #059669;
  --cat-4: #d97706;
  --cat-5: #dc2626;
  --cat-6: #db2777;
  --cat-7: #0d9488;
  --cat-8: #ea580c;
```

In the dark `:root[data-theme="dark"]` block, change:

```css
  --cat-1: #8b5cf6;
  --cat-2: #0284c7;
  --cat-3: #059669;
  --cat-4: #d97706;
```

to:

```css
  --cat-1: #8b5cf6;
  --cat-2: #0284c7;
  --cat-3: #059669;
  --cat-4: #d97706;
  --cat-5: #ef4444;
  --cat-6: #ec4899;
  --cat-7: #0d9488;
  --cat-8: #ea580c;
```

- [ ] **Step 2: Add the bar chart card markup**

In `frontend/index.html`, inside `.experiments-main` (added in Task 1), after the `#exp-stat-cards` section, add:

```html
            <div class="card">
              <div class="card-head"><h3>Model Performance Overview</h3><span class="muted small" id="exp-bar-sub"></span></div>
              <div class="exp-bar-chart" id="exp-bar-chart"></div>
              <ul class="exp-bar-legend">
                <li><span class="swatch" style="background:var(--accent-primary)"></span>Best Model</li>
                <li><span class="swatch" style="background:var(--border-subtle)"></span>Other Models</li>
              </ul>
            </div>
```

- [ ] **Step 3: Add the bar chart CSS**

Append to `frontend/styles.css`:

```css
/* ================= experiments: bar chart ================= */

.exp-bar-chart { display: flex; align-items: flex-end; gap: var(--sp-3); height: 200px; padding-top: 24px; }
.exp-bar-col { display: flex; flex-direction: column; align-items: center; justify-content: flex-end; flex: 1 1 0; min-width: 0; height: 100%; }
.exp-bar-value { font-size: var(--text-xs); font-family: var(--font-mono); font-weight: 650; margin-bottom: 4px; }
.exp-bar-track { width: 100%; max-width: 56px; height: 100%; display: flex; align-items: flex-end; }
.exp-bar-fill { width: 100%; border-radius: 4px 4px 0 0; background: var(--border-subtle); min-height: 2px; }
.exp-bar-fill.champion { background: var(--accent-primary); }
.exp-bar-name { font-size: var(--text-xs); color: var(--text-secondary); margin-top: 6px; text-align: center; overflow-wrap: break-word; max-width: 90px; }
.exp-bar-legend { list-style: none; padding: 0; display: flex; gap: var(--sp-4); font-size: var(--text-xs); color: var(--text-secondary); margin-top: var(--sp-2); }
.exp-bar-legend li { display: flex; align-items: center; gap: 6px; }
.exp-bar-legend .swatch { width: 11px; height: 11px; border-radius: 3px; flex-shrink: 0; }
```

- [ ] **Step 4: Implement `renderExperimentsBarChart`**

Add to `frontend/app.js`:

```javascript
function renderExperimentsBarChart(run) {
  const results = (run.training_results || []).filter((r) => r.status === "succeeded");
  const metric = (run.task_spec || {}).metric;
  const bestId = (run.best_model || {}).run_id;
  const card = $("exp-bar-chart").closest(".card");
  if (!results.length || !metric) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  const lowerIsBetter = metric === "rmse" || metric === "mae";
  const withMetric = results.filter((r) => r.metrics && metric in r.metrics);
  const sorted = [...withMetric].sort((a, b) =>
    lowerIsBetter ? a.metrics[metric] - b.metrics[metric] : b.metrics[metric] - a.metrics[metric]
  );
  $("exp-bar-sub").textContent = `ranked by ${metric}`;

  const values = sorted.map((r) => Number(r.metrics[metric]));
  const allUnitRange = values.every((v) => v >= 0 && v <= 1);
  const scaleMax = allUnitRange ? 1 : Math.max(...values) * 1.15;

  $("exp-bar-chart").innerHTML = sorted
    .map((r) => {
      const value = Number(r.metrics[metric]);
      const pct = Math.max((value / scaleMax) * 100, 1);
      const isBest = r.run_id === bestId;
      return `
      <div class="exp-bar-col" title="${escapeHtml(r.candidate_name)}: ${value.toFixed(3)}">
        <span class="exp-bar-value">${value.toFixed(3)}</span>
        <div class="exp-bar-track"><div class="exp-bar-fill ${isBest ? "champion" : ""}" style="height:${pct.toFixed(1)}%"></div></div>
        <span class="exp-bar-name">${escapeHtml(r.candidate_name)}</span>
      </div>`;
    })
    .join("");
}
```

Then add the call inside `renderExperimentsTab`:

```javascript
function renderExperimentsTab(run) {
  renderExperimentsStatCards(run);
  renderExperimentsBarChart(run);
}
```

- [ ] **Step 5: Manual verification**

Run the palette validator to confirm the committed values still pass (guards against a future edit accidentally drifting the hex codes):

```bash
node "<dataviz-skill-base>/scripts/validate_palette.js" "#7c3aed,#0284c7,#059669,#d97706,#dc2626,#db2777,#0d9488,#ea580c" --mode light --surface "#ffffff"
node "<dataviz-skill-base>/scripts/validate_palette.js" "#8b5cf6,#0284c7,#059669,#d97706,#ef4444,#ec4899,#0d9488,#ea580c" --mode dark --surface "#1a1d24"
```

Expected: both print `ALL CHECKS PASS`.

In the browser console with `FIXTURE_RUN` from Task 2:

```javascript
renderExperimentsTab(FIXTURE_RUN);
document.querySelectorAll("#exp-bar-chart .exp-bar-col").length
```

Expected: `3` (the failed "Decision Tree" candidate has no `roc_auc` in `metrics` in the fixture and is filtered out by `status === "succeeded"`). Then:

```javascript
document.querySelector("#exp-bar-chart .exp-bar-col:first-child .exp-bar-name").textContent
```

Expected: `"Logistic Regression"` (highest roc_auc, sorted first), and:

```javascript
document.querySelector("#exp-bar-chart .exp-bar-fill.champion") !== null
```

Expected: `true`. Also toggle the OS/browser dark mode (or run `document.documentElement.setAttribute("data-theme","dark")` if the app exposes that toggle) and visually confirm bars remain legible.

- [ ] **Step 6: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: extend categorical palette to 8 slots and add the Model Performance bar chart"
```

---

## Task 4: Experiment Trend multi-series line chart (retires the old single-model tuning chart)

**Files:**
- Modify: `frontend/index.html:471-495` (remove `#tuning-card` from `tab-models-panel`; add trend chart card inside `.experiments-main`)
- Modify: `frontend/app.js` (remove `renderTuningTrend` and its call site; add `renderExperimentsTrend`)
- Modify: `frontend/styles.css` (extend `.tt-*` rules to be reusable per-series; the existing rules assume exactly 2 lines, one per role — this task generalizes them)

**Interfaces:**
- Consumes: `run.training_results[].tuning` (`{enabled, history: [{trial, score, best_score}]}`), `--cat-1..8` from Task 3.
- Produces: fills `#exp-trend-chart` + `#exp-trend-legend`.

- [ ] **Step 1: Remove the old tuning card**

In `frontend/index.html`, inside `tab-models-panel` (`index.html:471-495`), delete this entire block:

```html
          <div class="card hidden" id="tuning-card">
            <div class="card-head">
              <h3>Hyperparameter tuning trend</h3>
              <span class="muted small" id="tuning-sub"></span>
            </div>
            <div id="tuning-chart"></div>
            <ul class="tuning-legend" id="tuning-legend"></ul>
          </div>
```

leaving just the `#results-card` inside `tab-models-panel`'s `.dash-grid`.

In `frontend/app.js`, delete the `renderTuningTrend(run);` call (`app.js:1096`) and the entire `renderTuningTrend` function (`app.js:2090-2149`).

- [ ] **Step 2: Add the trend chart card markup**

In `frontend/index.html`, inside `.experiments-main`, after the bar-chart card added in Task 3, add:

```html
            <div class="card">
              <div class="card-head"><h3>Experiment Trend</h3><span class="muted small" id="exp-trend-sub"></span></div>
              <div id="exp-trend-chart"></div>
              <ul class="tuning-legend" id="exp-trend-legend"></ul>
              <p class="muted small hidden" id="exp-trend-empty">No candidates had hyperparameter tuning enabled for this run, so there's no trial-by-trial trend to show.</p>
            </div>
```

- [ ] **Step 3: Generalize the `.tt-*` CSS for per-series color**

In `frontend/styles.css`, replace the existing tuning-chart rules block:

```css
/* tuning trend chart (score per Optuna trial + best-so-far line) */
.tt-grid { stroke: var(--border-subtle); stroke-width: 1; }
.tt-axis { fill: var(--text-secondary); font-size: 10px; font-family: var(--font-mono); }
.tt-final { fill: var(--text-primary); font-size: 11px; font-weight: 650; font-family: var(--font-mono); }
.tt-score-line { stroke: var(--cat-1); stroke-width: 2; stroke-linejoin: round; }
.tt-best-line { stroke: var(--cat-2); stroke-width: 2; stroke-dasharray: 5 4; stroke-linejoin: round; }
.tt-dot { fill: var(--cat-1); stroke: var(--bg-surface); stroke-width: 2; }
.tuning-legend { list-style: none; padding: 0; margin-top: var(--sp-2); display: flex; gap: var(--sp-4); font-size: var(--text-xs); color: var(--text-secondary); }
.tuning-legend li { display: flex; align-items: center; gap: 6px; }
.tt-chip { width: 14px; height: 0; border-top: 3px solid; border-radius: 2px; }
.tt-chip-score { border-color: var(--cat-1); }
.tt-chip-best { border-color: var(--cat-2); border-top-style: dashed; }
```

with:

```css
/* experiment trend chart (best-so-far score per Optuna trial, one line per candidate) */
.tt-grid { stroke: var(--border-subtle); stroke-width: 1; }
.tt-axis { fill: var(--text-secondary); font-size: 10px; font-family: var(--font-mono); }
.tt-final { fill: var(--text-primary); font-size: 11px; font-weight: 650; font-family: var(--font-mono); }
.tt-line { stroke-width: 2; stroke-linejoin: round; fill: none; }
.tt-line.champion { stroke-width: 3; }
.tt-line.other-fold { stroke: var(--text-secondary); opacity: 0.5; }
.tt-dot { stroke: var(--bg-surface); stroke-width: 2; }
.tuning-legend { list-style: none; padding: 0; margin-top: var(--sp-2); display: flex; gap: var(--sp-3); font-size: var(--text-xs); color: var(--text-secondary); flex-wrap: wrap; }
.tuning-legend li { display: flex; align-items: center; gap: 6px; }
.tt-chip { width: 14px; height: 3px; border-radius: 2px; flex-shrink: 0; }
```

(`.tt-line.champion` gets a thicker stroke so it stays visually distinct even among 8 colors; `.tt-line.other-fold` / a shared muted color is for the 9th+ candidate fold-in case in Step 4.)

- [ ] **Step 4: Implement `renderExperimentsTrend`**

Add to `frontend/app.js`, in place of the deleted `renderTuningTrend`:

```javascript
/* ================= experiment trend chart ================= */

const EXP_TREND_COLOR_KEYS = ["--cat-1", "--cat-2", "--cat-3", "--cat-4", "--cat-5", "--cat-6", "--cat-7", "--cat-8"];

function renderExperimentsTrend(run) {
  const card = $("exp-trend-chart").closest(".card");
  const results = run.training_results || [];
  const bestId = (run.best_model || {}).run_id;
  const tuned = results.filter((r) => r.tuning && r.tuning.enabled && (r.tuning.history || []).length > 1);

  $("exp-trend-empty").classList.toggle("hidden", tuned.length > 0);
  $("exp-trend-chart").classList.toggle("hidden", tuned.length === 0);
  $("exp-trend-legend").classList.toggle("hidden", tuned.length === 0);
  if (!tuned.length) {
    $("exp-trend-sub").textContent = "";
    return;
  }

  const named = tuned.slice(0, 8);
  const overflow = tuned.slice(8);
  $("exp-trend-sub").textContent = `best-so-far score per trial · ${tuned.length} of ${results.length} candidate(s) shown`;

  const styles = getComputedStyle(document.documentElement);
  const palette = EXP_TREND_COLOR_KEYS.map((k) => styles.getPropertyValue(k).trim());

  const W = 680, H = 240, padL = 50, padR = 16, padT = 14, padB = 30;
  const maxTrials = Math.max(...tuned.map((r) => r.tuning.history.length));
  const allScores = tuned.flatMap((r) => r.tuning.history.map((h) => h.best_score));
  let lo = Math.min(...allScores), hi = Math.max(...allScores);
  if (hi - lo < 1e-9) { hi += 0.001; lo -= 0.001; }
  const span = hi - lo;
  lo -= span * 0.08; hi += span * 0.08;
  const x = (i) => padL + (maxTrials <= 1 ? 0 : (i / (maxTrials - 1)) * (W - padL - padR));
  const y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);

  const gridValues = [lo + (hi - lo) * 0.1, lo + (hi - lo) * 0.5, lo + (hi - lo) * 0.9];
  const grid = gridValues
    .map((v) => `
      <line x1="${padL}" y1="${y(v)}" x2="${W - padR}" y2="${y(v)}" class="tt-grid"></line>
      <text x="${padL - 6}" y="${y(v) + 3}" class="tt-axis" text-anchor="end">${v.toFixed(3)}</text>`)
    .join("");

  let linesSvg = "";
  let finalLabel = "";
  named.forEach((r, i) => {
    const isBest = r.run_id === bestId;
    const color = palette[i % palette.length];
    const points = r.tuning.history.map((h, hi2) => `${x(hi2)},${y(h.best_score)}`).join(" ");
    linesSvg += `<polyline points="${points}" class="tt-line ${isBest ? "champion" : ""}" style="stroke:${color}"></polyline>`;
    if (isBest) {
      const last = r.tuning.history[r.tuning.history.length - 1];
      finalLabel = `<text x="${x(r.tuning.history.length - 1) - 6}" y="${y(last.best_score) - 8}" class="tt-final" text-anchor="end">${last.best_score.toFixed(4)}</text>`;
    }
  });
  overflow.forEach((r) => {
    const points = r.tuning.history.map((h, hi2) => `${x(hi2)},${y(h.best_score)}`).join(" ");
    linesSvg += `<polyline points="${points}" class="tt-line other-fold"></polyline>`;
  });

  $("exp-trend-chart").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Best-so-far tuning score per trial, one line per candidate" style="width:100%;height:auto">
      ${grid}
      ${linesSvg}
      ${finalLabel}
      <text x="${(padL + W - padR) / 2}" y="${H - 8}" class="tt-axis" text-anchor="middle">trial</text>
    </svg>`;

  $("exp-trend-legend").innerHTML = named
    .map((r, i) => `<li><span class="tt-chip" style="background:${palette[i % palette.length]}"></span>${escapeHtml(r.candidate_name)}${r.run_id === bestId ? " (champion)" : ""}</li>`)
    .join("") + (overflow.length ? `<li><span class="tt-chip" style="background:var(--text-secondary);opacity:0.5"></span>${overflow.length} other candidate(s)</li>` : "");
}
```

Add the call inside `renderExperimentsTab`:

```javascript
function renderExperimentsTab(run) {
  renderExperimentsStatCards(run);
  renderExperimentsBarChart(run);
  renderExperimentsTrend(run);
}
```

- [ ] **Step 5: Manual verification**

Browser console, with `FIXTURE_RUN` (has 2 tuned candidates: Logistic Regression 15 trials, CatBoost 20 trials; XGBoost and Decision Tree have `tuning.enabled: false`):

```javascript
renderExperimentsTab(FIXTURE_RUN);
document.querySelectorAll("#exp-trend-chart polyline").length
```

Expected: `2`. Then:

```javascript
document.querySelectorAll("#exp-trend-legend li").length
```

Expected: `2` (no overflow item, since only 2 tuned candidates ≤ 8). Then test the empty state:

```javascript
const noTuning = JSON.parse(JSON.stringify(FIXTURE_RUN));
noTuning.training_results.forEach(r => r.tuning.enabled = false);
renderExperimentsTab(noTuning);
$("exp-trend-empty").classList.contains("hidden")
```

Expected: `false` (empty-state message visible). Also confirm the old "Hyperparameter tuning trend" card no longer appears on the Models Compared tab (`document.getElementById("tuning-card")` should be `null`).

- [ ] **Step 6: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: replace the single-model tuning chart with a multi-series Experiment Trend chart"
```

---

## Task 5: Consolidated "All Experiments" table

**Files:**
- Modify: `frontend/index.html` (add table card inside `.experiments-main`)
- Modify: `frontend/app.js` (add `renderExperimentsTable`, call from `renderExperimentsTab`)

**Interfaces:**
- Consumes: `run.training_results`, `run.best_model`, `run.task_spec.metric`, reuses existing `cvCell(result, metric)` (`app.js:2080-2086`) and `errorDisclosure(error)` (`app.js:2553-2561`) helpers unchanged.
- Produces: fills `#exp-table`.

- [ ] **Step 1: Add the table card markup**

In `frontend/index.html`, inside `.experiments-main`, after the trend-chart card, add:

```html
            <div class="card">
              <div class="card-head"><h3>All Experiments</h3><span class="muted small" id="exp-table-sub"></span></div>
              <div class="table-scroll"><table class="results-table" id="exp-table"></table></div>
            </div>
```

- [ ] **Step 2: Implement `renderExperimentsTable`**

Add to `frontend/app.js`:

```javascript
function renderExperimentsTable(run) {
  const results = run.training_results || [];
  const metric = (run.task_spec || {}).metric;
  const bestId = (run.best_model || {}).run_id;
  const lowerIsBetter = metric === "rmse" || metric === "mae";
  const metricNames = [...new Set(results.flatMap((r) => Object.keys(r.metrics || {})))];
  const primary = metric && metricNames.includes(metric) ? metric : metricNames[0];
  const secondary = metricNames.find((m) => m !== primary);
  const hasCv = primary && results.some((r) => r.cv_metrics && primary in r.cv_metrics);

  const ranked = [...results].sort((a, b) => {
    const aHas = a.metrics && primary in (a.metrics || {}), bHas = b.metrics && primary in (b.metrics || {});
    if (!aHas && !bHas) return 0;
    if (!aHas) return 1;
    if (!bHas) return -1;
    return lowerIsBetter ? a.metrics[primary] - b.metrics[primary] : b.metrics[primary] - a.metrics[primary];
  });

  $("exp-table-sub").textContent = primary ? `${results.length} candidate(s) · ranked by ${primary}` : `${results.length} candidate(s)`;

  let html = `<tr><th>Rank</th><th>Model</th><th>Trials</th>${primary ? `<th>${escapeHtml(primary)}${hasCv ? " (CV)" : ""}</th>` : ""}${secondary ? `<th>${escapeHtml(secondary)}</th>` : ""}<th>Training Time</th><th>Status</th></tr>`;
  ranked.forEach((r, i) => {
    const isBest = r.run_id === bestId;
    const trials = r.tuning && r.tuning.enabled ? `${r.tuning.trials_done}/${r.tuning.trials_total}` : "no tuning";
    html += `<tr class="${isBest ? "best" : ""}">
      <td>${i + 1}</td>
      <td>${escapeHtml(r.candidate_name)}${isBest ? '<span class="winner-tag">★ CHAMPION</span>' : ""}</td>
      <td>${escapeHtml(trials)}</td>
      ${primary ? `<td class="num">${r.metrics && primary in r.metrics ? (hasCv ? cvCell(r, primary) : Number(r.metrics[primary]).toFixed(4)) : "—"}</td>` : ""}
      ${secondary ? `<td class="num">${r.metrics && secondary in r.metrics ? Number(r.metrics[secondary]).toFixed(4) : "—"}</td>` : ""}
      <td>${r.duration_seconds != null ? formatDuration(r.duration_seconds) : "—"}</td>
      <td>${escapeHtml(r.status.replaceAll("_", " "))}${r.error ? errorDisclosure(r.error) : ""}</td>
    </tr>`;
  });
  $("exp-table").innerHTML = html;
}
```

Add the call inside `renderExperimentsTab`:

```javascript
function renderExperimentsTab(run) {
  renderExperimentsStatCards(run);
  renderExperimentsBarChart(run);
  renderExperimentsTrend(run);
  renderExperimentsTable(run);
}
```

- [ ] **Step 3: Manual verification**

Browser console, with `FIXTURE_RUN`:

```javascript
renderExperimentsTab(FIXTURE_RUN);
document.querySelectorAll("#exp-table tr").length
```

Expected: `5` (1 header + 4 candidates, including the failed one).

```javascript
document.querySelector("#exp-table tr.best td:nth-child(2)").textContent
```

Expected to start with `"Logistic Regression"` (contains the winner-tag text too, but the leading text is the name).

```javascript
document.querySelector("#exp-table tr:last-child td:nth-child(3)").textContent
```

Expected: `"no tuning"` (Decision Tree, ranked last since it has no `roc_auc` in `metrics`). Confirm the failed row shows a collapsed error `<details>` (click it and confirm it expands to show the full "ValueError: input contains NaN" text).

- [ ] **Step 4: Commit**

```bash
git add frontend/index.html frontend/app.js
git commit -m "feat: add the consolidated All Experiments table"
```

---

## Task 6: Distribution donuts (By Model, By Status, By Outcome, By Compute Time)

**Files:**
- Modify: `frontend/index.html` (add a `.dash-grid` of 4 donut cards inside `.experiments-main`)
- Modify: `frontend/app.js` (add `renderExperimentsDonuts`, call from `renderExperimentsTab`)
- Modify: `frontend/styles.css` (small additions for ordinal compute-time shading; reuses existing `.donut-wrap`/`.donut-legend`/`.donut-center` otherwise)

**Interfaces:**
- Consumes: `run.training_results`, `run.best_model`.
- Produces: fills 4 independent donut widgets.
- Color jobs (per the dataviz skill — not all 4 use the same palette):
  - By Model → categorical identity → `--cat-1..8` (Task 3's palette), 9th+ folds into a shared muted "Other" slice.
  - By Status → status/state → `--accent-success` (succeeded) / `--accent-danger` (failed) / `--accent-warning` (timed_out) / `--text-secondary` (running/pending).
  - By Outcome (vs champion) → status-like state → `--accent-primary` (champion) / `--accent-success` (close contender, within 2% relative of champion's primary metric) / `--text-secondary` (trailed).
  - By Compute Time → ordinal magnitude tiers (<30s / 30s-2m / >2m) → single hue (`--cat-2`, blue) at 3 opacities (1 / 0.65 / 0.35), never 3 unrelated hues.

- [ ] **Step 1: Add the donut cards markup**

In `frontend/index.html`, inside `.experiments-main`, after the table card, add:

```html
            <section class="dash-grid">
              <div class="card">
                <div class="card-head"><h3>By Model</h3><span class="muted small" id="exp-donut-model-sub"></span></div>
                <div class="donut-wrap">
                  <svg id="exp-donut-model" viewBox="0 0 120 120" role="img" aria-label="Trial share by model"></svg>
                  <div class="donut-center" id="exp-donut-model-center"></div>
                  <ul class="donut-legend" id="exp-donut-model-legend"></ul>
                </div>
              </div>
              <div class="card">
                <div class="card-head"><h3>By Status</h3><span class="muted small" id="exp-donut-status-sub"></span></div>
                <div class="donut-wrap">
                  <svg id="exp-donut-status" viewBox="0 0 120 120" role="img" aria-label="Candidate count by status"></svg>
                  <div class="donut-center" id="exp-donut-status-center"></div>
                  <ul class="donut-legend" id="exp-donut-status-legend"></ul>
                </div>
              </div>
              <div class="card">
                <div class="card-head"><h3>By Outcome</h3><span class="muted small" id="exp-donut-outcome-sub"></span></div>
                <div class="donut-wrap">
                  <svg id="exp-donut-outcome" viewBox="0 0 120 120" role="img" aria-label="Candidate count vs champion"></svg>
                  <div class="donut-center" id="exp-donut-outcome-center"></div>
                  <ul class="donut-legend" id="exp-donut-outcome-legend"></ul>
                </div>
              </div>
              <div class="card">
                <div class="card-head"><h3>By Compute Time</h3><span class="muted small" id="exp-donut-compute-sub"></span></div>
                <div class="donut-wrap">
                  <svg id="exp-donut-compute" viewBox="0 0 120 120" role="img" aria-label="Candidate count by training time bucket"></svg>
                  <div class="donut-center" id="exp-donut-compute-center"></div>
                  <ul class="donut-legend" id="exp-donut-compute-legend"></ul>
                </div>
              </div>
            </section>
```

- [ ] **Step 2: Implement a shared donut-drawing helper + the 4 renderers**

Add to `frontend/app.js` (a generic helper factored out of the existing donut logic, used only by the new donuts — `renderDatasetSummary`/`renderClassDistribution` are untouched to avoid regressing already-working code):

```javascript
/* ================= experiments: distribution donuts ================= */

function drawDonut(svgId, centerId, legendId, entries, colors, centerLabel) {
  const total = entries.reduce((acc, [, n]) => acc + n, 0);
  if (!total) { $(svgId).innerHTML = ""; $(centerId).innerHTML = ""; $(legendId).innerHTML = ""; return; }
  const R = 44, C = 2 * Math.PI * R;
  const gapPx = entries.length > 1 ? 3 : 0;
  let offset = 0;
  let svg = "";
  entries.forEach(([, count], i) => {
    const frac = count / total;
    const len = Math.max(frac * C - gapPx, 1);
    svg += `<circle cx="60" cy="60" r="${R}" fill="none" stroke="${colors[i % colors.length]}"
      stroke-width="14" stroke-linecap="butt"
      stroke-dasharray="${len} ${C - len}" stroke-dashoffset="${-offset}"
      transform="rotate(-90 60 60)"/>`;
    offset += frac * C;
  });
  $(svgId).innerHTML = svg;
  $(centerId).innerHTML = centerLabel;
  $(legendId).innerHTML = entries
    .map(([label, count], i) => `
      <li><span class="swatch" style="background:${colors[i % colors.length]}"></span>
      ${escapeHtml(label)}<span class="count">${count} (${Math.round((count / total) * 100)}%)</span></li>`)
    .join("");
}

function renderExperimentsDonuts(run) {
  const results = run.training_results || [];
  const bestId = (run.best_model || {}).run_id;
  const metric = (run.task_spec || {}).metric;
  const styles = getComputedStyle(document.documentElement);
  const cssVar = (k) => styles.getPropertyValue(k).trim();

  // By Model — trial-count share per candidate, 8 named + "Other" fold-in
  const named = results.slice(0, 8);
  const overflow = results.slice(8);
  const modelEntries = named.map((r) => [r.candidate_name, r.tuning?.trials_done || 1]);
  const modelColors = EXP_TREND_COLOR_KEYS.map(cssVar);
  if (overflow.length) {
    modelEntries.push(["Other", overflow.reduce((s, r) => s + (r.tuning?.trials_done || 1), 0)]);
    modelColors.push(cssVar("--text-secondary"));
  }
  $("exp-donut-model-sub").textContent = `${results.length} candidate(s)`;
  drawDonut("exp-donut-model", "exp-donut-model-center", "exp-donut-model-legend", modelEntries, modelColors, `${results.length}<small>models</small>`);

  // By Status — fixed status colors, never categorical
  const statusOrder = ["succeeded", "failed", "timed_out", "running", "pending"];
  const statusColor = { succeeded: cssVar("--accent-success"), failed: cssVar("--accent-danger"), timed_out: cssVar("--accent-warning"), running: cssVar("--text-secondary"), pending: cssVar("--text-secondary") };
  const statusCounts = {};
  for (const r of results) statusCounts[r.status] = (statusCounts[r.status] || 0) + 1;
  const statusEntries = statusOrder.filter((s) => statusCounts[s]).map((s) => [s.replaceAll("_", " "), statusCounts[s]]);
  const statusColors = statusOrder.filter((s) => statusCounts[s]).map((s) => statusColor[s]);
  $("exp-donut-status-sub").textContent = `${results.length} total`;
  drawDonut("exp-donut-status", "exp-donut-status-center", "exp-donut-status-legend", statusEntries, statusColors, `${results.length}<small>total</small>`);

  // By Outcome vs champion — champion / close contender (within 2% relative) / trailed
  const best = run.best_model || {};
  const bestScore = metric && best.metrics && metric in best.metrics ? Number(best.metrics[metric]) : null;
  const lowerIsBetter = metric === "rmse" || metric === "mae";
  let champCount = 0, closeCount = 0, trailedCount = 0;
  for (const r of results) {
    if (r.run_id === bestId) { champCount += 1; continue; }
    if (bestScore == null || !r.metrics || !(metric in r.metrics)) { trailedCount += 1; continue; }
    const rel = Math.abs(r.metrics[metric] - bestScore) / Math.abs(bestScore || 1);
    const better = lowerIsBetter ? r.metrics[metric] < bestScore : r.metrics[metric] > bestScore;
    if (rel <= 0.02 || better) closeCount += 1; else trailedCount += 1;
  }
  const outcomeEntries = [["Champion", champCount], ["Close contender", closeCount], ["Trailed", trailedCount]].filter(([, n]) => n > 0);
  const outcomeColorMap = { Champion: cssVar("--accent-primary"), "Close contender": cssVar("--accent-success"), Trailed: cssVar("--text-secondary") };
  const outcomeColors = outcomeEntries.map(([label]) => outcomeColorMap[label]);
  $("exp-donut-outcome-sub").textContent = `vs ${escapeHtml(best.candidate_name || "champion")}`;
  drawDonut("exp-donut-outcome", "exp-donut-outcome-center", "exp-donut-outcome-legend", outcomeEntries, outcomeColors, `${results.length}<small>total</small>`);

  // By Compute Time — ordinal buckets, one hue at 3 opacities
  const buckets = [
    { label: "< 30s", test: (d) => d < 30 },
    { label: "30s - 2m", test: (d) => d >= 30 && d <= 120 },
    { label: "> 2m", test: (d) => d > 120 },
  ];
  const base = cssVar("--cat-2");
  const toRgba = (hex, alpha) => {
    const n = parseInt(hex.replace("#", ""), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
  };
  const computeCounts = buckets.map((b) => results.filter((r) => r.duration_seconds != null && b.test(r.duration_seconds)).length);
  const computeEntries = buckets.map((b, i) => [b.label, computeCounts[i]]).filter(([, n]) => n > 0);
  const computeColors = [toRgba(base, 1), toRgba(base, 0.65), toRgba(base, 0.35)].filter((_, i) => computeCounts[i] > 0);
  $("exp-donut-compute-sub").textContent = `${results.length} candidate(s)`;
  drawDonut("exp-donut-compute", "exp-donut-compute-center", "exp-donut-compute-legend", computeEntries, computeColors, `${results.length}<small>total</small>`);
}
```

Add the call inside `renderExperimentsTab`:

```javascript
function renderExperimentsTab(run) {
  renderExperimentsStatCards(run);
  renderExperimentsBarChart(run);
  renderExperimentsTrend(run);
  renderExperimentsTable(run);
  renderExperimentsDonuts(run);
}
```

- [ ] **Step 3: Manual verification**

Browser console, with `FIXTURE_RUN` (4 candidates: 1 succeeded+champion, 2 succeeded others, 1 failed):

```javascript
renderExperimentsTab(FIXTURE_RUN);
document.querySelectorAll("#exp-donut-model-legend li").length
```

Expected: `4` (no overflow — only 4 candidates, all named).

```javascript
document.querySelectorAll("#exp-donut-status-legend li").length
```

Expected: `2` ("succeeded" ×3, "failed" ×1 — both non-zero buckets shown; running/pending/timed_out absent).

```javascript
[...document.querySelectorAll("#exp-donut-outcome-legend li")].map(li => li.textContent)
```

Expected to include a "Champion" entry with count 1, and the other 2 succeeded candidates classified as "Close contender" or "Trailed" per the 2%-relative rule (CatBoost 0.889 vs champion 0.992: relative diff ≈ 10.4% → Trailed; XGBoost 0.888 → Trailed) — the failed Decision Tree (no `roc_auc` in `metrics`) also counts as Trailed. So expect `Champion (1)`, `Trailed (3)`, no "Close contender" entry for this fixture.

```javascript
document.querySelectorAll("#exp-donut-compute-legend li").length
```

Expected: `2` (durations 18s and 4s fall in `<30s`, 132s and 258s fall in `>2m`; the `30s-2m` bucket is empty for this fixture and is omitted from the legend).

- [ ] **Step 4: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: add the 4 distribution donuts to the Experiments tab"
```

---

## Task 7: Best Experiment side panel + two-column layout

**Files:**
- Modify: `frontend/styles.css` (add `.experiments-layout` grid)
- Modify: `frontend/app.js` (add `renderExperimentsBestPanel`, call from `renderExperimentsTab`)

**Interfaces:**
- Consumes: `run.best_model`, `run.created_at`, `run.resampling_plan`, `run.profile_columns` + `run.task_spec.target_column` (for the class-imbalance-ratio "Data Used" line, mirroring the computation already in `renderClassDistribution`, `app.js:1924-1973` — not refactored into a shared helper, since it's a small ~5-line computation used at only 2 call sites and extracting it adds an indirection for no real reuse benefit here).
- Produces: fills `#exp-best-panel` (already in the DOM from Task 1).

- [ ] **Step 1: Add the layout CSS**

Append to `frontend/styles.css`:

```css
/* ================= experiments: layout + best-experiment panel ================= */

.experiments-layout { display: grid; grid-template-columns: 2fr 1fr; gap: var(--sp-4); align-items: start; }
@media (max-width: 980px) { .experiments-layout { grid-template-columns: 1fr; } }
.experiments-main { display: grid; gap: var(--sp-3); }
.exp-best-header { display: flex; align-items: center; gap: var(--sp-2); margin-bottom: var(--sp-3); }
.exp-best-header .stat-icon { flex-shrink: 0; }
.exp-best-section { margin-top: var(--sp-3); }
.exp-best-section h4 { font-size: var(--text-xs); text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-secondary); margin-bottom: 6px; }
.exp-best-row { display: flex; justify-content: space-between; gap: var(--sp-2); font-size: var(--text-sm); padding: 5px 0; border-bottom: 1px solid var(--border-subtle); }
.exp-best-row:last-child { border-bottom: none; }
.exp-best-row .label { color: var(--text-secondary); }
.exp-best-row .value { font-family: var(--font-mono); font-weight: 600; text-align: right; }
```

- [ ] **Step 2: Implement `renderExperimentsBestPanel`**

Add to `frontend/app.js`:

```javascript
function classImbalanceRatioLabel(run) {
  const spec = run.task_spec || {};
  const target = (run.profile_columns || []).find((c) => c.name === spec.target_column);
  const entries = target && target.top_values ? Object.entries(target.top_values).sort((a, b) => b[1] - a[1]) : [];
  if (spec.task_type !== "classification" || entries.length < 2) return null;
  const majority = entries[0][1], minority = entries[entries.length - 1][1];
  if (!minority) return null;
  const total = majority + minority;
  return `${Math.round((majority / total) * 100)}:${Math.round((minority / total) * 100)}`;
}

function renderExperimentsBestPanel(run) {
  const best = run.best_model || {};
  const panel = $("exp-best-panel");
  if (!best.candidate_name) { panel.innerHTML = `<p class="muted small">No champion selected yet.</p>`; return; }

  const metric = (run.task_spec || {}).metric;
  const results = run.training_results || [];
  const metricNames = [...new Set(results.flatMap((r) => Object.keys(r.metrics || {})))];
  const secondary = metricNames.find((m) => m !== metric);
  const trials = best.tuning && best.tuning.enabled ? `${best.tuning.trials_done}/${best.tuning.trials_total}` : "no tuning";

  const metricRows = [metric, secondary]
    .filter(Boolean)
    .map((m) => {
      const cv = best.cv_metrics && best.cv_metrics[m];
      const value = cv ? `${cv.mean.toFixed(3)} ± ${cv.std.toFixed(3)}` : (best.metrics && m in best.metrics ? Number(best.metrics[m]).toFixed(3) : "—");
      return `<div class="exp-best-row"><span class="label">${escapeHtml(m)}${cv ? " (CV mean)" : ""}</span><span class="value">${value}</span></div>`;
    })
    .join("");

  const ratio = classImbalanceRatioLabel(run);
  const resampling = run.resampling_plan || {};
  const dataUsed = resampling.enabled && best.resampling_applied
    ? `${best.resampling_applied.replaceAll("_", " ")}${ratio ? ` (${ratio})` : ""}`
    : "no resampling";

  panel.innerHTML = `
    <div class="exp-best-header">
      <span class="stat-icon amber">${ICONS.trophy}</span>
      <div>
        <div class="stat-label">Best Experiment</div>
        <h3>${escapeHtml(best.candidate_name)}</h3>
      </div>
    </div>
    <div class="exp-best-row"><span class="label">Trials</span><span class="value">${escapeHtml(trials)}</span></div>
    <div class="exp-best-row"><span class="label">Status</span><span class="value">${escapeHtml((best.status || "succeeded").replaceAll("_", " "))}</span></div>
    <div class="exp-best-section">
      <h4>Key Metrics</h4>
      ${metricRows}
    </div>
    <div class="exp-best-section">
      <h4>Training Info</h4>
      <div class="exp-best-row"><span class="label">Training Time</span><span class="value">${best.duration_seconds != null ? formatDuration(best.duration_seconds) : "—"}</span></div>
      <div class="exp-best-row"><span class="label">Start Time</span><span class="value">${run.created_at ? new Date(run.created_at * 1000).toLocaleString() : "—"}</span></div>
      <div class="exp-best-row"><span class="label">Data Used</span><span class="value">${escapeHtml(dataUsed)}</span></div>
      <div class="exp-best-row"><span class="label">Folds</span><span class="value">${best.cv_folds ? `${best.cv_folds}-fold CV` : "no CV"}</span></div>
    </div>`;
}
```

Add the call inside `renderExperimentsTab`:

```javascript
function renderExperimentsTab(run) {
  renderExperimentsStatCards(run);
  renderExperimentsBarChart(run);
  renderExperimentsTrend(run);
  renderExperimentsTable(run);
  renderExperimentsDonuts(run);
  renderExperimentsBestPanel(run);
}
```

- [ ] **Step 3: Manual verification**

Browser console, with `FIXTURE_RUN`:

```javascript
renderExperimentsTab(FIXTURE_RUN);
$("exp-best-panel").querySelector("h3").textContent
```

Expected: `"Logistic Regression"`.

```javascript
$("exp-best-panel").querySelectorAll(".exp-best-row").length
```

Expected: `8` (Trials, Status, roc_auc, accuracy, Training Time, Start Time, Data Used, Folds).

```javascript
[...$("exp-best-panel").querySelectorAll(".exp-best-row .value")][2].textContent
```

Expected: `"0.992 ± 0.003"` (roc_auc CV mean ± std).

Resize the browser (or devtools responsive mode) below 980px width and confirm the layout stacks to a single column instead of side-by-side.

- [ ] **Step 4: Commit**

```bash
git add frontend/app.js frontend/styles.css
git commit -m "feat: add the Best Experiment side panel and two-column Experiments layout"
```

---

## Task 8: Full end-to-end pass against a real run

**Files:** none (verification only)

**Interfaces:** none — this task exercises the whole `renderExperimentsTab` pipeline built in Tasks 1-7 against real, non-fixture data.

- [ ] **Step 1: Start the app and drive one real run**

Use the `run` skill to start the server. Upload a small dataset (or reuse an existing one from `/tests/fixtures` if the local UI supports pointing at a local file) and let a run complete — mock LLM provider is fine per `config/models.yaml` if configured, to avoid real API spend for this manual pass.

- [ ] **Step 2: Verify every section against real data**

Open the completed run, click the Experiments tab, and confirm, with no console errors:
- 5 stat cards show plausible non-placeholder numbers.
- The bar chart shows one bar per successfully-trained candidate, champion highlighted.
- The trend chart shows a line per tuned candidate (or the empty-state message if tuning was disabled for the run).
- The table lists every candidate including any that failed, with the error disclosure working on a failed row if one exists.
- All 4 donuts render with legends whose percentages sum to 100 (rounding aside) and whose counts match the table.
- The side panel shows the same champion as the Overview tab's champion banner, with matching metric values.

- [ ] **Step 3: Edge cases**

If feasible, run (or fixture-simulate via the console as in earlier tasks) these cases and confirm no section throws or renders `NaN`/`undefined`:
- A run with exactly 1 candidate (bar chart, table, all donuts degenerate to one row/slice; trend chart shows either 1 line or the empty state).
- A run with tuning disabled entirely (trend chart empty state).
- A run with more than 8 candidates (By Model donut and trend chart legend show an "Other"/overflow entry; no 9th distinct hue appears).

- [ ] **Step 4: Final commit (only if Step 1-3 surfaced fixes)**

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "fix: address edge cases found in end-to-end Experiments tab verification"
```

If no fixes were needed, skip this commit — Tasks 1-7 already cover the feature.
