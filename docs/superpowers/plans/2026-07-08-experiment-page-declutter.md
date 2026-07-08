# Experiment Page Restructure + Declutter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the run view's tab structure — fold the Pipeline tab and Models Compared tab into the Experiments tab, relocate the Overview tab's dataset visuals into Experiments too — and reduce visual clutter (monotone spacing, redundant cards) across the run page.

**Architecture:** `frontend/` is a static, no-build vanilla-JS/HTML/CSS app (`frontend/index.html`, `frontend/app.js`, `frontend/styles.css`) served by FastAPI and polled every 1.5s via `GET /api/runs/{id}`. This plan is a pure frontend restructuring: markup moves between existing containers, a few render-function bodies get trimmed of now-dead code, and CSS gains a handful of reusable modifiers (`.collapsible-card`, `.stage-tracker.compact`, `.card.dense`). No backend/API changes, no new `PipelineState` fields.

**Tech Stack:** Vanilla JS (global scope, no modules, no bundler), plain CSS custom properties for theming (light/dark).

## Global Constraints

- Raw data never enters an LLM context window — not relevant here (pure frontend rendering of already-fetched run JSON).
- No new backend endpoints or `PipelineState` fields — everything needed already exists on the run JSON returned by `GET /api/runs/{run_id}`.
- No JS test runner exists in this repo (confirmed: `frontend/` has no `package.json`/test framework). Verification for every task is manual: load the app via the `run` skill, open browser devtools, and either (a) call render functions directly against a hand-built fixture `run` object pasted into the console, or (b) drive a real run to the relevant state.
- Reference spec: `docs/superpowers/specs/2026-07-08-experiment-page-declutter-design.md`.
- Every button/link that today navigates to `switchRunTab("pipeline")` or `switchRunTab("models")` must be repointed to `switchRunTab("experiments")`. Once `"pipeline"`/`"models"` are removed from `RUN_TABS` (Tasks 2/3), `switchRunTab("pipeline")` would call `$("tab-pipeline-btn")` / `$("tab-pipeline-panel")` on elements that no longer exist in the DOM, throwing on click — a missed repoint fails loudly (`Cannot read properties of null`), not silently, but must still be fixed in the same task that removes the tab.

---

## Task 1: Collapsible AI Summary / Recent Activity cards

**Files:**
- Modify: `frontend/index.html:587-591` (`#ai-summary-card`), `frontend/index.html:648-651` (`#activity-card`)
- Modify: `frontend/app.js` (add `chevron` icon to `ICONS`, add `initCollapsible()` helper + two call sites)
- Modify: `frontend/styles.css` (add `.collapsible-card` / `.collapsible-toggle` / `.card-collapsible-body` rules)

**Interfaces:**
- Produces: `initCollapsible(cardId)` — call once per card at load time. Later tasks don't depend on this.

- [ ] **Step 1: Add the chevron icon**

In `frontend/app.js`, in the `ICONS` object (`app.js:8-28`), add a new entry right after `bulb`:

```javascript
  bulb: SVG('<path d="M9 18h6M10 22h4M12 2a6 6 0 0 0-4 10.5c.6.6 1 1.4 1 2.5h6c0-1.1.4-1.9 1-2.5A6 6 0 0 0 12 2Z"/>'),
  chevron: SVG('<path d="m6 9 6 6 6-6"/>'),
};
```

- [ ] **Step 2: Restructure the AI Summary card markup**

In `frontend/index.html`, replace (`index.html:587-591`):

```html
        <div class="card hidden ai-summary-card" id="ai-summary-card">
          <div class="card-head"><h3><svg class="icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2 14 9 21 12 14 15 12 22 10 15 3 12 10 9Z"/></svg> AI Summary</h3></div>
          <div id="report-lede" class="report-lede hidden"></div>
          <div id="report-body" class="report-body"></div>
        </div>
```

with:

```html
        <div class="card hidden ai-summary-card collapsible-card" id="ai-summary-card">
          <div class="card-head">
            <button type="button" class="collapsible-toggle" id="ai-summary-toggle" aria-expanded="false" aria-controls="ai-summary-body">
              <h3><svg class="icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2 14 9 21 12 14 15 12 22 10 15 3 12 10 9Z"/></svg> AI Summary</h3>
              <span class="collapsible-chevron"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg></span>
            </button>
          </div>
          <div class="card-collapsible-body" id="ai-summary-body">
            <div class="card-collapsible-inner">
              <div id="report-lede" class="report-lede hidden"></div>
              <div id="report-body" class="report-body"></div>
            </div>
          </div>
        </div>
```

- [ ] **Step 3: Restructure the Recent Activity card markup**

In `frontend/index.html`, replace (`index.html:648-651`):

```html
        <div class="card hidden" id="activity-card">
          <div class="card-head"><h3>Recent activity</h3></div>
          <ul class="activity-list" id="activity-list"></ul>
        </div>
```

with:

```html
        <div class="card hidden collapsible-card" id="activity-card">
          <div class="card-head">
            <button type="button" class="collapsible-toggle" id="activity-toggle" aria-expanded="false" aria-controls="activity-body">
              <h3>Recent activity</h3>
              <span class="collapsible-chevron"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg></span>
            </button>
          </div>
          <div class="card-collapsible-body" id="activity-body">
            <div class="card-collapsible-inner">
              <ul class="activity-list" id="activity-list"></ul>
            </div>
          </div>
        </div>
```

- [ ] **Step 4: Add the collapsible CSS**

Append to `frontend/styles.css`:

```css
/* ================= collapsible cards ================= */

.collapsible-toggle {
  all: unset; display: flex; justify-content: space-between; align-items: center;
  width: 100%; cursor: pointer; gap: var(--sp-2); box-sizing: border-box;
}
.collapsible-toggle:focus-visible { outline: 2px solid var(--accent-primary); outline-offset: 2px; border-radius: var(--radius-sm); }
.collapsible-chevron { display: inline-flex; flex-shrink: 0; transition: transform 0.2s ease; }
.collapsible-chevron svg { width: 16px; height: 16px; }
.collapsible-toggle[aria-expanded="true"] .collapsible-chevron { transform: rotate(180deg); }

.card-collapsible-body { display: grid; grid-template-rows: 0fr; transition: grid-template-rows 0.2s ease; }
.card-collapsible-body.expanded { grid-template-rows: 1fr; margin-top: var(--sp-3); }
.card-collapsible-inner { overflow: hidden; min-height: 0; }

@media (prefers-reduced-motion: reduce) {
  .collapsible-chevron, .card-collapsible-body { transition: none; }
}
```

- [ ] **Step 5: Add the `initCollapsible` helper and wire both cards**

In `frontend/app.js`, add just before the final deep-link bootstrapping block (immediately after the `escapeHtml` function, `app.js:2865-2869`, and before the `$("logout-btn")...` listener at `app.js:2871`):

```javascript
/* ================= collapsible rail cards ================= */

function initCollapsible(cardId) {
  const card = $(cardId);
  const toggle = card.querySelector(".collapsible-toggle");
  const body = card.querySelector(".card-collapsible-body");
  const storageKey = `collapse:${cardId}`;
  const expanded = localStorage.getItem(storageKey) === "true";
  toggle.setAttribute("aria-expanded", String(expanded));
  body.classList.toggle("expanded", expanded);
  toggle.addEventListener("click", () => {
    const next = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(next));
    body.classList.toggle("expanded", next);
    localStorage.setItem(storageKey, String(next));
  });
}
initCollapsible("ai-summary-card");
initCollapsible("activity-card");
```

- [ ] **Step 6: Manual verification**

Start the app with the `run` skill (`python run_server.py`, open `http://127.0.0.1:8000`). Open any completed run. In devtools console:

```javascript
$("ai-summary-toggle").getAttribute("aria-expanded")
```

Expected: `"false"` (collapsed by default on first visit — no prior `localStorage` entry).

Click the "AI Summary" header. Confirm the report content becomes visible and `$("ai-summary-toggle").getAttribute("aria-expanded")` is now `"true"`. Reload the page (same run via the `?run=` deep link) and confirm it's still expanded (persisted). Click it closed, reload again, confirm it stays collapsed. Repeat the same check for "Recent activity" (`activity-toggle`/`activity-card`). Toggle OS-level "reduce motion" (or `prefers-reduced-motion` in devtools rendering emulation) and confirm the expand/collapse still works, just without the animated chevron rotation / grid transition.

- [ ] **Step 7: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: make AI Summary and Recent Activity collapsible, collapsed by default"
```

---

## Task 2: Move the pipeline progress bar to the top of Experiments; remove the Pipeline tab

**Files:**
- Modify: `frontend/index.html:189-198` (tab bar), `:456-470` (delete `#tab-pipeline-panel`, relocate its contents), `:489` (top of `#tab-experiments-panel`)
- Modify: `frontend/app.js:1181` (repoint `journey-view-pipeline-btn`), `frontend/app.js:1416-1475` (`renderStageTracker` gains a `.compact` toggle), `frontend/app.js:2678` (`RUN_TABS`)
- Modify: `frontend/styles.css` (add `.stage-tracker.compact` rules)

**Interfaces:**
- Consumes: existing `renderStageTracker(run)` / `renderTrainProgress(run)` — unchanged signatures, called from `render()` exactly as today (`app.js:1084`/`app.js:1087`); moving the DOM location of `#stage-tracker`/`#train-progress` requires no call-site change since both functions target elements by id.
- Produces: nothing new consumed by later tasks.

- [ ] **Step 1: Remove the Pipeline tab button**

In `frontend/index.html`, remove this line from the `.run-tab-bar` block (`index.html:192`):

```html
        <button class="tab-btn" id="tab-pipeline-btn" type="button" role="tab" aria-selected="false">Pipeline</button>
```

- [ ] **Step 2: Move the stage-tracker block into Experiments, delete the empty Pipeline panel**

In `frontend/index.html`, delete the entire `#tab-pipeline-panel` block (`index.html:456-470`):

```html
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

Then insert the same `<section class="card">...</section>` (its contents are byte-for-byte identical — only the surrounding wrapper `<div id="tab-pipeline-panel">` is gone) as the very first child of `#tab-experiments-panel`, i.e. immediately after `<div id="tab-experiments-panel" class="hidden">` (`index.html:489`) and before `<div class="experiments-layout">`:

```html
      <div id="tab-experiments-panel" class="hidden">
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
        <div class="experiments-layout">
```

(leave the rest of `.experiments-layout` and its contents untouched — this step only adds the `<section class="card">…</section>` block above it).

- [ ] **Step 3: Add the compact-mode toggle to `renderStageTracker`**

In `frontend/app.js`, inside `renderStageTracker` (`app.js:1416-1475`), the function already computes a `terminal` flag right after clearing the tracker. Reuse it for the compact toggle — change:

```javascript
  const tracker = $("stage-tracker");
  tracker.innerHTML = "";
  const terminal = ["completed", "failed", "cancelled"].includes(run.status);
  $("pipeline-sub").textContent = terminal
```

to:

```javascript
  const tracker = $("stage-tracker");
  tracker.innerHTML = "";
  const terminal = ["completed", "failed", "cancelled"].includes(run.status);
  tracker.classList.toggle("compact", terminal);
  $("pipeline-sub").textContent = terminal
```

- [ ] **Step 4: Add the compact CSS variant**

Append to `frontend/styles.css`, right after the existing stage-tracker rules (after `app.js`'s companion block ending at `styles.css:380`, i.e. right before the `/* training sub-progress */` comment at `styles.css:382`):

```css
.stage-tracker.compact { gap: 2px; padding-bottom: 0; }
.stage-tracker.compact .stage { min-width: 0; gap: 4px; }
.stage-tracker.compact .stage-dot { width: 26px; height: 26px; }
.stage-tracker.compact .stage-dot svg { width: 12px; height: 12px; }
.stage-tracker.compact .stage-label { font-size: 10px; }
.stage-tracker.compact .stage-status,
.stage-tracker.compact .stage-time,
.stage-tracker.compact .stage-retry { display: none; }
.stage-tracker.compact .stage:not(:first-child)::before { top: 13px; }
```

- [ ] **Step 5: Repoint the "View Full Pipeline" button and update `RUN_TABS`**

In `frontend/app.js:1181`, change:

```javascript
$("journey-view-pipeline-btn").addEventListener("click", () => switchRunTab("pipeline"));
```

to:

```javascript
$("journey-view-pipeline-btn").addEventListener("click", () => switchRunTab("experiments"));
```

In `frontend/app.js:2678`, change:

```javascript
const RUN_TABS = ["overview", "pipeline", "experiments", "models", "explainability", "artifacts", "logs"];
```

to (this task only removes `"pipeline"`; `"models"` is removed in Task 3):

```javascript
const RUN_TABS = ["overview", "experiments", "models", "explainability", "artifacts", "logs"];
```

- [ ] **Step 6: Manual verification**

Start the app via the `run` skill. Confirm the tab bar no longer shows "Pipeline". Start a new run (or open one still in progress) and open the Experiments tab — confirm the stage tracker + live train-progress bar appear at the very top, above the KPI stat cards / chart cards, and update live as the run progresses. Let the run finish (or open an already-completed run) and confirm the tracker switches to the compact strip (smaller dots, no status/time text below each stage). Click "View Full Pipeline" on the Overview tab's Journey card and confirm it lands on the Experiments tab with no console error. Confirm `document.getElementById("tab-pipeline-btn")` and `document.getElementById("tab-pipeline-panel")` are both `null`.

- [ ] **Step 7: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: move pipeline progress bar to top of Experiments tab, remove Pipeline tab"
```

---

## Task 3: Remove Models Compared tab; merge config chips into Experiments' All Experiments table

**Files:**
- Modify: `frontend/index.html:193` → button removal (note: after Task 2, the Pipeline button is already gone, so this is the tab bar's remaining `tab-models-btn` line), `:472-487` (delete `#tab-models-panel`), the "All Experiments" card inside `#tab-experiments-panel` (added by the prior Experiments Tab plan)
- Modify: `frontend/app.js:1152` (`champion-compare-btn`), `:1183` (`nextstep-compare-btn`), `:1249` (`leaderboard-view-all-btn`), `:2094-2159` (`renderResults` — strip dead table code), `frontend/app.js:2678` (`RUN_TABS`)

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing new consumed by later tasks. `renderResults(run)` keeps being called from `render()` (`app.js:1095`) unchanged — only its body is trimmed.

- [ ] **Step 1: Remove the Models Compared tab button**

In `frontend/index.html`, remove:

```html
        <button class="tab-btn" id="tab-models-btn" type="button" role="tab" aria-selected="false">Models Compared</button>
```

- [ ] **Step 2: Delete the Models Compared panel, relocate the chip-row into the All Experiments card**

In `frontend/index.html`, delete the entire `#tab-models-panel` block (`index.html:472-487`):

```html
      <div id="tab-models-panel" class="hidden">
        <section class="dash-grid">
          <div class="card hidden" id="results-card">
            <div class="card-head">
              <h3>Model comparison</h3>
              <span class="muted small" id="results-sub"></span>
            </div>
            <div class="chip-row">
              <div class="chip cv-config-chip" id="cv-config-chip"></div>
              <div class="chip cv-config-chip hidden" id="resampling-config-chip"></div>
              <div class="chip cv-config-chip hidden" id="fs-config-chip"></div>
            </div>
            <div class="table-scroll"><table class="results-table" id="results-table"></table></div>
          </div>
        </section>
      </div><!-- /tab-models-panel -->
```

Then, inside `#tab-experiments-panel`, find the "All Experiments" card (added by the prior Experiments Tab plan):

```html
            <div class="card">
              <div class="card-head"><h3>All Experiments</h3><span class="muted small" id="exp-table-sub"></span></div>
              <div class="table-scroll"><table class="results-table" id="exp-table"></table></div>
            </div>
```

Replace it with (chip-row inserted between the head and the table, `dense` class added per the density pass — this card holds a data table, not a hero element):

```html
            <div class="card dense">
              <div class="card-head"><h3>All Experiments</h3><span class="muted small" id="exp-table-sub"></span></div>
              <div class="chip-row">
                <div class="chip cv-config-chip" id="cv-config-chip"></div>
                <div class="chip cv-config-chip hidden" id="resampling-config-chip"></div>
                <div class="chip cv-config-chip hidden" id="fs-config-chip"></div>
              </div>
              <div class="table-scroll"><table class="results-table" id="exp-table"></table></div>
            </div>
```

(`dense` is added here now but its CSS (`.card.dense { padding: var(--sp-3); }`) isn't defined until Task 5 — harmless in the meantime, since an unrecognized class is simply inert; the card keeps its normal `.card` padding until Task 5 lands. The two tasks are independently committable in either order.)

- [ ] **Step 3: Strip the now-dead table code from `renderResults`**

In `frontend/app.js`, replace the whole `renderResults` function (`app.js:2094-2159`):

```javascript
function renderResults(run) {
  const results = run.training_results || [];
  const card = $("results-card");
  if (!results.length) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  const metric = (run.task_spec || {}).metric;
  $("results-sub").textContent = metric ? `ranked by ${metric}` : "";

  const cvConfig = run.cv_config || {};
  const cvChip = $("cv-config-chip");
  if (cvConfig.enabled) {
    cvChip.className = "chip detected cv-config-chip";
    cvChip.innerHTML = `${ICONS.check}${cvConfig.requested_folds}-fold cross-validation requested`;
  } else {
    cvChip.className = "chip cv-config-chip";
    cvChip.innerHTML = `Cross-validation disabled for this run`;
  }

  const resamplingChip = $("resampling-config-chip");
  const resamplingPlan = run.resampling_plan || {};
  if (resamplingPlan.enabled) {
    const applied = results.find((r) => r.resampling_applied)?.resampling_applied || resamplingPlan.method;
    const note = results.find((r) => r.resampling_note)?.resampling_note;
    resamplingChip.classList.remove("hidden");
    resamplingChip.className = "chip detected cv-config-chip";
    resamplingChip.title = note || "";
    resamplingChip.innerHTML = `${ICONS.check}${escapeHtml(applied.replaceAll("_", " "))} applied to training data`;
  } else {
    resamplingChip.classList.add("hidden");
  }

  const fsChip = $("fs-config-chip");
  const fsConfig = run.feature_selection_config || {};
  if (fsConfig.enabled) {
    const fs = run.feature_selection || {};
    fsChip.classList.remove("hidden");
    fsChip.className = "chip detected cv-config-chip";
    if (fs.enabled && fs.n_features_selected != null) {
      fsChip.title = `Selected with ${fs.basic_model || "a basic model"}: ${(fs.selected_features || []).join(", ")}`;
      fsChip.innerHTML = `${ICONS.check}RFE kept ${fs.n_features_selected} of ${fs.n_features_total} features (all models)`;
    } else {
      fsChip.title = fs.note || "";
      fsChip.innerHTML = fs.note ? `Feature selection skipped` : `Feature selection (RFE) requested`;
    }
  } else {
    fsChip.classList.add("hidden");
  }

  const metricNames = [...new Set(results.flatMap((r) => Object.keys(r.metrics || {})))];
  const bestId = (run.best_model || {}).run_id;
  const zebra = results.length > 15;
  const hasCv = metric && results.some((r) => r.cv_metrics && metric in r.cv_metrics);

  let html = `<tr><th>Candidate</th><th>Status</th>${metricNames.map((m) => `<th>${m}</th>`).join("")}${hasCv ? `<th>CV ${escapeHtml(metric)}</th>` : ""}</tr>`;
  for (const r of results) {
    const isBest = r.run_id === bestId;
    html += `<tr class="${isBest ? "best" : ""} ${zebra ? "zebra" : ""}">
      <td>${escapeHtml(r.candidate_name)}${isBest ? '<span class="winner-tag">★ BEST</span>' : ""}</td>
      <td>${escapeHtml(r.status.replaceAll("_", " "))}${r.error ? errorDisclosure(r.error) : ""}</td>
      ${metricNames.map((m) => `<td class="num">${r.metrics && m in r.metrics ? Number(r.metrics[m]).toFixed(4) : "—"}</td>`).join("")}
      ${hasCv ? `<td class="num">${cvCell(r, metric)}</td>` : ""}
    </tr>`;
  }
  $("results-table").innerHTML = html;
}
```

with (chip-filling logic unchanged; table-building code removed since `#exp-table` — populated by `renderExperimentsTable`, already added by the prior Experiments Tab plan — supersedes it; the `#results-card`/`#results-table` elements it used to target no longer exist):

```javascript
function renderResults(run) {
  const results = run.training_results || [];

  const cvConfig = run.cv_config || {};
  const cvChip = $("cv-config-chip");
  if (cvConfig.enabled) {
    cvChip.className = "chip detected cv-config-chip";
    cvChip.innerHTML = `${ICONS.check}${cvConfig.requested_folds}-fold cross-validation requested`;
  } else {
    cvChip.className = "chip cv-config-chip";
    cvChip.innerHTML = `Cross-validation disabled for this run`;
  }

  const resamplingChip = $("resampling-config-chip");
  const resamplingPlan = run.resampling_plan || {};
  if (resamplingPlan.enabled) {
    const applied = results.find((r) => r.resampling_applied)?.resampling_applied || resamplingPlan.method;
    const note = results.find((r) => r.resampling_note)?.resampling_note;
    resamplingChip.classList.remove("hidden");
    resamplingChip.className = "chip detected cv-config-chip";
    resamplingChip.title = note || "";
    resamplingChip.innerHTML = `${ICONS.check}${escapeHtml(applied.replaceAll("_", " "))} applied to training data`;
  } else {
    resamplingChip.classList.add("hidden");
  }

  const fsChip = $("fs-config-chip");
  const fsConfig = run.feature_selection_config || {};
  if (fsConfig.enabled) {
    const fs = run.feature_selection || {};
    fsChip.classList.remove("hidden");
    fsChip.className = "chip detected cv-config-chip";
    if (fs.enabled && fs.n_features_selected != null) {
      fsChip.title = `Selected with ${fs.basic_model || "a basic model"}: ${(fs.selected_features || []).join(", ")}`;
      fsChip.innerHTML = `${ICONS.check}RFE kept ${fs.n_features_selected} of ${fs.n_features_total} features (all models)`;
    } else {
      fsChip.title = fs.note || "";
      fsChip.innerHTML = fs.note ? `Feature selection skipped` : `Feature selection (RFE) requested`;
    }
  } else {
    fsChip.classList.add("hidden");
  }
}
```

Note: `cvCell` (`app.js:2161-2167`) is still used by `renderExperimentsTable` — do not delete it.

- [ ] **Step 4: Repoint the three buttons that navigated to Models Compared**

In `frontend/app.js`, change each of these three lines:

`app.js:1152`:
```javascript
$("champion-compare-btn").addEventListener("click", () => switchRunTab("models"));
```
→
```javascript
$("champion-compare-btn").addEventListener("click", () => switchRunTab("experiments"));
```

`app.js:1183`:
```javascript
$("nextstep-compare-btn").addEventListener("click", () => switchRunTab("models"));
```
→
```javascript
$("nextstep-compare-btn").addEventListener("click", () => switchRunTab("experiments"));
```

`app.js:1249`:
```javascript
$("leaderboard-view-all-btn").addEventListener("click", () => switchRunTab("models"));
```
→
```javascript
$("leaderboard-view-all-btn").addEventListener("click", () => switchRunTab("experiments"));
```

- [ ] **Step 5: Update `RUN_TABS`**

In `frontend/app.js:2678` (already edited by Task 2 to drop `"pipeline"`), remove `"models"` too:

```javascript
const RUN_TABS = ["overview", "experiments", "explainability", "artifacts", "logs"];
```

- [ ] **Step 6: Manual verification**

Start the app via the `run` skill, open a completed run. Confirm the tab bar no longer shows "Models Compared". Confirm `document.getElementById("tab-models-btn")`, `document.getElementById("tab-models-panel")`, and `document.getElementById("results-card")` are all `null`. On the Experiments tab, confirm the CV/resampling/feature-selection chips appear directly above the "All Experiments" table, with correct text (matches whatever `run.cv_config`/`run.resampling_plan`/`run.feature_selection_config` say). Click each of "Compare Models" (champion banner), "Compare Models" (Next Steps rail), and "View all models" (Overview leaderboard card) — confirm all three land on the Experiments tab with no console error.

- [ ] **Step 7: Commit**

```bash
git add frontend/index.html frontend/app.js
git commit -m "feat: remove Models Compared tab, merge config chips into Experiments' All Experiments table"
```

---

## Task 4: Relocate dataset visuals (Dataset Summary, Class Distribution, Data Quality) into a Data section on Experiments

**Files:**
- Modify: `frontend/index.html:395-432` (Overview's `.dash-grid` loses 3 cards), `#tab-experiments-panel`'s `.experiments-main` (gains a new Data section)

**Interfaces:**
- Consumes: nothing new — `renderDatasetSummary(run)`, `renderClassDistribution(run)`, `renderQuality(run)` are unchanged and keep being called from `render()` exactly where they are today (`app.js:1091-1093`); they target elements by id (`dataset-card`, `classdist-card`, `quality-card`, and their children), so relocating the markup requires no JS call-site change.
- Produces: nothing new consumed by later tasks.

- [ ] **Step 1: Remove the 3 cards from Overview's dash-grid**

In `frontend/index.html`, inside the Overview tab's `<section class="dash-grid">` (`index.html:396-432`), delete `dataset-card`, `classdist-card`, and `quality-card`, leaving only `insights-card`:

Before:
```html
      <section class="dash-grid">
        <div class="card hidden" id="insights-card">
          <div class="card-head">
            <h3><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18h6M10 22h4M12 2a6 6 0 0 0-4 10.5c.6.6 1 1.4 1 2.5h6c0-1.1.4-1.9 1-2.5A6 6 0 0 0 12 2Z"/></svg> Auto insights</h3>
            <span class="muted small" id="insights-sub"></span>
          </div>
          <ul class="callout-list insight-list" id="insights-list"></ul>
        </div>

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
      </section>
```

After:
```html
      <section class="dash-grid">
        <div class="card hidden" id="insights-card">
          <div class="card-head">
            <h3><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18h6M10 22h4M12 2a6 6 0 0 0-4 10.5c.6.6 1 1.4 1 2.5h6c0-1.1.4-1.9 1-2.5A6 6 0 0 0 12 2Z"/></svg> Auto insights</h3>
            <span class="muted small" id="insights-sub"></span>
          </div>
          <ul class="callout-list insight-list" id="insights-list"></ul>
        </div>
      </section>
```

- [ ] **Step 2: Add the Data section to Experiments, right after the KPI stat cards**

In `frontend/index.html`, inside `.experiments-main`, immediately after `<section class="stat-row" id="exp-stat-cards"></section>` and before the "Model Performance Overview" card, insert (the 3 relocated cards, each carrying `dense` per the density pass — they're data-density visuals, not hero elements):

```html
            <section class="dash-grid">
              <div class="card dense hidden" id="dataset-card">
                <div class="card-head"><h3>Dataset summary</h3><span class="muted small" id="dataset-sub"></span></div>
                <div class="donut-wrap">
                  <svg id="donut" viewBox="0 0 120 120" role="img" aria-label="Feature type breakdown"></svg>
                  <div class="donut-center" id="donut-center"></div>
                  <ul class="donut-legend" id="donut-legend"></ul>
                </div>
                <div class="chips" id="dataset-chips"></div>
              </div>

              <div class="card dense hidden" id="classdist-card">
                <div class="card-head"><h3>Class distribution</h3><span class="muted small" id="classdist-sub"></span></div>
                <div class="donut-wrap">
                  <svg id="classdist-donut" viewBox="0 0 120 120" role="img" aria-label="Target class distribution"></svg>
                  <div class="donut-center" id="classdist-center"></div>
                  <ul class="donut-legend" id="classdist-legend"></ul>
                </div>
                <div class="chips" id="classdist-chips"></div>
              </div>

              <div class="card dense hidden" id="quality-card">
                <div class="card-head"><h3>Data quality overview</h3><span class="muted small" id="quality-sub"></span></div>
                <div class="quality-wrap">
                  <div class="quality-ring" id="quality-ring"></div>
                  <div class="quality-bars" id="quality-bars"></div>
                </div>
              </div>
            </section>
```

(`.card.dense` is defined in Task 5 — until that task lands, the class has no visual effect, which is fine since these two tasks are independently committable in either order.)

- [ ] **Step 3: Manual verification**

Start the app via the `run` skill, open a completed classification run. Confirm the Overview tab's dashboard section now shows only "Auto insights" (or is entirely hidden if there are no insights). Confirm the Experiments tab shows "Dataset summary", "Class distribution", and "Data quality overview" cards directly below the KPI stat cards, above "Model Performance Overview", with the same donut/ring content that used to appear on Overview. Confirm `document.getElementById("dataset-card").closest("#tab-overview-panel")` is `null` (i.e., it's no longer inside Overview) and `document.getElementById("dataset-card").closest("#tab-experiments-panel")` is truthy.

- [ ] **Step 4: Commit**

```bash
git add frontend/index.html
git commit -m "feat: relocate dataset visuals from Overview into an Experiments Data section"
```

---

## Task 5: Pair the two Experiments chart cards side-by-side; apply the density pass

**Files:**
- Modify: `#tab-experiments-panel`'s `.experiments-main` in `frontend/index.html` (wrap the bar-chart and trend-chart cards, add `dense` to remaining data cards)
- Modify: `frontend/styles.css` (add `.card.dense`)

**Interfaces:** none new.

- [ ] **Step 1: Add the `.card.dense` padding override**

Append to `frontend/styles.css`, right after the `.card`/`.card-head` rules (`styles.css:236-245`):

```css
.card.dense { padding: var(--sp-3); }
```

- [ ] **Step 2: Wrap the bar-chart and trend-chart cards in a 2-column row**

In `frontend/index.html`, inside `.experiments-main`, find the "Model Performance Overview" and "Experiment Trend" cards (added by the prior Experiments Tab plan):

```html
            <div class="card">
              <div class="card-head"><h3>Model Performance Overview</h3><span class="muted small" id="exp-bar-sub"></span></div>
              <div class="exp-bar-chart" id="exp-bar-chart"></div>
              <ul class="exp-bar-legend">
                <li><span class="swatch" style="background:var(--accent-primary)"></span>Best Model</li>
                <li><span class="swatch" style="background:var(--border-subtle)"></span>Other Models</li>
              </ul>
            </div>
            <div class="card">
              <div class="card-head"><h3>Experiment Trend</h3><span class="muted small" id="exp-trend-sub"></span></div>
              <div id="exp-trend-chart"></div>
              <ul class="tuning-legend" id="exp-trend-legend"></ul>
              <p class="muted small hidden" id="exp-trend-empty">No candidates had hyperparameter tuning enabled for this run, so there's no trial-by-trial trend to show.</p>
            </div>
```

Replace with (wrapped in `.dash-grid` for the existing 2-col/980px-breakpoint behavior, both cards marked `dense`):

```html
            <section class="dash-grid">
              <div class="card dense">
                <div class="card-head"><h3>Model Performance Overview</h3><span class="muted small" id="exp-bar-sub"></span></div>
                <div class="exp-bar-chart" id="exp-bar-chart"></div>
                <ul class="exp-bar-legend">
                  <li><span class="swatch" style="background:var(--accent-primary)"></span>Best Model</li>
                  <li><span class="swatch" style="background:var(--border-subtle)"></span>Other Models</li>
                </ul>
              </div>
              <div class="card dense">
                <div class="card-head"><h3>Experiment Trend</h3><span class="muted small" id="exp-trend-sub"></span></div>
                <div id="exp-trend-chart"></div>
                <ul class="tuning-legend" id="exp-trend-legend"></ul>
                <p class="muted small hidden" id="exp-trend-empty">No candidates had hyperparameter tuning enabled for this run, so there's no trial-by-trial trend to show.</p>
              </div>
            </section>
```

- [ ] **Step 3: Mark the 4 distribution donut cards `dense`**

In `frontend/index.html`, inside `.experiments-main`, the 4 donut cards (By Model / By Status / By Outcome / By Compute Time, added by the prior Experiments Tab plan) are each `<div class="card">` inside a `<section class="dash-grid">`. Add `dense` to all 4:

```html
              <div class="card">
                <div class="card-head"><h3>By Model</h3>
```
→
```html
              <div class="card dense">
                <div class="card-head"><h3>By Model</h3>
```

(repeat the same `class="card"` → `class="card dense"` change for the "By Status", "By Outcome", and "By Compute Time" cards in that same donut `dash-grid` section).

- [ ] **Step 4: Manual verification**

Start the app via the `run` skill, open a completed run with tuning enabled on at least one candidate. On the Experiments tab, confirm "Model Performance Overview" and "Experiment Trend" now sit side-by-side in a 2-column row. Resize the browser below 980px and confirm they stack to 1 column. Confirm all data cards (both chart cards, the All Experiments table card, and the 4 donut cards) visually have tighter internal padding than the Best Experiment side panel and the champion banner (open devtools, check computed `padding` on `.card.dense` is `16px` vs `24px` on the Best Experiment panel).

- [ ] **Step 5: Commit**

```bash
git add frontend/index.html frontend/styles.css
git commit -m "feat: pair Experiments chart cards side-by-side and apply density styling to data cards"
```

---

## Task 6: Overview tab rhythm pass (tight grouping, wide separation before caveats/errors)

**Files:**
- Modify: `frontend/index.html` (add `overview-story-card` class to 3 Overview cards)
- Modify: `frontend/styles.css` (rhythm margin rules)

**Interfaces:** none new — purely cosmetic spacing.

- [ ] **Step 1: Tag the "why this model won" story cards**

In `frontend/index.html`, add a shared class to the 3 cards that make up the model-decision narrative (Journey card, Model Leaderboard card, and the "Why This Model?"/"Why Other Models" rationale card — all direct children of `#tab-overview-panel`):

```html
        <section class="card">
          <div class="card-head"><h3>Journey of This Run</h3></div>
```
→
```html
        <section class="card overview-story-card">
          <div class="card-head"><h3>Journey of This Run</h3></div>
```

```html
        <section class="card">
          <div class="card-head">
            <h3>Model Leaderboard</h3>
```
→
```html
        <section class="card overview-story-card">
          <div class="card-head">
            <h3>Model Leaderboard</h3>
```

```html
        <section class="card">
          <div class="rationale-grid">
```
→
```html
        <section class="card overview-story-card">
          <div class="rationale-grid">
```

- [ ] **Step 2: Add the rhythm CSS**

Append to `frontend/styles.css`:

```css
/* ================= overview rhythm ================= */

.overview-story-card + .overview-story-card { margin-top: var(--sp-2); }
#caveats-card { margin-top: var(--sp-5); }
#error-card { margin-top: var(--sp-5); }
```

(`#tab-overview-panel` has no grid/flex of its own — its children stack as plain block boxes with the default zero margin, so these are additive rules creating intentional rhythm rather than overrides of an existing gap.)

- [ ] **Step 3: Manual verification**

Start the app via the `run` skill, open a completed run with at least one caveat and a visible error (or check with devtools by inspecting computed `margin-top` if none are present in your test run — the rule applies regardless of visibility). Confirm the Journey / Leaderboard / Rationale cards sit close together (8px apart), and there's a visibly larger gap (32px) before "Caveats & limitations" and before "Issues encountered" compared to the gap between the story cards.

- [ ] **Step 4: Commit**

```bash
git add frontend/index.html frontend/styles.css
git commit -m "style: add intentional spacing rhythm to Overview tab cards"
```

---

## Task 7: Full end-to-end pass

**Files:** none (verification only)

- [ ] **Step 1: Start the app and drive one real run**

Use the `run` skill to start the server. Upload a small dataset and let a run complete (mock LLM provider is fine per `config/models.yaml`, to avoid real API spend).

- [ ] **Step 2: Verify the full restructured page**

With no console errors at any point:
- Tab bar reads exactly: Overview, Data, Experiments, Explainability, Artifacts, Logs.
- Experiments tab, top to bottom: pipeline stage tracker (compact, since the run is done) → KPI stat cards → Data section (Dataset summary / Class distribution / Data quality) → paired Model Performance Overview + Experiment Trend charts → config chips + All Experiments table → 4 distribution donuts → (aside) Best Experiment panel.
- AI Summary and Recent Activity are collapsed by default; expanding one persists across a page reload.
- All 4 repointed buttons/links (`journey-view-pipeline-btn`, `champion-compare-btn`, `nextstep-compare-btn`, `leaderboard-view-all-btn`) land on Experiments with no error.
- The "Data" nav tab still navigates to the separate Dataset Detail page unchanged.

- [ ] **Step 3: Edge cases**

Drive or fixture-simulate (via devtools console, pasting a modified run object and calling `render(...)`) these cases:
- A run still in progress: stage tracker is NOT compact, shows live progress; Experiments tab is reachable and shows partial data without throwing.
- A run with 0 caveats and 0 errors: `#caveats-card`/`#error-card` stay `hidden` (unaffected by the new margin rules, which only apply spacing when the elements are shown).
- A non-classification (regression) run: `classdist-card` stays hidden in the new Data section (unchanged behavior, just relocated).

- [ ] **Step 4: Final commit (only if Steps 1-3 surfaced fixes)**

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "fix: address edge cases found in end-to-end restructure verification"
```

If no fixes were needed, skip this commit — Tasks 1-6 already cover the feature.
