# Experiments tab — design

## Context

The run view (`frontend/index.html`, `frontend/app.js`) currently has a text-heavy
Overview tab (journey narrative, leaderboard, "why this / why not others" prose)
and a separate Models Compared tab (raw results table + single-line tuning
trend for the champion only). The user wants a visual, dashboard-style
"Model Experiment Summary" page, modeled on a reference screenshot from a
different multi-project AutoML SaaS (AutoML.ai), added as its own tab.

The reference image assumes a scale we don't have: 120 experiments run across
a day across many projects, with per-model wall-clock trend lines and
outcome-vs-baseline donuts. Our app trains one run's candidate models once
(typically 5–10 candidates), each optionally Optuna-tuned. Every section below
is mapped to data that actually exists in `PipelineState` / `TrainingResult`
(`src/state.py`) — nothing is fabricated to match the image's exact numbers or
axes.

## Non-goals

- No per-candidate drill-down modal/detail route (the image's row-level
  chart-icon / `...` actions menu is dropped — there's nothing to open).
- No wall-clock "trend by time of day" — trial history has no per-trial
  timestamp, so the trend chart uses trial index on the x-axis instead.
- Does not touch the Overview or Models Compared tabs' existing content.

## Placement

New tab `Experiments`, inserted between `Pipeline` and `Models Compared`:

```
Overview | Data | Pipeline | Experiments | Models Compared | Explainability | Artifacts | Logs
```

- `index.html`: new `<button id="tab-experiments-btn">` in `.run-tab-bar`
  (`index.html:189-196`), new `<div id="tab-experiments-panel" class="hidden">`
  section following the same pattern as `tab-pipeline-panel` /
  `tab-models-panel`.
- `app.js`: add `"experiments"` to `RUN_TABS` (`app.js:2376`) so
  `switchRunTab` wires it up automatically. Render unconditionally from the
  master `render(run)` function (`app.js:1053-1103`), the same convention
  every other tab's sections already follow (they're all cheap re-derivations
  of the in-memory `run` object on each 1.5s poll tick, hidden via CSS when
  their tab isn't active). This does NOT use the `explainability` tab's
  lazy-load-on-click pattern (`app.js:2387`), since that pattern exists
  specifically because Explainability needs an extra network call the other
  tabs don't.

## Data mapping

All sections read from `run.training_results` (`TrainingResult[]`),
`run.best_model`, `run.task_spec.metric`, `run.cv_config`,
`run.resampling_plan`, and `run.created_at`.

### KPI stat cards (5)

| Card | Value |
|---|---|
| Total Experiments | `sum(r.tuning.trials_done || 1 for r in results)` — every Optuna trial across every candidate, not just candidate count |
| Models Evaluated | `results.length` |
| Best CV Score | champion's `cv_metrics[metric].mean`, falling back to `metrics[metric]` if CV was disabled; label includes the actual metric name (varies by task) |
| Avg Training Time | mean of `duration_seconds` across results |
| Total Compute Time | sum of `duration_seconds` |

### Model Performance Overview (bar chart)

One bar per candidate, height = primary metric value, sorted per the metric's
direction (reuse the sort in `renderLeaderboardCondensed`, `app.js:1220-1228`).
Champion bar rendered in the accent color; others muted, matching the existing
"Best Model / Other Models" legend convention used elsewhere in the UI.

### Experiment Trend (line chart)

Replaces the current single-line champion-only tuning chart
(`renderTuningTrend`, `app.js:2090-2149`) with a multi-series version: one
line per candidate where `tuning.history.length > 1`. X-axis = trial index
(not wall-clock time — no per-trial timestamp exists in `TuningInfo.history`).
Candidates with tuning disabled/skipped are omitted from the chart, with a
caption noting how many of N candidates are shown. If zero candidates were
tuned, render an empty state instead of hiding the card.

Needs more than the 4 existing `--cat-1..4` colors for 5-10 series; the
dataviz skill will be used when implementing to extend/validate the palette
for both light and dark themes rather than reusing only 4 hues by rotation.

### All Experiments table

Consolidates `renderLeaderboardCondensed` (`app.js:1207-1246`) and
`renderResults` (`app.js:2013-2078`) into one table: Rank, Model,
Trials (`trials_done`/`trials_total`, or "no tuning"), primary metric
(CV mean ± std when available, via existing `cvCell` helper), secondary
metric, Training Time, Status, Champion badge. Failed rows keep the existing
inline error disclosure. No Actions column (see Non-goals).

### Distribution donuts (4)

Reuses the existing donut SVG pattern (`renderDatasetSummary`,
`app.js:1867-1920`).

| Donut | Buckets |
|---|---|
| By Model | trial-count share per candidate (reflects compute spent, not just candidate count) |
| By Status | grouped by whatever distinct `result.status` values are present (`succeeded` / `failed` / `timed_out` / `running` / `pending` — the last two only while a run is still in progress) |
| By Outcome | Champion / Close contender (candidate's primary metric within 2% relative of the champion's) / Trailed (beyond that) — computed vs the champion, since there's no prior-run baseline to diff against like the reference image |
| By Compute Time | `duration_seconds` bucketed <30s / 30s–2m / >2m |

### Best Experiment side panel

Champion name + trophy icon, trial count, status, key metrics (CV mean/std
for primary + secondary metric), training info block: Training Time
(`duration_seconds`), Start Time (run `created_at`), Data Used (resampling
method + class ratio if applied, else "no resampling"), Folds (`cv_folds`).
No "View Full Details" button (no separate detail route exists).

## Visual language

Follows the existing dashboard theme already modeled on this same reference
(`styles.css:1-5`) — violet accent, `stat-card`/`dash-grid`/`donut-wrap`
classes, light/dark CSS custom properties. No new design system is
introduced; this is new content within the established one.

## Files touched

- `frontend/index.html` — new tab button + panel markup
- `frontend/app.js` — `RUN_TABS` entry, new render functions
  (`renderExperimentsTab` orchestrator + per-section helpers), multi-series
  rework of the tuning trend chart
- `frontend/styles.css` — bar chart classes, extended categorical palette
  entries (validated for light + dark per the dataviz skill), any new layout
  needed for the side-panel + donut grid within the Experiments panel

## Testing

Manual verification via the `run` skill: start the app, drive a run to
completion (or open an existing completed run), open the Experiments tab, and
confirm each section renders against real data — including edge cases: a run
with only one candidate (bar/table degenerate to one row, trend chart empty
state), a run with tuning disabled entirely (trend chart empty state), and a
run with a failed candidate (status donut + table error disclosure).
