# Model Summary "Overview" Tab Design Spec

Date: 2026-07-07
Status: Approved for planning
Source: reference screenshot of an "AutoML.ai" Model Summary page (external design reference, not from this codebase)

## Scope

The reference image shows a full run-results page: a 7-tab bar (Overview / Data / Pipeline / Models Compared / Explainability / Artifacts / Logs), a champion-model banner, a model leaderboard, "why this model" / "why others not selected" panels, a Model Trust Score gauge, a Business Recommendations card, a SHAP beeswarm chart, and an AI chat panel.

Building all of this in one pass mixes a pure frontend layout change with three pieces of new backend computation that don't exist anywhere in the pipeline today. It has been decomposed into four sub-projects:

- **A (this spec)** — tab restructure + Overview tab, wired entirely to data the app already computes and serializes today. No backend changes.
- **B** — Model Trust Score: new deterministic composite in `src/profiling/`, new `PipelineState` field.
- **C** — Business Recommendations: new LLM-backed agent node, structured output, new `PipelineState` field.
- **D** — SHAP Explainability: new `shap`-based computation added to training/evaluation, new capped/aggregated state field, new beeswarm chart component.

**Explicitly out of scope for this spec** (omitted entirely, not stubbed as "coming soon" placeholders, to avoid a half-built card sitting in the layout):
- Model Trust Score gauge and the "confidence in this recommendation" ring inside AI Summary (needs B)
- Business Recommendations card (needs C)
- SHAP Summary beeswarm chart (needs D)
- "Deploy Model", "Generate API", "Schedule Retraining", "Monitor Model" as working actions — stay disabled/"soon", matching the existing sidebar convention (`Model registry <em>soon</em>`, `Deployments <em>soon</em>`, `Monitoring <em>soon</em>`) since no deployment/monitoring feature exists anywhere in this app
- A "Projects" breadcrumb entity — this app has no project grouping (only runs and datasets)

## Why no backend work is needed for this spec

Every widget in scope maps to a run field already returned by `GET /api/runs/{id}` (confirmed in `src/api/server.py`): `best_model`, `training_results`, `task_spec`, `report`, `stage_timeline`, `profile_summary`, `eda_report`/`feature_plan`, `caveats` (via `report`), `insights`, `suggested_questions`, `chat_history`. This is a frontend-only restructure: new markup in `frontend/index.html`, new render functions in `frontend/app.js` (mostly re-skinning existing render functions), and new rules in `frontend/styles.css`.

## Tab structure (`frontend/index.html`, `#report-card` region)

Replace the current two-tab bar (`Report` / `Test the model`) with:

```
Overview | Data | Pipeline | Models Compared | Explainability | Artifacts | Logs
```

- **Overview** — new content, detailed below.
- **Data** — reuses the existing dataset preview/profiling tabs (Column Summary/Correlations/Missing Values/Outliers), scoped to this run's dataset. Reuses `#ptab-*` markup/logic from the dataset-detail view (extracted into a shared render call rather than duplicated).
- **Pipeline** — reuses the existing full stage tracker (`#stage-tracker`, 13 stages) and simplified pipeline logs (`#reasoning-log`), currently only visible while a run is in progress; made visible/read-only here for completed runs too.
- **Models Compared** — reuses `renderResults()`/`#results-table` and `renderTuningTrend()`, restyled as a full-width leaderboard card.
- **Explainability** — reuses `renderFeatureImportance()`/`#fi-list`, full width. Reserved space (empty state: "SHAP analysis not yet available for this run") for sub-project D's beeswarm chart.
- **Artifacts** — new tab; relocates the existing `#download-btn` (best model), `#download-script-btn` (training script), and report export into a file-list card. No new backend endpoint — same three existing links.
- **Logs** — relocates `#trace-toggle-btn`/`#trace-body` (LLM audit trace) here, always visible rather than behind a details/summary toggle.
- **Test the model** (`#tab-test-panel`) is no longer a top-level tab. It becomes a button ("Test this model") on the Overview champion banner that opens the existing predict form in a modal/inline expansion, reusing `loadPredictTab()` unchanged.

Tab switching reuses the existing `switchTab()` pattern, extended from 2 to 7 named panels.

### Breadcrumb

New `.breadcrumb` element above the page header (reusing the exact markup/class used in `#dataset-detail-view`): `Runs > {dataset filename} > {run title}`. Clicking `Runs` behaves like the existing sidebar "Recent runs" navigation (goes to dashboard); clicking the dataset name navigates to `#dataset-detail-view` for that dataset if `source_run_id` resolves to one (per the dataset-identity model established in the dataset-preview spec) — else the segment is non-interactive text.

## Overview tab — left column

All new render functions live in `frontend/app.js` near their existing counterparts, called from a new `renderOverview(run)` dispatched from the same place `renderReport(run)` is today.

1. **Champion banner** (`renderChampionBanner`) — trophy icon, `best_model.candidate_name`, primary metric value (`task_spec.metric` looked up in `best_model.metrics`), delta vs. runner-up (second-best entry in `training_results` sorted by that metric, respecting `tuning.lower_is_better`), accuracy/training-time/CV-fold secondary stats (reusing values already computed in `renderStatCards`). Buttons: "Compare Models" (switches to Models Compared tab), "Download Report" (existing export), "Deploy Model" (disabled, title="Not available in this local build", matching the sidebar convention), "Test this model" (opens predict form).

2. **Journey of This Run** (`renderJourneyCondensed`) — a 6-step condensed view built by grouping `stage_timeline` entries:
   - Data Received ← `profile`
   - Data Inspection ← `leakage_check` + `eda`
   - Feature Engineering ← `feature_engineering` + `apply_feature_plan`
   - Model Search ← `model_selection` + `dispatch_training` + `poll_training`
   - Evaluation ← `evaluate`
   - Champion Selected ← `report` (label reads "Champion Selected", subtext is `best_model.candidate_name`)

   Each group's timestamp is its last constituent stage's completion time; a group renders as done only when all its constituent stages are in `stages_done`. "View Full Pipeline" link switches to the Pipeline tab.

3. **Model Leaderboard (condensed)** (`renderLeaderboardCondensed`) — top rows of `training_results` (all if ≤ 6, else top 5 + champion pinned + "View all N models" link to Models Compared), columns: Model, primary metric, secondary metric (first other metric key present), Training Time, Explainability (★ rating from a static lookup table by `library`/`estimator`: linear models (`LogisticRegression`, `LinearRegression`, `Ridge`, `Lasso`) = 5★, single trees / KNN / Naive Bayes = 4★, Random Forest / Extra Trees = 3★, Gradient Boosting / XGBoost / LightGBM = 2★, anything unrecognized = 3★ default), Champion badge on the winning row. The lookup table is a small constant object in `app.js`, documented inline as a static interpretability heuristic, not a per-run computed metric.

4. **Why This Model? / Why Others Not Selected** (`renderModelRationale`) — deterministic, not LLM text:
   - *Why This Model*: bullet list built from real facts — "Highest {metric} ({value}) among all candidates", "Stable across folds (CV std {std})" when `cv_metrics` present, "Fastest training time" if true, tuning note if `tuning.enabled`.
   - *Why Others Not Selected*: one row per non-champion candidate — metric delta vs. champion, duration ratio vs. champion, and an "Impact" bucket computed as: `duration_ratio > 2 → "High Cost"`, `1.3–2 → "Medium Cost"`, else `"Marginal gain"` if metric delta is small (< 1% relative) regardless of duration. Documented inline as a display heuristic over real numbers, no free-form generation involved.

5. **What AI Did During This Run** (`renderPipelineActions`) — reuses `eda_report.insights` / `feature_plan.steps` (already-rendered `insights-card` content) restyled as a checkmark list: transformations applied, resampling applied (`resampling_applied`), feature selection result (`feature_selection_result`).

6. **Top Drivers** — reuse `renderFeatureImportance()` output (`#fi-list`) unchanged, restyled to match the reference's bar-list look (no logic change).

7. **Caveats & Limitations** — reuse `renderCaveats()`/`#caveats-list` unchanged, restyled with the existing warning icon.

## Overview tab — right rail

1. **AI Summary** — reuses `report` narrative (first paragraph as lede, same as today's `#report-lede`). No confidence ring (deferred to B).
2. **Next Steps grid** — 2×N button grid: "Compare Models" (real, switches tab), "Download Artifacts" (real, switches to Artifacts tab), "Share Report" (real, existing `#share-btn` action), "View SHAP Report" (disabled until D), "Deploy Model" / "Generate API" / "Schedule Retraining" / "Monitor Model" (disabled, "soon", matching sidebar convention).
3. **Ask AI About This Model** — restyle of the existing `#assistant-card` (chat thread, `suggested_questions` chips, input). No logic change, only CSS (pinned input at card bottom, pill-styled suggestion chips matching the reference).

No Model Trust Score card and no Business Recommendations card are rendered in this pass (see Scope).

## Styling

Extends `styles.css`'s existing CSS custom-property theme (`--accent-primary` etc. are already a violet/purple close to the reference's palette — no new palette introduced, per the "adapt to existing theme tokens" decision). New classes: `.champion-banner`, `.journey-condensed`, `.leaderboard-condensed`, `.rationale-grid`, `.next-steps-grid`, all following the existing `.card`/`.chip`/`.stat-card` vocabulary.

## Testing

- No backend changes → no new backend tests required.
- Manual UI verification via the `verify` skill: open a completed run, confirm all 7 tabs render, confirm Overview's derived numbers (leaderboard delta, why-others impact buckets, journey grouping) match the underlying `training_results`/`stage_timeline` data for at least one fixture run in each of: a run with ≤ 3 candidates, a run with tuning enabled, a run with resampling applied, a failed/errored run (caveats + partial leaderboard still render sanely).
- Confirm disabled affordances ("Deploy Model", "View SHAP Report", etc.) are inert and carry the same `title="Not available in this local build"` convention as existing disabled nav items.

## Open questions deferred to implementation

None blocking. The explainability-star lookup table and impact-bucket thresholds above are concrete enough to implement directly; if a candidate's `library`/`estimator` doesn't match any bucket, default to 3★ rather than erroring.
