# Experiment Page Restructure + Declutter — Design

**Goal:** Consolidate the run view's tab structure (fold Pipeline and Models Compared into the Experiments tab, relocate dataset visuals there too) and reduce visual clutter across the run page, without introducing any new backend endpoints or `PipelineState` fields — this is a pure `frontend/` change (`index.html`, `app.js`, `styles.css`).

**Non-goals:** No changes to the separate Dataset Detail page (`dataset-detail-view`) or its own tabs (Column Summary, Correlations, Missing Values, Outliers) — the "Data" nav tab keeps navigating there unchanged. No backend/API changes. No changes to the Overview tab's champion banner, journey card, leaderboard, or rationale grid content (only their spacing/rhythm, per Section E).

---

## Section A: Collapsible AI Summary / Recent Activity

**Problem:** `#ai-summary-card` and `#activity-card` in the run rail (`#run-rail`) are always fully expanded, adding fixed height to every run view regardless of whether the user wants to read them right now.

**Design:**
- Introduce a reusable `.collapsible-card` pattern: the existing `.card-head` becomes a `<button type="button">` wrapping the `<h3>`, with a chevron icon (rotates 180° on expand/collapse, CSS `transition: transform`, `@media (prefers-reduced-motion: reduce)` fallback to an instant flip).
- Content below the head is wrapped in a `.card-collapsible-body` div, collapsed via `max-height: 0; overflow: hidden` ↔ an expanded `max-height` (measured or a generous fixed cap), transitioning over 200ms.
- Applies to `#ai-summary-card` and `#activity-card` only (not Next Steps or AI Assistant — not in scope).
- Default state: **collapsed** on first load. The user's expand/collapse choice per card persists across reloads via `localStorage` (keyed e.g. `collapse:ai-summary`, `collapse:activity`), so returning to a run doesn't re-collapse a card the user deliberately opened.
- A single `initCollapsible(cardId)` JS helper wires the click handler, `aria-expanded` state, and the `localStorage` read/write; called once per card at init time (not per-render, since the DOM structure is static — only content inside updates on each `render(run)` call).

**Data flow:** No new state. Purely a UI/DOM behavior layered on top of the existing render functions (`renderReportRail` / activity list renderer, whichever populates `#report-body` and `#activity-list` today) — those keep writing into the same inner containers; the collapse wrapper sits around them structurally in `index.html`.

---

## Section B: Data section relocates into Experiments tab

**Problem:** Overview's `dash-grid` today holds `insights-card`, `dataset-card` (feature-type donut), `classdist-card` (target class donut), and `quality-card` (data quality ring) — general dataset visuals mixed into the "why this model won" narrative tab. The Experiments tab is where a user reasons about model search results against the data, so the data visuals belong there instead.

**Design:**
- Move `dataset-card`, `classdist-card`, and `quality-card` markup out of Overview's `.dash-grid` (`index.html:405-431`) into a new `.experiments-data-section` inside `.experiments-main`, positioned directly after the KPI stat cards (`#exp-stat-cards`). Final top-to-bottom order in Experiments (progress bar per Section C sits above everything else): **progress bar → KPI stat cards → Data section (3 cards) → chart row (Section E.2) → chip-row + All Experiments table (Section D) → distribution donuts → (aside) Best Experiment panel**.
- `insights-card` ("Auto insights") stays on Overview — it's model-rationale content, not dataset-shape content, so it's out of scope for this move.
- JS: the `render(run)` call site moves `renderDatasetSummary(run)` and `renderClassDistribution(run)` (currently called at `app.js:1091-1092`) into `renderExperimentsTab(run)` instead; same for whatever function renders `quality-card`/`quality-ring` (currently rendered alongside, per `app.js:1979`). Function bodies are unchanged — only the call site and the DOM location of their target elements move.
- The "Data" nav tab (`tab-data-btn`) is untouched — it still calls `openDatasetDetail(...)` and navigates to the separate Dataset Detail page. This section is additive/relocation only, not a new link.

---

## Section C: Pipeline progress bar moves to top of Experiments; Pipeline tab removed

**Problem:** The stage tracker + live train-progress bar currently live in their own `Pipeline` tab (`#tab-pipeline-panel`), meaning a user has to leave Experiments to check run progress, and the tab disappears in relevance once nothing is "in progress" anymore.

**Design:**
- Move the entire block (`<ol class="stage-tracker" id="stage-tracker">` + `#train-progress`) from `#tab-pipeline-panel` (`index.html:456-470`) to the very top of `#tab-experiments-panel`, above the Data section and KPI cards.
- Behavior while a run is active: unchanged — live-updating stage chips, `#train-progress` with per-candidate tuning rows expanded, exactly as today.
- Behavior once the run reaches a terminal state (`succeeded`/`failed`/`cancelled`): the tracker switches to a new compact CSS variant, `.stage-tracker.compact` — same stage chips, but tighter horizontal spacing, no `#train-progress` detail block shown (it hides, matching today's hidden state when not training).
- Remove `tab-pipeline-btn` and the (now empty) `#tab-pipeline-panel` wrapper from `index.html`. Remove `"pipeline"` from `RUN_TABS` (`app.js:2678`).
- `renderStageTracker(run)` (`app.js:1416`) keeps its logic; only its target container's tab location changes, plus it gains one branch: toggle the `.compact` class based on run status.
- Repoint the one button that navigates to the old Pipeline tab: `journey-view-pipeline-btn` (`app.js:1181`, currently `switchRunTab("pipeline")`) → `switchRunTab("experiments")`.

---

## Section D: Models Compared tab removed, merged into Experiments' All Experiments table

**Problem:** `#tab-models-panel` (`index.html:472-487`) holds a `chip-row` (CV folds / resampling / feature-selection config chips) plus a `results-table` that duplicates almost exactly what the Experiments tab's "All Experiments" table (`#exp-table`, built in the existing Experiments Tab plan) already shows — same candidates, same/similar columns, same ranking.

**Design:**
- Delete `#tab-models-panel` and its `results-card`/`results-table` entirely — `#exp-table` already supersedes it (from the already-implemented Experiments Tab work: `docs/superpowers/plans/2026-07-08-experiments-tab.md` Task 5).
- Move the `chip-row` (`#cv-config-chip`, `#resampling-config-chip`, `#fs-config-chip`) from the deleted panel to sit directly above the "All Experiments" table card in `.experiments-main`.
- Remove `tab-models-btn`; remove `"models"` from `RUN_TABS`.
- Repoint three button handlers that currently do `switchRunTab("models")` to `switchRunTab("experiments")` instead: `champion-compare-btn` (`app.js:1152`), `nextstep-compare-btn` (`app.js:1183`), `leaderboard-view-all-btn` (`app.js:1249`).
- Whatever function currently populates the chip-row's text content (chip fill-in logic near the old `results-card` rendering) is unchanged — only its target elements' DOM location moves alongside the chips themselves.

---

## Section E: Decluttering / spacing rhythm (Impeccable-guided)

Consulted the `impeccable:layout` guidance against this project's own `DESIGN.md` (product register: predictable grids, density-as-feature, structural — not fluid — responsiveness). Diagnosis: the clutter isn't literally "too much empty space" — it's **monotone spacing**: nearly every card uses identical `--sp-4` (24px) padding, and nearly every gap between stacked cards is the same `--sp-3` (16px), so nothing signals relative importance and the page has no rhythm. Fixes below reuse the existing `--sp-1..5` scale (4/8/16/24/32px) — no new spacing tokens.

1. **Card count reduction (free, from Sections B/C/D):** 3 Overview data-cards relocated out, the Pipeline tab's card removed, the Models Compared card removed. Net effect: Overview drops from ~9 stacked cards to ~6; the removed tabs' content isn't lost, just consolidated.
2. **Pair the two Experiments chart cards side-by-side:** the "Model Performance Overview" bar chart and "Experiment Trend" line chart (currently two full-width stacked cards per the existing Experiments Tab plan) go into a single 2-column grid row (`grid-template-columns: 1fr 1fr`, collapsing to 1 column under 980px, matching the existing `.experiments-layout` breakpoint convention) — cuts vertical scroll on the tab that just gained the most content from Sections B/C/D.
3. **Padding varies by density:** hero/decision elements (champion banner, Best Experiment side panel) keep `--sp-4`/`--sp-5` padding; dense data cards (tables, donuts, the two chart cards) drop to `--sp-3` internal padding.
4. **Inter-card rhythm by topic group:** within the Overview "why this model won" story (journey card → leaderboard card → rationale grid), gaps tighten to `--sp-2`; the gap immediately before the caveats/error cards (a distinct, must-notice group) widens to `--sp-5`. This is a targeted override on specific adjacent-card margins, not a global scale change.
5. **Rail collapse (Section A)** further reduces default page height on load.

---

## Cross-cutting implementation notes

- **File touch pattern:** nearly every section above touches the same three files (`index.html` for markup moves, `app.js` for render-call-site moves + `RUN_TABS`/button-handler repoints, `styles.css` for the new `.collapsible-card`, `.stage-tracker.compact`, chart-pairing grid, and padding/rhythm overrides). Because the moves are structural (cut markup from one place, paste into another) rather than independent features, they should be grouped into **one pass per section (A–E)** rather than split further — each section is already the natural task boundary, and further splitting would just fragment single logical moves across artificial task boundaries.
- **Order matters:** implement C and D (tab removal) before B (data section relocation) is *not* required — B/C/D are independent moves into the same target (`.experiments-main`) and can be done in any order, but doing all three before E.2 (chart pairing) avoids re-touching the same markup twice.
- **Verification approach:** unchanged from the existing Experiments Tab plan's convention — this repo's `frontend/` has no test runner; verification is manual via the `run` skill (start the app, open a run, exercise each tab) plus direct devtools console calls against a fixture `run` object for edge cases (no tuning enabled, single candidate, run still in progress vs. terminal state — to check the stage-tracker compact toggle).
- **Regression risk to explicitly check:** every button/link that navigated to `"pipeline"` or `"models"` tabs must be re-verified (Section C/D lists them explicitly) — a missed repoint would silently break navigation with no console error, since `switchRunTab` on a removed tab id would try to toggle a `null` element and throw.

---

## Testing / manual verification checklist (for the implementation plan to expand)

- Tab bar shows exactly: Overview, Data, Experiments, Explainability, Artifacts, Logs (no Pipeline, no Models Compared).
- Experiments tab, top to bottom: progress bar (live while running, compact once done) → KPI cards → Data section (3 cards) → paired chart row → chip-row + All Experiments table → 4 donuts → (aside) Best Experiment panel.
- All 4 repointed buttons (`journey-view-pipeline-btn`, `champion-compare-btn`, `nextstep-compare-btn`, `leaderboard-view-all-btn`) land on the Experiments tab with no console error.
- AI Summary and Recent Activity render collapsed by default; clicking each expands/collapses independently; reduced-motion setting shows an instant toggle instead of an animated one.
- Resize below 980px: Experiments layout stacks to 1 column (existing behavior) and the new paired chart row also stacks to 1 column.
- No `null`-element console errors on any tab switch or on initial load of a fresh run (nothing trained yet) and a completed run.
