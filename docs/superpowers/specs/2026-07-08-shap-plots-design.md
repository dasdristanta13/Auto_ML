# SHAP Plots + Explanations — Design

Date: 2026-07-08
Status: approved

## Problem

`explainability_node` (`src/agents/explainability_node.py`) and its backend
(`compute_explainability`/`explain_prediction` in `src/training/dispatch.py`)
already compute a ranked mean-|SHAP| `feature_impact` list, an LLM narrative,
and per-row signed contributions — but the frontend only ever shows this as
custom CSS bar-lists (`fi-list`/`fi-fill`). No actual SHAP visualizations
(beeswarm/summary, real bar plot, dependence plots, waterfall) exist. The
user wants "all relevant SHAP plots with explanation" on both the
Explainability tab (aggregate, per-run) and the Predict tab (per-row).

## Decisions (from brainstorming)

- Aggregate (Explainability tab) plots: summary/beeswarm, bar (mean |SHAP|),
  and dependence plots for the top 3 ranked features. No heatmap, no force
  plot.
- Per-row (Predict tab) plot: waterfall only.
- Rendering: server-side, using SHAP's own matplotlib plotting functions,
  encoded as base64 PNG. The frontend (no-build-step vanilla JS) just embeds
  `<img src="data:image/png;base64,...">` — no new client-side charting
  dependency.
- Captions: aggregate plots (summary, bar, 3 dependence) get one **batched**
  LLM call producing all captions together (adds exactly one extra LLM call
  per run, alongside the existing overall-narrative call). The per-row
  waterfall plot gets a **static** (non-LLM) caption — `/predict` stays
  synchronous with no added LLM latency/cost per prediction.
- The existing custom bar-lists are **replaced** by the new images (Explainability
  tab's `fi-list` → SHAP bar-plot image; Predict tab's "Why" list → waterfall
  image) rather than shown alongside them, to avoid duplicating the same
  ranking in two visual forms.
- Dependence plots are capped to the top 3 features (new
  `explainability.dependence_plot_top_n` config), independent from the
  existing `top_n_features: 8` cap on the narrated/ranked list.

## Architecture

### A. State (`src/state.py`)

```python
class ShapPlot(BaseModel):
    title: str
    feature: Optional[str] = None       # set only for dependence plots
    image_base64: str                   # PNG, base64-encoded
    caption: Optional[str] = None

class ExplainabilityResult(BaseModel):
    method: Literal["tree", "linear", "kernel", "unavailable"]
    feature_impact: list[FeatureImpact] = Field(default_factory=list)
    narrative: Optional[str] = None
    note: Optional[str] = None
    summary_plot: Optional[ShapPlot] = None
    bar_plot: Optional[ShapPlot] = None
    dependence_plots: list[ShapPlot] = Field(default_factory=list)
```

`explain_prediction`'s return type changes from `Optional[list[dict]]` to
`Optional[dict]` shaped:

```python
{"contributions": [{"feature": str, "shap_value": float}, ...],
 "waterfall_plot_base64": Optional[str]}
```

This is a breaking change to `explain_prediction`'s existing contract
(`tests/test_explainability.py`, `src/api/server.py`'s `/predict` route) —
both are updated in the same task since nothing outside this repo depends on
this internal function's shape.

### B. SHAP plot generation (`src/training/dispatch.py`)

- `matplotlib.use("Agg")` set once at module import (headless server — no
  display backend available).
- `_fig_to_base64(fig) -> str`: render the current figure to PNG in a
  `BytesIO`, base64-encode it, always `plt.close(fig)` in a `finally` (avoids
  leaking figures across repeated calls in a long-running server process).
- `compute_explainability` already builds a `shap.Explanation` (`shap_values`)
  via `explainer(background)`, and already reduces it to a 2D array via
  `_reduce_shap_values` for the `feature_impact` ranking (unchanged). Reuse
  that same reduced array to construct a synthetic
  `shap.Explanation(values=reduced, base_values=np.zeros(len(reduced)),
  data=background, feature_names=feature_names)` for plotting — no second
  SHAP computation, no new binary/multiclass reduction logic (already
  covered by existing `_reduce_shap_values` tests). Aggregate plots don't
  need a meaningful base value (beeswarm/bar/scatter show distributions, not
  a cumulative path), so zeros are fine.
- From that synthetic Explanation, each independently wrapped in its own
  try/except (one plot type failing must not block the others or blank out
  `feature_impact`):
  - `shap.plots.beeswarm(explanation, show=False)` → `summary_plot`
  - `shap.plots.bar(explanation, show=False)` → `bar_plot`
  - `shap.plots.scatter(explanation[:, feature_idx], show=False)` for each of
    the top `explainability.dependence_plot_top_n` (default 3) ranked
    features → `dependence_plots` (each tagged with its `feature` name)
- `explain_prediction` builds a **real** single-row `Explanation` (actual
  `base_values` from `explainer.expected_value`, since a waterfall plot's
  entire purpose is showing the cumulative path from base value to the
  final output) and renders `shap.plots.waterfall(row_explanation, show=False)`
  → `waterfall_plot_base64`.
- All plot generation is best-effort: any failure (unsupported shape, SHAP
  internal error, matplotlib error) omits that specific image
  (`None`/omitted from the list) rather than raising — consistent with the
  existing `method="unavailable"` graceful-degradation philosophy already in
  this module.

### C. LLM captions (`src/agents/explainability_node.py`)

- After `compute_explainability` returns (when `method != "unavailable"` and
  at least one plot was generated), make **one** additional LLM call using a
  new prompt `src/agents/prompts/explainability_plot_captions.md`, under a
  distinct node name `"explainability_captions"` (own trace entry, own
  `config/models.yaml` node config, own mock-mode branch in
  `src/llm/client.py`) — separate from the existing `"explainability"`
  narrative call for clean observability per-node.
- Uses `LLMClient.generate(..., json_schema=...)` (already supported) to
  request:
  ```json
  {
    "summary_plot_caption": "string",
    "bar_plot_caption": "string",
    "dependence_plot_captions": {"<feature>": "string", ...}
  }
  ```
- Input to the LLM is only the ranked `feature_impact` list (already used
  for the existing narrative) plus the names of the top 3 dependence
  features — no raw data, no image bytes. The LLM writes "how to read this
  plot type, and what it shows for these specific top features" from
  structured numbers, not by inspecting pixels.
- On failure: caught, appended to `state["errors"]`, captions stay `None` on
  all plots but the images themselves are kept — enrichment, not a
  validation gate, matching how the existing narrative failure is handled.
- The waterfall plot's caption is a **static** string constant (e.g. "Each
  bar shows how much that feature pushed this specific prediction up or
  down from the model's average output.") — defined once in the frontend
  (or as a backend constant returned alongside `waterfall_plot_base64`), no
  LLM call added to `/predict`.

### D. Config

- `config/runtime.yaml`, add to the existing `explainability:` block:
  ```yaml
  explainability:
    top_n_features: 8
    max_background_rows: 100
    dependence_plot_top_n: 3 # number of top-ranked features to render dependence plots for
  ```
- `config/models.yaml`: add an `explainability_captions` node entry (same
  cheap/fast tier as `explainability`).

### E. API (`src/api/server.py`)

- `GET /api/runs/{run_id}/explainability`: unchanged route; response now
  includes `summary_plot`, `bar_plot`, `dependence_plots`. The "not computed
  yet" fallback dict gains the same keys (`None`/`[]`) so the frontend never
  has to special-case a missing key.
- `POST /api/runs/{run_id}/predict`: unpacks the new `explain_prediction`
  dict shape into `result["contributions"]` and
  `result["waterfall_plot_base64"]` (both `None` if `explain_prediction`
  itself returns `None`).

### F. Frontend (`frontend/index.html`, `frontend/app.js`)

- Explainability tab: keep the existing narrative paragraph at the top, then
  replace the current `fi-list`/`fi-fill` bar rendering with, in order: SHAP
  bar-plot image + caption, beeswarm image + caption, then one card per
  dependence plot (image + caption, labeled by feature name).
- Predict tab: replace the current custom "Why" bar-list in
  `renderPredictResult` with the waterfall image + the static caption.
- `method === "unavailable"` / missing plot fields continue to render the
  existing "not available" message instead of a broken `<img>`.

## Error handling

- Any single plot's generation failure → that plot is omitted, others still
  render; `feature_impact`/`method` are unaffected (plotting is a strictly
  additive best-effort layer on top of already-working computation).
- Captions LLM call failure → all plots keep their images, captions are
  `None`, `state["errors"]` gets an entry; no retry (not a validation gate).
- `explain_prediction` failure (as today) → `None`, `/predict` still returns
  the core prediction/probabilities; `contributions` and
  `waterfall_plot_base64` are both `None`.

## Testing

- `tests/test_explainability.py`: each present plot field decodes as a valid
  PNG (`b"\x89PNG"` header); `dependence_plots` length ≤
  `dependence_plot_top_n`; a plot-type failure (e.g. monkeypatched
  `shap.plots.bar` raising) still leaves `feature_impact` and the other
  plots intact; update the two existing `explain_prediction` tests for the
  new dict return shape and add a waterfall-PNG assertion.
- `tests/test_explainability_node.py`: captions call attaches
  `summary_plot.caption`/`bar_plot.caption`/each dependence plot's
  `caption` on success; on failure, images are preserved, captions stay
  `None`, and an entry is appended to `errors`.
- `tests/test_api_explainability.py`: response includes the new plot fields;
  the "not computed yet" fallback includes them as `None`/`[]`; `/predict`
  response includes `waterfall_plot_base64`.
- Run the full `/tests` suite once at the end — `src/training/dispatch.py`
  is a core module touched by this change even though no graph edges change.

## Out of scope

- Client-side/interactive charting (zoom, tooltips) — everything is a static
  server-rendered image.
- Per-row LLM-generated captions (waterfall caption is static).
- Dependence plots beyond the top 3 features.
- Heatmap plot, force plot.
- Caching plot images across requests beyond the existing "aggregate
  computed once after training, per-row computed on demand" split already
  in place for `feature_impact`/`contributions`.
