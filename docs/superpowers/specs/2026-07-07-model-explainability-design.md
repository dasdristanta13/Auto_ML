# Model Explainability (SHAP) — Design

Date: 2026-07-07
Status: approved

## Problem

The frontend already ships an "Explainability" tab (`tab-explainability-panel`,
`frontend/index.html`) and a "View SHAP Report" button, but both are dead
placeholders — static apology text, no fetch, no click handler. The Predict
tab (`renderPredictResult` in `frontend/app.js`) shows a prediction and class
probabilities but has no slot for "why". `shap>=0.45.0` is listed in
`requirements.txt` but never imported anywhere in `src/`. There is no backend
field, endpoint, node, or config for real (SHAP-based) explainability —
`TrainingResult.feature_importance` is `feature_importances_`/`coef_`-based,
not SHAP, and only covers the aggregate case.

## Decisions (from brainstorming)

- Deliver both aggregate (per-run, "what drives this model") and
  per-prediction ("why this row") explanations.
- Aggregate SHAP + its LLM narrative are **precomputed once**, right after
  `evaluate_node` picks `best_model` — only for the winner, not every
  candidate, to avoid multiplying SHAP/LLM cost across discarded candidates.
- Per-prediction SHAP cannot be precomputed (depends on a user-submitted row)
  and stays on-demand inside the existing `/predict` endpoint.
- Include an LLM-narrated summary of top drivers, in addition to the raw
  structured SHAP data, via a new LLM-backed node.
- Include minimal frontend wiring so the feature is actually reachable, not
  just a backend data source nobody calls.

## Architecture

### A. State (`src/state.py`)

```python
class FeatureImpact(BaseModel):
    feature: str
    mean_abs_shap: float

class ExplainabilityResult(BaseModel):
    method: Literal["tree", "linear", "kernel", "unavailable"]
    feature_impact: list[FeatureImpact] = Field(default_factory=list)
    narrative: Optional[str] = None
    note: Optional[str] = None
```

`TrainingResult` gains `explainability: Optional[ExplainabilityResult] = None`.
No new top-level `PipelineState` field — it rides inside `best_model`, the
same way `feature_importance` already does, so `report_node` and the API need
no extra plumbing to see it.

### B. SHAP computation (`src/training/dispatch.py`)

- `_build_shap_explainer(estimator, library, background_df)`: picks
  `shap.TreeExplainer` for sklearn/xgboost/lightgbm tree-based estimators,
  `shap.LinearExplainer` for linear models, `shap.KernelExplainer` (capped to
  `explainability.max_background_rows` background rows) as the fallback for
  everything else (e.g. SVM, KNN).
- `compute_explainability(model_path, transformed_dataset_path)`: loads the
  joblib bundle (same load path as `predict_one`/`load_model_schema`), builds
  a background sample from `transformed_dataset_path` capped at
  `max_background_rows`, runs the explainer, returns mean `|SHAP|` per feature
  ranked and truncated to `explainability.top_n_features` (mirrors the
  existing `_TOP_N_FEATURE_IMPORTANCE` pattern). Wrapped in try/except — any
  failure (unsupported estimator, SHAP internal error) returns
  `method="unavailable"` with an explanatory `note`; never raises.
- `explain_prediction(model_path, values, transformed_dataset_path)`: same
  explainer-building path, applied to the single submitted row, returns
  `[{feature, shap_value}]` capped to `explainability.top_n_features`. Used
  on-demand by `/predict`. Same graceful-fallback behavior on failure (returns
  `None`/empty rather than erroring the whole predict call).

### C. New node (`src/agents/explainability_node.py`)

LLM-backed, one file per convention. After `evaluate_node`:

1. Calls `compute_explainability(best_model["model_path"], state["transformed_dataset_path"])`.
2. If `method != "unavailable"`, prompts the LLM (new
   `src/agents/prompts/explainability.md`) with only the ranked
   `feature_impact` list — feature names and numbers, ≤ `top_n_features`
   entries, never raw rows — to produce a short plain-language narrative
   (e.g. "Income and Age were the strongest drivers..."). If SHAP itself was
   unavailable, skip the LLM call and leave `narrative=None` with the `note`
   explaining why.
3. Writes the `ExplainabilityResult` onto `state["best_model"]["explainability"]`.

If the LLM call fails or returns empty content, catch it and proceed with
`narrative=None` — this is enrichment, not a validation gate, so it never
blocks or retries into `report_node`.

### D. Graph wiring (`src/graph/build_graph.py`)

Insert the node between `evaluate` and `report` in both the pre-checkpoint and
full graphs:

```python
graph.add_node("explainability", explainability_node)
...
graph.add_edge("evaluate", "explainability")
graph.add_edge("explainability", "report")
```

(replaces the current direct `evaluate -> report` edge in both places it
appears in `build_graph.py`).

### E. Config

- `config/runtime.yaml`: add
  ```yaml
  explainability:
    top_n_features: 8
    max_background_rows: 100
  ```
- `config/models.yaml`: add an `explainability` node entry, using a cheap/fast
  model tier (same class as `chat`/routing nodes — it narrates 8 numbers, not
  a planning task).

### F. API (`src/api/server.py`)

- `GET /api/runs/{run_id}/explainability` — new endpoint. `_require_model_path`
  guard, then returns `entry["best_model"].get("explainability")` through
  `_json_safe`. Purely a read of the precomputed value — no recomputation on
  request, matching the "computed once, cached with the run" decision.
- `POST /api/runs/{run_id}/predict` — response gains a `contributions` field:
  `explain_prediction(model_path, body.values, transformed_dataset_path)`,
  wrapped the same way existing predict errors are (400, not 500, on failure).
  `contributions` may be `null` if explanation wasn't possible for this
  estimator.

### G. Frontend (`frontend/index.html`, `frontend/app.js`)

- `tab-explainability-panel`: on tab activation (extending the existing
  `switchRunTab` dispatch, same lazy-load pattern already used elsewhere),
  fetch `GET /api/runs/{run_id}/explainability` and render a bar list reusing
  the existing `fi-list`/`fi-fill` CSS classes, plus the `narrative` text.
  Replace the static apology paragraph with this rendered content. Remove the
  disabled "View SHAP Report" button entirely — the Explainability tab itself
  now is the report, so there's nothing left for that button to link to.
  When `method === "unavailable"`, show the `note` in place of the chart.
- `renderPredictResult()`: append a small "Why" block listing
  `contributions` (feature + signed value) when present, styled consistently
  with the existing probability bars; omitted entirely when `contributions`
  is `null`.
- The unrelated cosmetic `explainabilityStars()` keyword-based leaderboard
  column and the Overview "Top Drivers" card (`feature_importance`-based) are
  left untouched — they are a different, already-working feature.

## Error handling

- Unsupported estimator / SHAP internal failure → `method="unavailable"` +
  `note`, no exception surfaces past `compute_explainability`/
  `explain_prediction`; pipeline and predict endpoint continue normally.
- LLM narrative call failure → `narrative=None`, pipeline continues to
  `report_node` (no retry loop; this isn't a validation gate).
- Frontend: `method === "unavailable"` or `contributions === null` render an
  explanatory message instead of an empty/broken chart.

## Testing

- `tests/test_explainability.py`: SHAP feature-impact shape/ranking for a
  tree-based estimator and a linear estimator; graceful `"unavailable"`
  fallback for an estimator SHAP can't handle; `max_background_rows` and
  `top_n_features` caps are respected; `explain_prediction` returns per-row
  contributions capped the same way.
- Extend the `/tests/fixtures` full-suite run (required for any
  `src/graph/` change per CLAUDE.md) to confirm `explainability_node` doesn't
  break existing fixture pipelines, including the PII and high-cardinality
  fixtures.
- No new LLM-facing tool is introduced, so no new `test_tool_caps.py`
  assertion is required — but confirm by inspection that the prompt built in
  `explainability_node` only ever receives the ranked `feature_impact` list,
  never a dataframe or raw sample.

## Out of scope

- Per-candidate (non-winner) SHAP or narratives.
- A dedicated "compare candidates' explanations" UI.
- Caching/store of per-prediction contributions across requests (each
  `/predict` call recomputes for its submitted row).
- Changes to the existing `feature_importance` (`feature_importances_`/
  `coef_`-based) Overview card or the cosmetic leaderboard stars.
