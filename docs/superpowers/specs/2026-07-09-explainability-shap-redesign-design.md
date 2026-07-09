# Explainability (SHAP) Tab Redesign

**Goal:** Redesign the Explainability tab so it matches the richer reference
layout (stat cards, a Global View / Local Explanation sub-tab switcher, a
Key Insights panel, and a per-example waterfall with "View another example"),
and make sure every SHAP plot carries a visible textual explanation.

**Status:** per-plot captions already exist (`explainability_node` batches one
LLM call that captions the beeswarm/bar/dependence plots — see commit
`4c6292a`). This spec folds that existing capability into the new layout and
adds the pieces that don't exist yet: fidelity/sample stat cards, the
sub-tab switcher, a Key Insights panel, and the Local Explanation view.

**Scope decisions (confirmed with user):**
- Full visual redesign of the Explainability tab, not just captions.
- Two sub-tabs only: **Global View** and **Local Explanation**. **Cohort
  Analysis** (segment comparison from the reference image) is explicitly
  out of scope for this pass — no backend or UI stub for it.
- Plot captions stay deterministic-LLM-authored as they are today (no
  change). The new **Key Insights** bullets are LLM-generated, added as an
  extra field on the *existing* captions LLM call (no new LLM round trip).

## Architecture

No new pipeline nodes or graph edges. All new work extends the existing
`explainability_node` (`src/agents/explainability_node.py`) and
`compute_explainability`/`explain_prediction` (`src/training/dispatch.py`),
and adds one new on-demand function + one new API route for the Local
Explanation sub-tab (per-row SHAP can't be precomputed, same reasoning that
already governs `explain_prediction`).

```
evaluate_node -> explainability_node -> report_node   (unchanged graph shape)
```

## Backend changes

### `src/training/dispatch.py`

`compute_explainability` gains two fields on its return dict, computed from
data it already has in memory (no second SHAP pass):

- `fidelity_r2: float | None` — `r2_score(model_output, base_value +
  shap_values.sum(axis=1))` over the background sample, where
  `model_output` is `model.predict_proba(background)[:, 1]` for binary
  classification or `model.predict(background)` for regression. `None`
  (never raises) when the model output isn't a single scalar per row
  (e.g. multiclass) or the computation fails.
  **Implementation note (added during Task 7 regression pass, commit
  after `4846853`):** for binary classification, `base_value +
  shap_values.sum(axis=1)` isn't always in probability space. sklearn's
  bagged ensembles (e.g. `RandomForestClassifier`) reconstruct
  `predict_proba` directly, but boosting libraries with a log-odds link
  (`XGBClassifier`, `LGBMClassifier`, sklearn's own
  `GradientBoostingClassifier`) reconstruct the pre-sigmoid margin
  instead. Comparing that margin against `predict_proba` without an
  inverse-link transform produced a meaningless, wildly negative R² (a
  live smoke test surfaced `-207.20` for a real LightGBM-champion run)
  even when the SHAP values perfectly reconstructed the model. The fix:
  if the reconstructed value falls outside `[0, 1]`, apply a sigmoid
  before scoring against `predict_proba`. See
  `_shap_fidelity_r2`/`test_shap_fidelity_r2_applies_sigmoid_for_margin_space_reconstruction`
  in `src/training/dispatch.py` / `tests/test_explainability.py`.
- `background_sample_size: int` — `len(background)` (already computed).

New function, alongside `explain_prediction`:

```python
def sample_local_explanation(
    model_path: str,
    transformed_dataset_path: str,
    target_column: str,
    task_type: str,
    time_column: Optional[str],
    seed: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """Best-effort SHAP explanation for one row sampled from the real
    held-out test split (reproduced via the same deterministic _split()
    used at training time — read-only, no leakage risk). Returns None
    rather than raising when SHAP can't explain this estimator/split."""
```

Returns `{"row_values": dict, "prediction": Any, "probability": float |
None, "contributions": [...], "waterfall_plot_base64": str | None,
"base_value": float, "test_set_size": int}` on success, mirroring
`explain_prediction`'s shape plus the row/prediction context needed for
display. Reuses `_split()`, `_build_shap_explainer`, `_reduce_shap_values`,
`_reduce_shap_base_values`, `_render_waterfall_plot` — no new SHAP logic.

Never raises: any failure (missing artifact, split failure, SHAP failure)
returns `None`, and the API/frontend show the same "not available"
empty-state already used elsewhere in this module.

### `src/agents/explainability_node.py`

- Passes `state["target_column"]`, `state["task_type"]`, and
  `state.get("time_column")` (all top-level `PipelineState` fields, see
  `src/state.py:19-25`) through so `compute_explainability`'s stat-card
  fields have what they need. (Note:
  `sample_local_explanation` itself is called on-demand from the API, not
  from this node — the node only needs to store what's needed for
  precomputed stat cards.)
- Extends the existing `_CAPTIONS_JSON_SCHEMA` / `explainability_plot_captions.md`
  prompt call with a new field:

```python
_CAPTIONS_JSON_SCHEMA["properties"]["key_insights"] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "tone": {"type": "string", "enum": ["driver", "risk", "minor"]},
            "message": {"type": "string"},
        },
    },
}
```

  `result["key_insights"] = captions.get("key_insights", [])` stored the
  same way `narrative` and plot captions already are. Same graceful
  degradation: LLM failure leaves `key_insights: []`, tab just hides that
  panel.

### `config/runtime.yaml`

No new caps needed — reuses `explainability.max_background_rows` and
`top_n_features` already defined.

### `src/api/server.py`

- `GET /api/runs/{run_id}/explainability` response gains `fidelity_r2`,
  `background_sample_size`, `key_insights` — all already present on
  `best_model.explainability` once the above lands, so this is a
  pass-through, not new logic.
- New route:

```python
@app.get("/api/runs/{run_id}/explainability/local-example")
def get_local_explanation(
    run_id: str, seed: Optional[int] = None, _session=Depends(require_session)
) -> dict[str, Any]:
    """Computed on demand (not precomputed) — each call can return a
    different sampled row, matching the "View another example" UX."""
```

  Reads `target_column`/`task_type`/`time_column`/`transformed_dataset_path`
  off the stored run state, calls `sample_local_explanation`, and wraps the
  result: `{"available": True, **result}` on success, or
  `{"available": False, "note": "Local explanation isn't available for this
  run."}` when it returns `None` — the frontend keys off `available` to
  decide whether to render the waterfall/method card or the empty state.

## Frontend changes

### `frontend/index.html`

Replace the current flat `tab-explainability-panel` (currently
`index.html:535-542`) with:

```
tab-explainability-panel
├── section.stat-row#explainability-stat-cards        (new; same pattern as #stat-cards / #exp-stat-cards)
├── sub-tab switcher: "Global View" | "Local Explanation"   (new)
├── div#tab-explainability-global (visible by default)
│   └── div.run-layout                                  (reuses existing run-layout/run-rail pattern)
│       ├── main: existing #explainability-plots markup, unchanged
│       └── aside.card#explainability-insights-card      (new; styled like #insights-card)
└── div#tab-explainability-local (hidden by default)
    ├── predicted-value headline + waterfall image
    ├── button#local-example-next ("View another example")
    └── card#local-example-method (Explainer / Model / Test set size / Sampled rows / Baseline / Target class)
```

### `frontend/app.js`

- `renderExplainability(data)` extended to populate the new stat-cards row
  and the Key Insights list (same list-rendering pattern as
  `renderInsights`), in addition to its existing plot rendering (unchanged).
- New `switchExplainabilitySubTab(name)` — same shape as the existing
  `switchRunTab`, scoped to the two sub-tabs.
- New `loadLocalExplanation(run, seed)` — fetches
  `/api/runs/{run_id}/explainability/local-example`, renders the waterfall +
  headline + Data & Method card; called on first visit to the Local
  Explanation sub-tab and again on "View another example" clicks (a fresh
  random `seed` client-side, or omitted so the backend picks one).
- All new rendering follows the existing empty-state convention (`.hidden`
  toggle + a muted-small message) used throughout `app.js` today.

### Visual design

No new CSS framework. Reuses `card`, `stat-row`/`stat-card`,
`insights-list`/`insight-{tone}`, `shap-plot`/`shap-dependence-grid`,
`run-layout`/`run-rail` classes already in `styles.css`. Pixel-level spacing,
color, and icon choices for the new stat cards and Key Insights panel are
worked out live during implementation using the `impeccable` skill against
the reference image, rather than specified upfront here.

## Error handling

Every new piece follows the convention already established across this
module: never raise, degrade to `None`/`[]`/an "unavailable" note, and let
the existing empty-state UI patterns show a clear message instead of
crashing the pipeline, the API, or the tab.

## Testing

Per `CLAUDE.md`, any change to `src/agents/` or a tool/API surface requires:

- Unit tests for `fidelity_r2` computation (binary classification, regression,
  and a degrade-to-`None` case).
- Unit tests for `sample_local_explanation`: reproducible row given a fixed
  `seed`, degrade-to-`None` on missing artifact/bad split.
- API tests for the new `local-example` route (happy path + degrade path),
  and for the extended `/explainability` response fields.
- `explainability_node` test extended to assert `key_insights` is parsed
  from the captions LLM call and defaults to `[]` on failure.
- Full fixture suite (`/tests/fixtures`) run once, per the `src/agents/`
  change rule in `CLAUDE.md`.
- Manual browser verification (mock-LLM mode) of both sub-tabs against a
  real trained run, per the existing plan docs' convention in this repo.

## Out of scope

- Cohort Analysis sub-tab (explicit user decision — future iteration).
- Any change to the Predict tab's existing waterfall/"why" section — it
  already uses `explain_prediction` and is unaffected by this spec.
- New CSS design system or component library.
