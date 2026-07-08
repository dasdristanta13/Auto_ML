# SHAP Plots + Explanations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real server-rendered SHAP plots (beeswarm/summary, bar, top-3 dependence plots on the Explainability tab; a waterfall plot on the Predict tab) with LLM-written captions for the aggregate plots and a static caption for the waterfall, replacing the existing custom HTML bar-lists.

**Architecture:** `compute_explainability`/`explain_prediction` (`src/training/dispatch.py`) already build a `shap.Explanation` for the aggregate case and a per-row one for the on-demand case. This plan reuses those same reduced SHAP arrays to build lightweight `shap.Explanation` objects purely for plotting, renders each plot via SHAP's own matplotlib functions (`shap.plots.beeswarm`/`bar`/`scatter`/`waterfall`) with `matplotlib.use("Agg")`, and returns each as a base64 PNG. `explainability_node` gets one extra batched LLM call (`node="explainability_captions"`) to caption the three aggregate plot types; the waterfall's caption is a static string, added directly in the frontend, keeping `/predict` LLM-free.

**Tech Stack:** Python 3.11, `shap>=0.45.0` (already a dependency), `matplotlib` (added as an explicit direct dependency — used directly via `pyplot`, not just internally by shap), FastAPI, vanilla JS frontend (no build step).

## Global Constraints

- Non-negotiable: raw data never enters an LLM prompt — the new `explainability_captions` LLM call receives only the ranked `feature_impact` list and the names of the top dependence features, never a dataframe, image bytes, or row sample.
- Any single plot's generation failure must degrade gracefully (that plot omitted, `None`/`[]`) and must never blank out `feature_impact`/`method` or block the other plots.
- The captions LLM call failure must never block the pipeline — captions stay `None`, images are kept, failure is logged to `state["errors"]` (enrichment, not a validation gate — same as the existing narrative call).
- `/predict` stays LLM-free and synchronous — the waterfall caption is static, not LLM-generated.
- `config/runtime.yaml`'s `explainability.dependence_plot_top_n` (new) is the only cap read for dependence plot count — no hardcoded numbers inline.
- Spec: `docs/superpowers/specs/2026-07-08-shap-plots-design.md`.

---

### Task 1: SHAP plot generation core (`src/training/dispatch.py`)

**Files:**
- Modify: `src/state.py` (add `ShapPlot`, extend `ExplainabilityResult`)
- Modify: `config/runtime.yaml` (add `dependence_plot_top_n`)
- Modify: `requirements.txt` (add explicit `matplotlib` dependency)
- Modify: `src/training/dispatch.py` (imports, `_fig_to_base64`, `_shap_plot_explanation`, `_render_beeswarm_plot`, `_render_bar_plot`, `_render_dependence_plots`, `_reduce_shap_base_values`, `_render_waterfall_plot`; extend `compute_explainability` and `explain_prediction`)
- Test: `tests/test_explainability.py`

**Interfaces:**
- Produces: `compute_explainability(...)` return dict gains `"summary_plot"`, `"bar_plot"`, `"dependence_plots"` keys (each `None`/`[]` on failure, `{"title": str, "feature": Optional[str], "image_base64": str, "caption": None}` shape on success).
- Produces: `explain_prediction(model_path, values, transformed_dataset_path) -> Optional[dict[str, Any]]` — return type changes from `Optional[list[dict]]` to `Optional[dict]` shaped `{"contributions": [...], "waterfall_plot_base64": Optional[str]}`.
- Consumes: nothing new from other tasks — this is the foundation layer, same as Task 1 was for the original explainability feature.

- [ ] **Step 1: Add `ShapPlot` to state and extend `ExplainabilityResult`**

In `src/state.py`, replace the existing `ExplainabilityResult` class (currently lines 78-82):

```python
class ExplainabilityResult(BaseModel):
    method: Literal["tree", "linear", "kernel", "unavailable"]
    feature_impact: list[FeatureImpact] = Field(default_factory=list)
    narrative: Optional[str] = None
    note: Optional[str] = None
```

with:

```python
class ShapPlot(BaseModel):
    title: str
    feature: Optional[str] = None
    image_base64: str
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

- [ ] **Step 2: Add `dependence_plot_top_n` to `config/runtime.yaml`**

Change the existing `explainability:` block:

```yaml
explainability:
  top_n_features: 8 # cap on ranked SHAP features returned/narrated
  max_background_rows: 100 # background/reference sample size for SHAP explainers
```

to:

```yaml
explainability:
  top_n_features: 8 # cap on ranked SHAP features returned/narrated
  max_background_rows: 100 # background/reference sample size for SHAP explainers
  dependence_plot_top_n: 3 # number of top-ranked features to render dependence plots for
```

- [ ] **Step 3: Add `matplotlib` as an explicit dependency**

In `requirements.txt`, change:

```
# reporting / explainability
shap>=0.45.0
```

to:

```
# reporting / explainability
shap>=0.45.0
matplotlib>=3.8
```

- [ ] **Step 4: Write the failing tests**

Replace `tests/test_explainability.py` in full with:

```python
"""Aggregate (per-run) and per-prediction SHAP-based explainability, plus
the SHAP plots layered on top (docs/superpowers/specs/2026-07-08-shap-plots-
design.md). compute_explainability/explain_prediction degrade to
method="unavailable" / None rather than raising; a single plot's failure
must never block the other plots or blank out feature_impact."""

from __future__ import annotations

import base64

import numpy as np
import pandas as pd
import pytest

from src.training.dispatch import (
    _build_shap_explainer,
    _reduce_shap_base_values,
    _reduce_shap_values,
    _registry,
    _render_dependence_plots,
    _run_job,
    _shap_method_label,
    _shap_plot_explanation,
    compute_explainability,
    explain_prediction,
)

_PNG_HEADER = b"\x89PNG"


def _train(tag: str, dataset_path, estimator: str, hyperparams: dict) -> dict:
    _registry[tag] = {"status": "pending", "feature_selection": {"enabled": False}}
    _run_job(
        tag, str(dataset_path), "target", "classification", "sklearn",
        estimator, hyperparams, None, [], False, None, False, "none", False, None, None,
    )
    result = _registry[tag]
    assert result["status"] == "succeeded", result.get("error")
    return result


@pytest.fixture()
def trained_tree_model(tmp_path):
    rng = np.random.default_rng(3)
    n = 150
    df = pd.DataFrame({"x1": rng.random(n), "x2": rng.random(n), "x3": rng.random(n), "target": rng.integers(0, 2, n)})
    dataset_path = tmp_path / "tree.csv"
    df.to_csv(dataset_path, index=False)
    result = _train("expl-tree", dataset_path, "RandomForestClassifier", {"n_estimators": 10, "max_depth": 3, "random_state": 0})
    return result["model_path"], str(dataset_path)


@pytest.fixture()
def trained_linear_model(tmp_path):
    rng = np.random.default_rng(4)
    n = 150
    df = pd.DataFrame({"x1": rng.random(n), "x2": rng.random(n), "target": rng.integers(0, 2, n)})
    dataset_path = tmp_path / "linear.csv"
    df.to_csv(dataset_path, index=False)
    result = _train("expl-linear", dataset_path, "LogisticRegression", {"max_iter": 500})
    return result["model_path"], str(dataset_path)


def test_compute_explainability_tree_model(trained_tree_model):
    model_path, dataset_path = trained_tree_model
    result = compute_explainability(model_path, dataset_path)
    assert result["method"] == "tree"
    assert result["note"] is None
    assert 1 <= len(result["feature_impact"]) <= 8
    assert {f["feature"] for f in result["feature_impact"]} <= {"x1", "x2", "x3"}
    assert all(f["mean_abs_shap"] >= 0 for f in result["feature_impact"])


def test_compute_explainability_linear_model(trained_linear_model):
    model_path, dataset_path = trained_linear_model
    result = compute_explainability(model_path, dataset_path)
    assert result["method"] == "linear"
    assert result["note"] is None
    assert result["feature_impact"]


def test_compute_explainability_caps_to_top_n_features(tmp_path):
    rng = np.random.default_rng(5)
    n = 150
    data = {f"f{i}": rng.random(n) for i in range(12)}
    data["target"] = rng.integers(0, 2, n)
    dataset_path = tmp_path / "wide.csv"
    pd.DataFrame(data).to_csv(dataset_path, index=False)
    result = _train("expl-wide", dataset_path, "RandomForestClassifier", {"n_estimators": 10, "max_depth": 3, "random_state": 0})

    explainability = compute_explainability(result["model_path"], str(dataset_path))
    assert len(explainability["feature_impact"]) == 8


def test_compute_explainability_never_raises_on_missing_artifact(tmp_path):
    result = compute_explainability(str(tmp_path / "missing.joblib"), str(tmp_path / "missing.csv"))
    assert result["method"] == "unavailable"
    assert result["feature_impact"] == []
    assert "SHAP explanation unavailable" in result["note"]
    assert result["summary_plot"] is None
    assert result["bar_plot"] is None
    assert result["dependence_plots"] == []


def test_compute_explainability_includes_plots(trained_tree_model):
    model_path, dataset_path = trained_tree_model
    result = compute_explainability(model_path, dataset_path)

    assert result["summary_plot"] is not None
    assert base64.b64decode(result["summary_plot"]["image_base64"])[:4] == _PNG_HEADER
    assert result["bar_plot"] is not None
    assert base64.b64decode(result["bar_plot"]["image_base64"])[:4] == _PNG_HEADER
    assert 1 <= len(result["dependence_plots"]) <= 3
    for plot in result["dependence_plots"]:
        assert plot["feature"] in {"x1", "x2", "x3"}
        assert base64.b64decode(plot["image_base64"])[:4] == _PNG_HEADER


def test_compute_explainability_caps_dependence_plots_to_config(tmp_path):
    rng = np.random.default_rng(5)
    n = 150
    data = {f"f{i}": rng.random(n) for i in range(12)}
    data["target"] = rng.integers(0, 2, n)
    dataset_path = tmp_path / "wide-plots.csv"
    pd.DataFrame(data).to_csv(dataset_path, index=False)
    result = _train("expl-wide-plots", dataset_path, "RandomForestClassifier", {"n_estimators": 10, "max_depth": 3, "random_state": 0})

    explainability = compute_explainability(result["model_path"], str(dataset_path))
    assert len(explainability["dependence_plots"]) == 3


def test_explain_prediction_returns_ranked_row_contributions(trained_tree_model):
    model_path, dataset_path = trained_tree_model
    result = explain_prediction(model_path, {"x1": 0.9, "x2": 0.1, "x3": 0.5}, dataset_path)
    assert result is not None
    contributions = result["contributions"]
    assert 1 <= len(contributions) <= 8
    assert {c["feature"] for c in contributions} <= {"x1", "x2", "x3"}


def test_explain_prediction_returns_waterfall_plot(trained_tree_model):
    model_path, dataset_path = trained_tree_model
    result = explain_prediction(model_path, {"x1": 0.9, "x2": 0.1, "x3": 0.5}, dataset_path)
    assert result is not None
    assert result["waterfall_plot_base64"] is not None
    assert base64.b64decode(result["waterfall_plot_base64"])[:4] == _PNG_HEADER


def test_explain_prediction_returns_none_on_failure(tmp_path):
    assert explain_prediction(str(tmp_path / "missing.joblib"), {}, str(tmp_path / "missing.csv")) is None


def test_reduce_shap_values_keeps_positive_class_for_binary():
    values = np.zeros((2, 3, 2))
    values[:, :, 1] = 5.0
    reduced = _reduce_shap_values(values)
    assert reduced.shape == (2, 3)
    assert (reduced == 5.0).all()


def test_reduce_shap_values_averages_multiclass():
    values = np.array([[[1.0, 2.0, 3.0]]])  # 1 sample, 1 feature, 3 classes
    reduced = _reduce_shap_values(values)
    assert reduced.shape == (1, 1)
    assert reduced[0][0] == pytest.approx(2.0)


def test_reduce_shap_base_values_keeps_positive_class_for_binary():
    values = np.array([[1.0, 5.0], [2.0, 6.0]])
    reduced = _reduce_shap_base_values(values)
    assert list(reduced) == [5.0, 6.0]


def test_reduce_shap_base_values_passes_through_1d():
    values = np.array([1.0, 2.0, 3.0])
    reduced = _reduce_shap_base_values(values)
    assert list(reduced) == [1.0, 2.0, 3.0]


def test_shap_method_label_falls_back_to_kernel_for_unsupported_estimator():
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(1)
    X = rng.random((60, 3))
    y = rng.integers(0, 2, 60)
    model = KNeighborsClassifier().fit(X, y)
    explainer = _build_shap_explainer(model, X[:20], ["a", "b", "c"])
    assert _shap_method_label(explainer) == "kernel"


def test_render_dependence_plots_skips_unknown_feature_without_raising():
    explanation = _shap_plot_explanation(
        np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[0.1, 0.2], [0.3, 0.4]]), ["a", "b"]
    )
    plots = _render_dependence_plots(explanation, ["a", "does-not-exist", "b"], 3)
    assert {p["feature"] for p in plots} == {"a", "b"}
```

- [ ] **Step 5: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_explainability.py -v`
Expected: FAIL with `ImportError: cannot import name '_shap_plot_explanation' from 'src.training.dispatch'` (none of the new helpers exist yet), plus `KeyError`/assertion failures on the plot fields and the `explain_prediction` dict shape.

- [ ] **Step 6: Update the imports at the top of `src/training/dispatch.py`**

Change:

```python
from __future__ import annotations

import inspect
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
```

to:

```python
from __future__ import annotations

import base64
import inspect
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - backend must be set before pyplot import
import numpy as np
import pandas as pd
import yaml
```

- [ ] **Step 7: Add the plot-rendering helpers**

Add this block right after `_shap_background` (currently lines 374-377), before `compute_explainability`:

```python
def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        plt.close(fig)


def _shap_plot_explanation(values: np.ndarray, background: np.ndarray, feature_names: list[str]):
    """Synthetic Explanation reusing the already-reduced 2D SHAP array (no
    second SHAP computation, no new binary/multiclass reduction logic).
    base_values are zeroed — the aggregate plots below (beeswarm/bar/scatter)
    show distributions, not a cumulative path, so a meaningful base value
    isn't needed here (contrast with _render_waterfall_plot, which uses a
    real one)."""
    import shap

    return shap.Explanation(
        values=values,
        base_values=np.zeros(len(values)),
        data=background,
        feature_names=feature_names,
    )


def _render_beeswarm_plot(explanation) -> Optional[dict[str, Any]]:
    import shap

    try:
        shap.plots.beeswarm(explanation, show=False)
        return {
            "title": "Impact distribution (beeswarm)",
            "feature": None,
            "image_base64": _fig_to_base64(plt.gcf()),
            "caption": None,
        }
    except Exception:  # noqa: BLE001 - one failing plot must not block the others
        return None


def _render_bar_plot(explanation) -> Optional[dict[str, Any]]:
    import shap

    try:
        shap.plots.bar(explanation, show=False)
        return {
            "title": "Feature impact (bar)",
            "feature": None,
            "image_base64": _fig_to_base64(plt.gcf()),
            "caption": None,
        }
    except Exception:  # noqa: BLE001 - one failing plot must not block the others
        return None


def _render_dependence_plots(explanation, ranked_feature_names: list[str], top_n: int) -> list[dict[str, Any]]:
    import shap

    plots: list[dict[str, Any]] = []
    for feature_name in ranked_feature_names[:top_n]:
        try:
            idx = explanation.feature_names.index(feature_name)
            shap.plots.scatter(explanation[:, idx], show=False)
            plots.append(
                {
                    "title": f"Dependence: {feature_name}",
                    "feature": feature_name,
                    "image_base64": _fig_to_base64(plt.gcf()),
                    "caption": None,
                }
            )
        except Exception:  # noqa: BLE001 - one failing feature's plot must not block the others
            continue
    return plots


def _reduce_shap_base_values(base_values: np.ndarray) -> np.ndarray:
    """Mirrors _reduce_shap_values' binary/multiclass reduction, applied to
    base_values instead of per-feature contributions, so a waterfall plot's
    starting point matches the class its bars explain."""
    arr = np.asarray(base_values)
    if arr.ndim == 1:
        return arr
    if arr.shape[-1] == 2:
        return arr[:, 1]
    return arr.mean(axis=-1)


def _render_waterfall_plot(
    row_values: np.ndarray, base_value: float, row_data: np.ndarray, feature_names: list[str]
) -> Optional[str]:
    import shap

    try:
        explanation = shap.Explanation(
            values=row_values,
            base_values=base_value,
            data=row_data,
            feature_names=feature_names,
        )
        shap.plots.waterfall(explanation, show=False)
        return _fig_to_base64(plt.gcf())
    except Exception:  # noqa: BLE001 - waterfall is best-effort, contributions list is still returned
        return None
```

- [ ] **Step 8: Wire plot generation into `compute_explainability`**

Change:

```python
        ranked = sorted(zip(feature_names, mean_abs), key=lambda pair: pair[1], reverse=True)
        top_n = cfg["top_n_features"]
        return {
            "method": _shap_method_label(explainer),
            "feature_impact": [
                {"feature": name, "mean_abs_shap": round(float(value), 6)} for name, value in ranked[:top_n]
            ],
            "narrative": None,
            "note": None,
        }
```

to:

```python
        ranked = sorted(zip(feature_names, mean_abs), key=lambda pair: pair[1], reverse=True)
        top_n = cfg["top_n_features"]

        explanation = _shap_plot_explanation(values, background, feature_names)
        summary_plot = _render_beeswarm_plot(explanation)
        bar_plot = _render_bar_plot(explanation)
        dependence_plots = _render_dependence_plots(
            explanation, [name for name, _ in ranked], cfg["dependence_plot_top_n"]
        )

        return {
            "method": _shap_method_label(explainer),
            "feature_impact": [
                {"feature": name, "mean_abs_shap": round(float(value), 6)} for name, value in ranked[:top_n]
            ],
            "narrative": None,
            "note": None,
            "summary_plot": summary_plot,
            "bar_plot": bar_plot,
            "dependence_plots": dependence_plots,
        }
```

Also update the `except` branch immediately below (the `unavailable` fallback) to include the new keys:

```python
    except Exception as exc:  # noqa: BLE001 - explainability is best-effort, never fatal
        return {
            "method": "unavailable",
            "feature_impact": [],
            "narrative": None,
            "note": f"SHAP explanation unavailable for this model: {exc}",
            "summary_plot": None,
            "bar_plot": None,
            "dependence_plots": [],
        }
```

- [ ] **Step 9: Change `explain_prediction`'s return shape and add the waterfall plot**

Change the whole function from:

```python
def explain_prediction(model_path: str, values: dict[str, Any], transformed_dataset_path: str) -> Optional[list[dict[str, Any]]]:
    """Best-effort per-row SHAP contribution for a single user-submitted
    prediction row, computed on demand (POST /predict). Returns None rather
    than raising when SHAP can't explain this estimator/input."""
    try:
        cfg = _explainability_config()
        bundle = joblib.load(model_path)
        fit_estimator = bundle["estimator"]
        prep = fit_estimator.named_steps["prep"]
        model = fit_estimator.named_steps["model"]
        feature_columns = bundle["feature_columns"]
        feature_names = [str(n) for n in prep.get_feature_names_out()]

        sample = _shap_background(transformed_dataset_path, feature_columns, cfg["max_background_rows"])
        background = np.asarray(prep.transform(sample))

        row = pd.DataFrame([{col: values.get(col, np.nan) for col in feature_columns}])
        row_transformed = np.asarray(prep.transform(row))

        explainer = _build_shap_explainer(model, background, feature_names)
        shap_values = explainer(row_transformed)
        row_values = _reduce_shap_values(np.asarray(shap_values.values))[0]

        ranked = sorted(zip(feature_names, row_values), key=lambda pair: abs(pair[1]), reverse=True)
        top_n = cfg["top_n_features"]
        return [{"feature": name, "shap_value": round(float(value), 6)} for name, value in ranked[:top_n]]
    except Exception:  # noqa: BLE001 - per-row explanation is best-effort
        return None
```

to:

```python
def explain_prediction(
    model_path: str, values: dict[str, Any], transformed_dataset_path: str
) -> Optional[dict[str, Any]]:
    """Best-effort per-row SHAP contribution (+ waterfall plot) for a single
    user-submitted prediction row, computed on demand (POST /predict).
    Returns None rather than raising when SHAP can't explain this
    estimator/input; on success returns {"contributions": [...],
    "waterfall_plot_base64": Optional[str]}."""
    try:
        cfg = _explainability_config()
        bundle = joblib.load(model_path)
        fit_estimator = bundle["estimator"]
        prep = fit_estimator.named_steps["prep"]
        model = fit_estimator.named_steps["model"]
        feature_columns = bundle["feature_columns"]
        feature_names = [str(n) for n in prep.get_feature_names_out()]

        sample = _shap_background(transformed_dataset_path, feature_columns, cfg["max_background_rows"])
        background = np.asarray(prep.transform(sample))

        row = pd.DataFrame([{col: values.get(col, np.nan) for col in feature_columns}])
        row_transformed = np.asarray(prep.transform(row))

        explainer = _build_shap_explainer(model, background, feature_names)
        shap_values = explainer(row_transformed)
        row_values = _reduce_shap_values(np.asarray(shap_values.values))[0]
        row_base_value = _reduce_shap_base_values(np.asarray(shap_values.base_values))[0]
        row_data = row_transformed[0]

        ranked = sorted(zip(feature_names, row_values), key=lambda pair: abs(pair[1]), reverse=True)
        top_n = cfg["top_n_features"]
        contributions = [{"feature": name, "shap_value": round(float(value), 6)} for name, value in ranked[:top_n]]
        waterfall_plot_base64 = _render_waterfall_plot(row_values, float(row_base_value), row_data, feature_names)

        return {"contributions": contributions, "waterfall_plot_base64": waterfall_plot_base64}
    except Exception:  # noqa: BLE001 - per-row explanation is best-effort
        return None
```

- [ ] **Step 10: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_explainability.py -v`
Expected: PASS (15 tests).

- [ ] **Step 11: Commit**

```bash
git add src/state.py config/runtime.yaml requirements.txt src/training/dispatch.py tests/test_explainability.py
git commit -m "feat: add SHAP plot rendering (beeswarm, bar, dependence, waterfall)"
```

---

### Task 2: LLM plot captions (`explainability_node.py`)

**Files:**
- Create: `src/agents/prompts/explainability_plot_captions.md`
- Modify: `src/agents/explainability_node.py`
- Modify: `src/llm/client.py` (mock response for the `explainability_captions` node)
- Modify: `config/models.yaml` (node generation params for `explainability_captions`)
- Modify: `tests/test_pipeline_smoke.py` (`_fake_generate` must handle the new node)
- Test: `tests/test_explainability_node.py`

**Interfaces:**
- Consumes: `compute_explainability(...)` from Task 1, whose result now carries `summary_plot`/`bar_plot`/`dependence_plots`.
- Produces: `explainability_node(state) -> PipelineState` now also fills in `.caption` on each present plot (or leaves it `None` on LLM failure, appending to `state["errors"]`).

- [ ] **Step 1: Write the plot-captions prompt template**

Create `src/agents/prompts/explainability_plot_captions.md`:

```
You are the Explainability agent in an agentic AutoML pipeline, writing
captions for SHAP plots that will be shown to a non-technical business user
alongside the images. You are given the top features driving the winning
model's predictions, ranked by mean absolute SHAP value, and the names of
the features selected for dependence plots (a subset of the ranked list).
For each plot, write 1-3 sentences explaining what that plot type shows and,
where useful, what it reveals about these specific features. Do not claim
causation or a guarantee — SHAP values explain the model's behavior, not a
proven real-world cause-and-effect relationship.

## Ranked feature impact (highest first)
{{FEATURE_IMPACT_JSON}}

## Dependence plot features (in order)
{{DEPENDENCE_FEATURES_JSON}}
```

- [ ] **Step 2: Write the failing tests**

Append these two tests to the end of `tests/test_explainability_node.py`:

```python
def test_explainability_node_captions_computed_plots(monkeypatch, tmp_path):
    monkeypatch.setattr(
        explainability_node_module,
        "compute_explainability",
        lambda model_path, transformed_dataset_path: {
            "method": "tree",
            "feature_impact": [{"feature": "age", "mean_abs_shap": 0.4}],
            "narrative": None,
            "note": None,
            "summary_plot": {"title": "Impact distribution (beeswarm)", "feature": None, "image_base64": "aaaa", "caption": None},
            "bar_plot": {"title": "Feature impact (bar)", "feature": None, "image_base64": "bbbb", "caption": None},
            "dependence_plots": [{"title": "Dependence: age", "feature": "age", "image_base64": "cccc", "caption": None}],
        },
    )

    def _fake_generate(self, run_id, node, system_prompt, user_prompt, json_schema=None, retries=1):
        if node == "explainability":
            return "Age drives most predictions."
        if node == "explainability_captions":
            return {
                "summary_plot_caption": "Each dot is a row; color shows feature value.",
                "bar_plot_caption": "Bars rank features by average impact.",
                "dependence_plot_captions": {"age": "Older customers push predictions higher."},
            }
        raise AssertionError(f"unexpected node {node}")

    monkeypatch.setattr(LLMClient, "generate", _fake_generate)

    state = new_state(run_id="expl-node-5", dataset_path="unused.csv", use_case_description="test")
    state["best_model"] = {"model_path": str(tmp_path / "model.joblib")}

    result = explainability_node(state)
    explainability = result["best_model"]["explainability"]
    assert explainability["summary_plot"]["caption"] == "Each dot is a row; color shows feature value."
    assert explainability["bar_plot"]["caption"] == "Bars rank features by average impact."
    assert explainability["dependence_plots"][0]["caption"] == "Older customers push predictions higher."


def test_explainability_node_tolerates_caption_llm_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        explainability_node_module,
        "compute_explainability",
        lambda model_path, transformed_dataset_path: {
            "method": "tree",
            "feature_impact": [{"feature": "age", "mean_abs_shap": 0.4}],
            "narrative": None,
            "note": None,
            "summary_plot": {"title": "Impact distribution (beeswarm)", "feature": None, "image_base64": "aaaa", "caption": None},
            "bar_plot": None,
            "dependence_plots": [],
        },
    )

    def _fake_generate(self, run_id, node, system_prompt, user_prompt, json_schema=None, retries=1):
        if node == "explainability":
            return "Age drives most predictions."
        if node == "explainability_captions":
            raise RuntimeError("captions LLM down")
        raise AssertionError(f"unexpected node {node}")

    monkeypatch.setattr(LLMClient, "generate", _fake_generate)

    state = new_state(run_id="expl-node-6", dataset_path="unused.csv", use_case_description="test")
    state["best_model"] = {"model_path": str(tmp_path / "model.joblib")}

    result = explainability_node(state)
    explainability = result["best_model"]["explainability"]
    assert explainability["summary_plot"]["image_base64"] == "aaaa"
    assert explainability["summary_plot"]["caption"] is None
    assert any("plot captions unavailable" in e for e in result["errors"])
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_explainability_node.py -v`
Expected: FAIL — `test_explainability_node_captions_computed_plots` fails because captions stay `None` (no captions call exists yet); `test_explainability_node_tolerates_caption_llm_failure` fails because no error message mentioning "plot captions unavailable" is appended yet.

- [ ] **Step 4: Implement the captions call in `src/agents/explainability_node.py`**

Replace the whole file with:

```python
"""LLM-backed node: computes aggregate SHAP feature impact (+ plots) for the
winning model, narrates its top drivers in plain language, and captions each
generated plot. Runs once, only for best_model (not every candidate) — SHAP +
LLM calls for every discarded candidate would multiply cost for data that's
thrown away."""

from __future__ import annotations

from src.agents.prompt_utils import render_prompt
from src.llm.client import get_llm_client
from src.state import PipelineState
from src.training.dispatch import compute_explainability

_CAPTIONS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary_plot_caption": {"type": "string"},
        "bar_plot_caption": {"type": "string"},
        "dependence_plot_captions": {"type": "object"},
    },
}


def explainability_node(state: PipelineState) -> PipelineState:
    best_model = state.get("best_model") or {}
    model_path = best_model.get("model_path")
    if not model_path:
        return state

    result = compute_explainability(model_path, state.get("transformed_dataset_path", ""))

    if result["method"] != "unavailable":
        client = get_llm_client()
        system_prompt = render_prompt("explainability.md", FEATURE_IMPACT_JSON=result["feature_impact"])
        try:
            result["narrative"] = client.generate(
                run_id=state["run_id"],
                node="explainability",
                system_prompt=system_prompt,
                user_prompt="Write the explanation now.",
                json_schema=None,
            )
        except Exception as exc:  # noqa: BLE001 - narrative is enrichment, never fatal
            state.setdefault("errors", []).append(f"explainability: narrative unavailable: {exc}")

        dependence_plots = result.get("dependence_plots") or []
        has_plots = result.get("summary_plot") or result.get("bar_plot") or dependence_plots
        if has_plots:
            captions_prompt = render_prompt(
                "explainability_plot_captions.md",
                FEATURE_IMPACT_JSON=result["feature_impact"],
                DEPENDENCE_FEATURES_JSON=[p["feature"] for p in dependence_plots],
            )
            try:
                captions = client.generate(
                    run_id=state["run_id"],
                    node="explainability_captions",
                    system_prompt=captions_prompt,
                    user_prompt="Write the plot captions now.",
                    json_schema=_CAPTIONS_JSON_SCHEMA,
                )
                if result.get("summary_plot"):
                    result["summary_plot"]["caption"] = captions.get("summary_plot_caption")
                if result.get("bar_plot"):
                    result["bar_plot"]["caption"] = captions.get("bar_plot_caption")
                dependence_captions = captions.get("dependence_plot_captions") or {}
                for plot in dependence_plots:
                    plot["caption"] = dependence_captions.get(plot["feature"])
            except Exception as exc:  # noqa: BLE001 - captions are enrichment, never fatal
                state.setdefault("errors", []).append(f"explainability: plot captions unavailable: {exc}")

    best_model["explainability"] = result
    state["best_model"] = best_model
    return state
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_explainability_node.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Add the mock-mode response for `AUTOML_MOCK_LLM=1`**

In `src/llm/client.py`, `_mock_response` has an `explainability` branch (around line 201) followed by a `chat` branch. Add a new branch between them:

```python
    if node == "explainability_captions":
        return {
            "summary_plot_caption": (
                "MOCK-MODE CAPTION (AUTOML_MOCK_LLM=1) — each dot represents one row; "
                "its position shows the feature's impact and its color the feature's value."
            ),
            "bar_plot_caption": (
                "MOCK-MODE CAPTION (AUTOML_MOCK_LLM=1) — bars rank features by their "
                "average impact on the model's predictions."
            ),
            "dependence_plot_captions": {},
        }
```

- [ ] **Step 7: Add the node's generation params to `config/models.yaml`**

In the `nodes:` block, change:

```yaml
  explainability:
    temperature: 0.2
    max_tokens: 512

  chat:
```

to:

```yaml
  explainability:
    temperature: 0.2
    max_tokens: 512

  explainability_captions:
    temperature: 0.2
    max_tokens: 768

  chat:
```

- [ ] **Step 8: Update `tests/test_pipeline_smoke.py`'s `_fake_generate` for the new node**

Change:

```python
    if node == "explainability":
        return "This model is driven mainly by tenure and monthly spend."
    raise ValueError(f"unexpected node in fake_generate: {node}")
```

to:

```python
    if node == "explainability":
        return "This model is driven mainly by tenure and monthly spend."
    if node == "explainability_captions":
        return {
            "summary_plot_caption": "Each dot is a customer; color shows whether that feature's value was high or low.",
            "bar_plot_caption": "Bars rank features by their average impact on the prediction.",
            "dependence_plot_captions": {},
        }
    raise ValueError(f"unexpected node in fake_generate: {node}")
```

- [ ] **Step 9: Run the full-suite smoke test to verify the graph wiring still works end to end**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline_smoke.py -v`
Expected: PASS. This exercises `explainability_node` for real (real SHAP + plot computation against the trained RandomForestClassifier, real mocked LLM narrative + captions) inside the full graph.

- [ ] **Step 10: Run the full fixture suite (required for any change touching `src/training/dispatch.py` per CLAUDE.md)**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: PASS (no regressions).

- [ ] **Step 11: Commit**

```bash
git add src/agents/prompts/explainability_plot_captions.md src/agents/explainability_node.py src/llm/client.py config/models.yaml tests/test_pipeline_smoke.py tests/test_explainability_node.py
git commit -m "feat: caption SHAP plots via a batched LLM call in explainability_node"
```

---

### Task 3: API updates (`src/api/server.py`)

**Files:**
- Modify: `src/api/server.py` (`get_explainability`'s fallback shape; `predict`'s response unpacking)
- Test: `tests/test_api_explainability.py`

**Interfaces:**
- Consumes: `compute_explainability`/`explain_prediction` from Task 1 (plot fields on the precomputed aggregate value; the new `explain_prediction` dict shape for the per-row case).
- Produces: `GET /api/runs/{run_id}/explainability` response gains `summary_plot`/`bar_plot`/`dependence_plots`; `POST /api/runs/{run_id}/predict` response gains `waterfall_plot_base64` alongside the existing `contributions`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_api_explainability.py` in full with:

```python
"""GET /api/runs/{id}/explainability (reads the precomputed SHAP summary,
now including plots) and the /predict endpoint's `contributions` +
`waterfall_plot_base64` fields (computed on demand)."""

from __future__ import annotations

import base64
import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api import server
from src.training.dispatch import _registry, _run_job

_PNG_HEADER = b"\x89PNG"


def _make_run_with_model(tmp_path, monkeypatch, run_id="run-1", explainability=None):
    rng = np.random.default_rng(2)
    n = 150
    df = pd.DataFrame({"x1": rng.random(n), "x2": rng.random(n), "target": rng.integers(0, 2, n)})
    dataset_path = tmp_path / "data.csv"
    df.to_csv(dataset_path, index=False)

    tag = f"api-{run_id}"
    _registry[tag] = {"status": "pending", "feature_selection": {"enabled": False}}
    _run_job(
        tag, str(dataset_path), "target", "classification", "sklearn",
        "RandomForestClassifier", {"n_estimators": 10, "max_depth": 3, "random_state": 0},
        None, [], False, None, False, "none", False, None, None,
    )
    trained = _registry[tag]
    assert trained["status"] == "succeeded", trained.get("error")

    best_model = dict(trained)
    if explainability is not None:
        best_model["explainability"] = explainability

    now = time.time()
    monkeypatch.setitem(
        server._runs,
        run_id,
        {
            "state": {
                "dataset_path": str(dataset_path),
                "transformed_dataset_path": str(dataset_path),
                "best_model": best_model,
            },
            "status": "completed",
            "events": [],
            "filename": "data.csv",
            "created_at": now,
            "finished_at": now,
            "cancel_requested": False,
            "chat_history": [],
        },
    )
    return str(dataset_path), best_model


def test_get_explainability_returns_precomputed_summary(tmp_path, monkeypatch):
    canned = {
        "method": "tree",
        "feature_impact": [{"feature": "x1", "mean_abs_shap": 0.3}],
        "narrative": "x1 drives this model.",
        "note": None,
        "summary_plot": {"title": "Impact distribution (beeswarm)", "feature": None, "image_base64": "aaaa", "caption": "c1"},
        "bar_plot": {"title": "Feature impact (bar)", "feature": None, "image_base64": "bbbb", "caption": "c2"},
        "dependence_plots": [{"title": "Dependence: x1", "feature": "x1", "image_base64": "cccc", "caption": "c3"}],
    }
    _make_run_with_model(tmp_path, monkeypatch, explainability=canned)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/explainability").json()
    assert result == canned


def test_get_explainability_default_when_not_yet_computed(tmp_path, monkeypatch):
    _make_run_with_model(tmp_path, monkeypatch, explainability=None)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/explainability").json()
    assert result["method"] == "unavailable"
    assert result["feature_impact"] == []
    assert result["summary_plot"] is None
    assert result["bar_plot"] is None
    assert result["dependence_plots"] == []


def test_get_explainability_404_without_trained_model(monkeypatch):
    monkeypatch.setitem(
        server._runs,
        "run-2",
        {
            "state": {"best_model": {}}, "status": "running", "events": [], "filename": "d.csv",
            "created_at": time.time(), "finished_at": None, "cancel_requested": False, "chat_history": [],
        },
    )
    client = TestClient(server.app)
    res = client.get("/api/runs/run-2/explainability")
    assert res.status_code == 404


def test_predict_endpoint_includes_contributions(tmp_path, monkeypatch):
    _make_run_with_model(tmp_path, monkeypatch)
    client = TestClient(server.app)
    result = client.post("/api/runs/run-1/predict", json={"values": {"x1": 0.5, "x2": 0.5}}).json()
    assert "prediction" in result
    assert isinstance(result["contributions"], list) and result["contributions"]


def test_predict_endpoint_includes_waterfall_plot(tmp_path, monkeypatch):
    _make_run_with_model(tmp_path, monkeypatch)
    client = TestClient(server.app)
    result = client.post("/api/runs/run-1/predict", json={"values": {"x1": 0.5, "x2": 0.5}}).json()
    assert result["waterfall_plot_base64"] is not None
    assert base64.b64decode(result["waterfall_plot_base64"])[:4] == _PNG_HEADER
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_explainability.py -v`
Expected: FAIL — the fallback-shape test fails (missing keys), and both predict tests fail (`contributions`/`waterfall_plot_base64` come back `None` because `predict`'s current code assigns the whole `explain_prediction(...)` dict to `result["contributions"]` instead of unpacking it).

- [ ] **Step 3: Update the fallback shape in `get_explainability`**

In `src/api/server.py`, change:

```python
    explainability = best_model.get("explainability") or {
        "method": "unavailable",
        "feature_impact": [],
        "narrative": None,
        "note": "explainability has not been computed for this run yet",
    }
```

to:

```python
    explainability = best_model.get("explainability") or {
        "method": "unavailable",
        "feature_impact": [],
        "narrative": None,
        "note": "explainability has not been computed for this run yet",
        "summary_plot": None,
        "bar_plot": None,
        "dependence_plots": [],
    }
```

- [ ] **Step 4: Unpack the new `explain_prediction` shape in `predict`**

Change:

```python
    transformed_dataset_path = entry["state"].get("transformed_dataset_path", "")
    result["contributions"] = explain_prediction(model_path, body.values, transformed_dataset_path)
    return _json_safe(result)
```

to:

```python
    transformed_dataset_path = entry["state"].get("transformed_dataset_path", "")
    explanation = explain_prediction(model_path, body.values, transformed_dataset_path)
    result["contributions"] = explanation["contributions"] if explanation else None
    result["waterfall_plot_base64"] = explanation["waterfall_plot_base64"] if explanation else None
    return _json_safe(result)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_explainability.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Run the full test suite to check for regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/api/server.py tests/test_api_explainability.py
git commit -m "feat: expose SHAP plots via the explainability/predict endpoints"
```

---

### Task 4: Frontend (`frontend/index.html`, `frontend/app.js`, `frontend/styles.css`)

**Files:**
- Modify: `frontend/index.html` (replace the Explainability tab's bar-list markup with a single plots container)
- Modify: `frontend/app.js` (`renderExplainability`, `loadExplainabilityTab`, the run-switch reset block, `renderPredictResult`)
- Modify: `frontend/styles.css` (add `.shap-plot`/`.shap-plot-caption`/`.shap-dependence-grid` rules)

**Interfaces:**
- Consumes: `GET /api/runs/{run_id}/explainability`'s `summary_plot`/`bar_plot`/`dependence_plots` and `POST /api/runs/{run_id}/predict`'s `waterfall_plot_base64`, both from Task 3.
- No new interfaces produced — this is the leaf UI layer.

- [ ] **Step 1: Replace the Explainability tab's bar-list markup with a single plots container**

In `frontend/index.html`, replace (currently lines 497-504):

```html
      <div id="tab-explainability-panel" class="hidden">
        <div class="card" id="explainability-card">
          <div class="card-head"><h3>What drives this model (SHAP)</h3></div>
          <p class="muted small" id="explainability-narrative"></p>
          <div class="fi-list" id="explainability-list"></div>
          <p class="muted small hidden" id="explainability-empty"></p>
        </div>
      </div><!-- /tab-explainability-panel -->
```

with:

```html
      <div id="tab-explainability-panel" class="hidden">
        <div class="card" id="explainability-card">
          <div class="card-head"><h3>What drives this model (SHAP)</h3></div>
          <p class="muted small" id="explainability-narrative"></p>
          <div id="explainability-plots"></div>
          <p class="muted small hidden" id="explainability-empty"></p>
        </div>
      </div><!-- /tab-explainability-panel -->
```

- [ ] **Step 2: Add CSS for the new plot images/captions**

In `frontend/styles.css`, add right after the feature-importance block (currently ending at line 517, `.fi-value { ... }`):

```css
/* SHAP plots (Explainability tab, Predict tab waterfall) */
.shap-plot { margin-bottom: var(--sp-4); }
.shap-plot img { max-width: 100%; height: auto; display: block; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); }
.shap-plot-caption { font-size: var(--text-sm); color: var(--text-secondary); margin-top: var(--sp-2); }
.shap-dependence-grid { display: grid; gap: var(--sp-4); grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
```

- [ ] **Step 3: Rewrite `renderExplainability` and `loadExplainabilityTab` in `frontend/app.js`**

Change:

```javascript
async function loadExplainabilityTab(run) {
  if (!(run.best_model || {}).model_path) {
    $("explainability-narrative").textContent = "";
    $("explainability-list").innerHTML = "";
    $("explainability-empty").textContent = "No trained model is available for this run yet.";
    $("explainability-empty").classList.remove("hidden");
    return;
  }
  if (explainabilityLoadedFor === run.run_id) return;
  explainabilityLoadedFor = run.run_id;

  $("explainability-narrative").textContent = "";
  $("explainability-list").innerHTML = "";
  $("explainability-empty").textContent = "Loading explainability data…";
  $("explainability-empty").classList.remove("hidden");

  try {
    const data = await (await authFetch(`/api/runs/${run.run_id}/explainability`)).json();
    renderExplainability(data);
  } catch {
    $("explainability-empty").textContent = "Could not load explainability data for this run.";
    $("explainability-empty").classList.remove("hidden");
    explainabilityLoadedFor = null;
  }
}

function renderExplainability(data) {
  const empty = $("explainability-empty");
  const list = $("explainability-list");
  const narrative = $("explainability-narrative");

  if (data.method === "unavailable" || !data.feature_impact || !data.feature_impact.length) {
    list.innerHTML = "";
    narrative.textContent = "";
    empty.textContent = data.note || "SHAP-based impact analysis isn't available for this run.";
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  narrative.textContent = data.narrative || "";

  const max = Math.max(...data.feature_impact.map((f) => f.mean_abs_shap), 0.0001);
  list.innerHTML = data.feature_impact
    .map(
      (f) => `
      <div class="fi-row">
        <span class="fi-name" title="${escapeHtml(f.feature)}">${escapeHtml(f.feature)}</span>
        <span class="fi-track"><span class="fi-fill" style="width:${((f.mean_abs_shap / max) * 100).toFixed(1)}%"></span></span>
        <span class="fi-value">${f.mean_abs_shap.toFixed(3)}</span>
      </div>`
    )
    .join("");
}
```

to:

```javascript
async function loadExplainabilityTab(run) {
  if (!(run.best_model || {}).model_path) {
    $("explainability-narrative").textContent = "";
    $("explainability-plots").innerHTML = "";
    $("explainability-empty").textContent = "No trained model is available for this run yet.";
    $("explainability-empty").classList.remove("hidden");
    return;
  }
  if (explainabilityLoadedFor === run.run_id) return;
  explainabilityLoadedFor = run.run_id;

  $("explainability-narrative").textContent = "";
  $("explainability-plots").innerHTML = "";
  $("explainability-empty").textContent = "Loading explainability data…";
  $("explainability-empty").classList.remove("hidden");

  try {
    const data = await (await authFetch(`/api/runs/${run.run_id}/explainability`)).json();
    renderExplainability(data);
  } catch {
    $("explainability-empty").textContent = "Could not load explainability data for this run.";
    $("explainability-empty").classList.remove("hidden");
    explainabilityLoadedFor = null;
  }
}

function renderShapPlot(plot) {
  return `
    <div class="shap-plot">
      <img src="data:image/png;base64,${plot.image_base64}" alt="${escapeHtml(plot.title)}" />
      ${plot.caption ? `<p class="shap-plot-caption">${escapeHtml(plot.caption)}</p>` : ""}
    </div>`;
}

function renderExplainability(data) {
  const empty = $("explainability-empty");
  const plotsEl = $("explainability-plots");
  const narrative = $("explainability-narrative");

  const hasPlots = data.summary_plot || data.bar_plot || (data.dependence_plots && data.dependence_plots.length);
  if (data.method === "unavailable" || !hasPlots) {
    plotsEl.innerHTML = "";
    narrative.textContent = "";
    empty.textContent = data.note || "SHAP-based impact analysis isn't available for this run.";
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  narrative.textContent = data.narrative || "";

  let html = "";
  if (data.bar_plot) html += renderShapPlot(data.bar_plot);
  if (data.summary_plot) html += renderShapPlot(data.summary_plot);
  if (data.dependence_plots && data.dependence_plots.length) {
    html += `<div class="shap-dependence-grid">${data.dependence_plots.map(renderShapPlot).join("")}</div>`;
  }
  plotsEl.innerHTML = html;
}
```

- [ ] **Step 4: Update the run-switch reset block**

Search `frontend/app.js` for the block that resets explainability state on run switch (currently around lines 1007-1010):

```javascript
  explainabilityLoadedFor = null;
  $("explainability-narrative").textContent = "";
  $("explainability-list").innerHTML = "";
  $("explainability-empty").classList.add("hidden");
```

Change it to:

```javascript
  explainabilityLoadedFor = null;
  $("explainability-narrative").textContent = "";
  $("explainability-plots").innerHTML = "";
  $("explainability-empty").classList.add("hidden");
```

- [ ] **Step 5: Replace the Predict tab's "Why" bar-list with the waterfall plot**

In `frontend/app.js`, add this constant right above `function renderPredictResult(data) {`:

```javascript
const WATERFALL_CAPTION =
  "Each bar shows how much that feature pushed this specific prediction up or down from the model's average output. The final value (f(x)) is this row's predicted output.";
```

Then change `renderPredictResult` from:

```javascript
function renderPredictResult(data) {
  const resultBox = $("predict-result");
  let html = `<div class="predict-headline">Prediction: ${escapeHtml(String(data.prediction))}</div>`;
  if (data.probabilities) {
    const max = Math.max(...Object.values(data.probabilities), 0.0001);
    html += Object.entries(data.probabilities)
      .sort((a, b) => b[1] - a[1])
      .map(
        ([label, p]) => `
        <div class="predict-proba-row">
          <span>${escapeHtml(label)}</span>
          <span class="fi-track"><span class="fi-fill" style="width:${((p / max) * 100).toFixed(1)}%"></span></span>
          <span class="mono">${(p * 100).toFixed(1)}%</span>
        </div>`
      )
      .join("");
  }
  if (data.contributions && data.contributions.length) {
    const maxAbs = Math.max(...data.contributions.map((c) => Math.abs(c.shap_value)), 0.0001);
    html += `<p class="muted small" style="margin-top:10px">Why:</p>`;
    html += data.contributions
      .map(
        (c) => `
        <div class="predict-proba-row">
          <span>${escapeHtml(c.feature)}</span>
          <span class="fi-track"><span class="fi-fill" style="width:${((Math.abs(c.shap_value) / maxAbs) * 100).toFixed(1)}%"></span></span>
          <span class="mono">${c.shap_value >= 0 ? "+" : ""}${c.shap_value.toFixed(3)}</span>
        </div>`
      )
      .join("");
  }
  resultBox.innerHTML = html;
}
```

to:

```javascript
function renderPredictResult(data) {
  const resultBox = $("predict-result");
  let html = `<div class="predict-headline">Prediction: ${escapeHtml(String(data.prediction))}</div>`;
  if (data.probabilities) {
    const max = Math.max(...Object.values(data.probabilities), 0.0001);
    html += Object.entries(data.probabilities)
      .sort((a, b) => b[1] - a[1])
      .map(
        ([label, p]) => `
        <div class="predict-proba-row">
          <span>${escapeHtml(label)}</span>
          <span class="fi-track"><span class="fi-fill" style="width:${((p / max) * 100).toFixed(1)}%"></span></span>
          <span class="mono">${(p * 100).toFixed(1)}%</span>
        </div>`
      )
      .join("");
  }
  if (data.waterfall_plot_base64) {
    html += `
      <div class="shap-plot" style="margin-top:10px">
        <p class="muted small">Why:</p>
        <img src="data:image/png;base64,${data.waterfall_plot_base64}" alt="Waterfall plot of this prediction's SHAP contributions" />
        <p class="shap-plot-caption">${escapeHtml(WATERFALL_CAPTION)}</p>
      </div>`;
  }
  resultBox.innerHTML = html;
}
```

- [ ] **Step 6: Manually verify in the browser**

Run: `.venv/Scripts/python.exe run_server.py` (set `AUTOML_MOCK_LLM=1` first if no LLM API key is configured, so the pipeline completes without real API calls)

Then, in a browser:
1. Upload a small CSV, describe a use case, confirm the task spec and feature plan, and let training finish.
2. Click the "Explainability" tab — verify it shows the narrative paragraph, a bar-plot image, a beeswarm image, and up to 3 dependence-plot images, each with a caption underneath (or a clear "not available" message if the winning model's estimator isn't SHAP-supported).
3. Open the Predict tab, submit a row, and verify a waterfall plot image with a caption appears below the prediction/probabilities (replacing the old text "Why" bar-list).
4. Confirm no broken `<img>` (empty `src`) ever appears — the "not available" text path should show instead when plots are missing.

- [ ] **Step 7: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: render SHAP plots on the Explainability and Predict tabs"
```
