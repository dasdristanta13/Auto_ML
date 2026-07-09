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
    _render_beeswarm_plot,
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
    assert result["fidelity_r2"] is None
    assert result["background_sample_size"] == 0


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


def test_compute_explainability_includes_fidelity_and_sample_size(trained_tree_model):
    model_path, dataset_path = trained_tree_model
    result = compute_explainability(model_path, dataset_path)
    assert isinstance(result["fidelity_r2"], float)
    assert result["background_sample_size"] == 100  # min(max_background_rows=100, n=150)


def test_compute_explainability_fidelity_none_for_multiclass(tmp_path):
    rng = np.random.default_rng(6)
    n = 150
    df = pd.DataFrame({"x1": rng.random(n), "x2": rng.random(n), "target": rng.integers(0, 3, n)})
    dataset_path = tmp_path / "multiclass.csv"
    df.to_csv(dataset_path, index=False)
    result = _train("expl-multiclass", dataset_path, "RandomForestClassifier", {"n_estimators": 10, "max_depth": 3, "random_state": 0})

    explainability = compute_explainability(result["model_path"], str(dataset_path))
    assert explainability["fidelity_r2"] is None


def test_compute_explainability_feature_impact_includes_signed_shap(trained_tree_model):
    model_path, dataset_path = trained_tree_model
    result = compute_explainability(model_path, dataset_path)
    assert all("mean_signed_shap" in f for f in result["feature_impact"])


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


def test_compute_explainability_survives_plot_explanation_failure(monkeypatch, trained_tree_model):
    import src.training.dispatch as dispatch_module

    def _boom(values, background, feature_names):
        raise RuntimeError("boom")

    monkeypatch.setattr(dispatch_module, "_shap_plot_explanation", _boom)
    model_path, dataset_path = trained_tree_model
    result = compute_explainability(model_path, dataset_path)

    assert result["method"] == "tree"
    assert result["feature_impact"]
    assert result["summary_plot"] is None
    assert result["bar_plot"] is None
    assert result["dependence_plots"] == []


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


class _FakeBinaryModel:
    def predict_proba(self, X):
        return np.column_stack([1 - X[:, 0], X[:, 0]])


class _FakeMulticlassModel:
    def predict_proba(self, X):
        return np.column_stack([X[:, 0], X[:, 0], 1 - 2 * X[:, 0]])


class _FakeRegressor:
    def predict(self, X):
        return X[:, 0] + X[:, 1]


def test_shap_fidelity_r2_perfect_reconstruction_for_binary():
    from src.training.dispatch import _shap_fidelity_r2

    background = np.array([[0.2, 0.0], [0.7, 0.0], [0.5, 0.0]])
    model = _FakeBinaryModel()
    proba = model.predict_proba(background)[:, 1]
    base_values = np.zeros(3)
    values = np.column_stack([proba, np.zeros(3)])  # reconstruction == proba exactly
    result = _shap_fidelity_r2(model, background, values, base_values)
    assert result == pytest.approx(1.0)


def test_shap_fidelity_r2_none_for_multiclass():
    from src.training.dispatch import _shap_fidelity_r2

    background = np.array([[0.2, 0.0], [0.7, 0.0]])
    model = _FakeMulticlassModel()
    result = _shap_fidelity_r2(model, background, np.zeros((2, 2)), np.zeros(2))
    assert result is None


def test_shap_fidelity_r2_uses_predict_for_regression():
    from src.training.dispatch import _shap_fidelity_r2

    background = np.array([[1.0, 2.0], [3.0, 4.0]])
    model = _FakeRegressor()
    base_values = np.zeros(2)
    values = np.column_stack([background[:, 0], background[:, 1]])  # sum == predict() exactly
    result = _shap_fidelity_r2(model, background, values, base_values)
    assert result == pytest.approx(1.0)


def test_shap_fidelity_r2_returns_none_on_failure():
    from src.training.dispatch import _shap_fidelity_r2

    class _Boom:
        def predict(self, X):
            raise RuntimeError("boom")

    result = _shap_fidelity_r2(_Boom(), np.array([[1.0]]), np.array([[1.0]]), np.zeros(1))
    assert result is None


def test_render_dependence_plots_skips_unknown_feature_without_raising():
    explanation = _shap_plot_explanation(
        np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[0.1, 0.2], [0.3, 0.4]]), ["a", "b"]
    )
    plots = _render_dependence_plots(explanation, ["a", "does-not-exist", "b"], 3)
    assert {p["feature"] for p in plots} == {"a", "b"}


def test_render_beeswarm_plot_closes_figure_on_failure(monkeypatch):
    import matplotlib.pyplot as plt
    import shap

    def _boom(explanation, show=False):
        plt.figure()
        raise RuntimeError("boom")

    monkeypatch.setattr(shap.plots, "beeswarm", _boom)
    before = set(plt.get_fignums())

    explanation = _shap_plot_explanation(
        np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[0.1, 0.2], [0.3, 0.4]]), ["a", "b"]
    )
    result = _render_beeswarm_plot(explanation)
    assert result is None
    assert set(plt.get_fignums()) == before
