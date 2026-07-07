"""Aggregate (per-run) and per-prediction SHAP-based explainability
(docs/superpowers/specs/2026-07-07-model-explainability-design.md).
compute_explainability/explain_prediction degrade to method="unavailable" /
None rather than raising; they never block the pipeline or predict endpoint."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.dispatch import (
    _build_shap_explainer,
    _reduce_shap_values,
    _registry,
    _run_job,
    _shap_method_label,
    compute_explainability,
    explain_prediction,
)


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


def test_explain_prediction_returns_ranked_row_contributions(trained_tree_model):
    model_path, dataset_path = trained_tree_model
    contributions = explain_prediction(model_path, {"x1": 0.9, "x2": 0.1, "x3": 0.5}, dataset_path)
    assert contributions is not None
    assert 1 <= len(contributions) <= 8
    assert {c["feature"] for c in contributions} <= {"x1", "x2", "x3"}


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


def test_shap_method_label_falls_back_to_kernel_for_unsupported_estimator():
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(1)
    X = rng.random((60, 3))
    y = rng.integers(0, 2, 60)
    model = KNeighborsClassifier().fit(X, y)
    explainer = _build_shap_explainer(model, X[:20], ["a", "b", "c"])
    assert _shap_method_label(explainer) == "kernel"
