"""Shared feature elimination (docs/superpowers/specs/2026-07-06-eda-drops-
and-rfe-design.md, revision): select_features runs RFECV ONCE with a basic
linear model, and every candidate trains on that shared subset. Narrow
datasets skip with a note; disabled runs report enabled=False."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.dispatch import _registry, _run_job, select_features


@pytest.fixture()
def informative_plus_noise(tmp_path) -> str:
    """3 informative + 5 pure-noise features; RFECV should keep a subset."""
    rng = np.random.default_rng(7)
    n = 120
    x1, x2, x3 = rng.random(n), rng.random(n), rng.random(n)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            **{f"noise{i}": rng.random(n) for i in range(5)},
            "target": (x1 + x2 + x3 + rng.normal(0, 0.05, n) > 1.5).astype(int),
        }
    )
    path = tmp_path / "fs.csv"
    df.to_csv(path, index=False)
    return str(path)


def _run(tag: str, dataset: str, selected: list | None, note: str | None = None,
         estimator: str = "RandomForestClassifier") -> dict:
    _registry[tag] = {"status": "pending", "feature_selection": {"enabled": False}}
    _run_job(
        tag, dataset, "target", "classification", "sklearn", estimator,
        {"n_estimators": 20} if estimator == "RandomForestClassifier" else {"max_iter": 500},
        None, [], False, None, False, "none", False, None, None,
        selected_features=selected, feature_selection_note=note,
    )
    return _registry[tag]


def test_select_features_uses_basic_model_and_selects_subset(informative_plus_noise):
    selection = select_features(informative_plus_noise, "target", "classification", None, [], "f1")
    assert selection["enabled"] is True
    assert selection["basic_model"] == "LogisticRegression"
    assert selection["note"] is None
    assert 1 <= selection["n_features_selected"] <= selection["n_features_total"] == 8
    assert selection["selected_features"]
    assert set(selection["selected_features"]) <= {"x1", "x2", "x3"} | {f"noise{i}" for i in range(5)}


def test_select_features_regression_uses_ridge(tmp_path):
    rng = np.random.default_rng(9)
    n = 100
    df = pd.DataFrame({f"f{i}": rng.random(n) for i in range(6)})
    df["target"] = df["f0"] * 3 + df["f1"] + rng.normal(0, 0.01, n)
    path = tmp_path / "reg.csv"
    df.to_csv(path, index=False)

    selection = select_features(str(path), "target", "regression", None, [], "rmse")
    assert selection["enabled"] is True
    assert selection["basic_model"] == "Ridge"


def test_all_candidates_share_the_selected_subset(informative_plus_noise):
    """The subset chosen by the basic model is applied to a DIFFERENT model
    family (random forest) — the point of the shared-selection design."""
    selection = select_features(informative_plus_noise, "target", "classification", None, [], "f1")
    result = _run("fs-shared", informative_plus_noise, selection["selected_features"])
    assert result["status"] == "succeeded", result.get("error")
    fs = result["feature_selection"]
    assert fs["enabled"] is True
    assert fs["selected_features"] == selection["selected_features"]
    assert fs["n_features_total"] == 8
    # importance is computed over the selected space only
    importance_names = {f["feature"] for f in result["feature_importance"]}
    assert importance_names <= set(selection["selected_features"])


def test_narrow_dataset_skips_with_note(tmp_path):
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"a": rng.random(60), "b": rng.random(60), "target": rng.integers(0, 2, 60)})
    path = tmp_path / "narrow.csv"
    df.to_csv(path, index=False)

    selection = select_features(str(path), "target", "classification", None, [], "f1")
    assert selection["enabled"] is False
    assert "too few" in selection["note"]
    assert selection["selected_features"] == []

    # the job trains on all features and echoes the skip note
    result = _run("fs-narrow", str(path), selection["selected_features"] or None, selection["note"],
                  estimator="LogisticRegression")
    assert result["status"] == "succeeded", result.get("error")
    assert result["feature_selection"]["enabled"] is False
    assert "too few" in result["feature_selection"]["note"]


def test_disabled_runs_report_enabled_false(informative_plus_noise):
    result = _run("fs-off", informative_plus_noise, None)
    assert result["status"] == "succeeded", result.get("error")
    fs = result["feature_selection"]
    assert fs["enabled"] is False
    assert fs["note"] is None
    assert result["feature_importance"]


def test_select_features_never_raises(tmp_path):
    selection = select_features(str(tmp_path / "missing.csv"), "target", "classification", None, [], "f1")
    assert selection["enabled"] is False
    assert "skipped after error" in selection["note"]
