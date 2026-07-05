"""Optuna hyperparameter tuning inside the async training job.

Contract (docs/superpowers/specs/2026-07-04-hyperparameter-tuning-design.md):
- the LLM-proposed hyperparams are trial 0 (baseline), so the tuned model can
  never score worse in CV than the untuned one;
- per-trial progress is written to the job registry as it happens
  ({trials_done, history: [{trial, score, best_score}], best_params, ...});
- estimators with nothing to tune, or tuning_enabled=False, skip with a note.
"""

from __future__ import annotations

import time

import numpy as np
import optuna
import pandas as pd
import pytest

from src.training.dispatch import _suggest_hyperparams, poll_training_job, train_model

_CANONICAL = [
    ("sklearn", "LogisticRegression"),
    ("sklearn", "Ridge"),
    ("sklearn", "RandomForestClassifier"),
    ("sklearn", "RandomForestRegressor"),
    ("sklearn", "GradientBoostingClassifier"),
    ("sklearn", "GradientBoostingRegressor"),
    ("xgboost", "XGBClassifier"),
    ("xgboost", "XGBRegressor"),
    ("lightgbm", "LGBMClassifier"),
    ("lightgbm", "LGBMRegressor"),
]


@pytest.mark.parametrize("library,estimator", _CANONICAL)
def test_search_space_defined_for_tunable_estimators(library, estimator):
    study = optuna.create_study()
    trial = study.ask()
    params = _suggest_hyperparams(trial, library, estimator)
    assert params, f"no search space for {library}.{estimator}"


def test_search_space_empty_for_linear_regression():
    study = optuna.create_study()
    trial = study.ask()
    assert _suggest_hyperparams(trial, "sklearn", "LinearRegression") == {}


def _wait_for(run_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = poll_training_job.invoke({"run_id": run_id})
        if result["status"] in ("succeeded", "failed"):
            return result
        time.sleep(0.2)
    raise TimeoutError(f"training job {run_id} did not finish in {timeout}s")


def _dataset(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = (x1 + 0.5 * x2 + rng.normal(0, 0.5, n) > 0).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_tuned_job_records_history_and_best_params(tmp_path):
    run_id = train_model.invoke(
        {
            "candidate_name": "tuned-rf",
            "library": "sklearn",
            "estimator": "RandomForestClassifier",
            "hyperparams": {"n_estimators": 20, "max_depth": 3, "random_state": 0},
            "dataset_path": _dataset(tmp_path),
            "target_column": "target",
            "task_type": "classification",
            "cv_enabled": False,
            "tuning_enabled": True,
            "tuning_trials": 4,
            "tuning_metric": "f1",
        }
    )
    result = _wait_for(run_id)

    assert result["status"] == "succeeded", result.get("error")
    tuning = result["tuning"]
    assert tuning["enabled"] is True
    assert tuning["trials_total"] == 4
    assert 1 <= tuning["trials_done"] <= 4
    assert tuning["metric"] == "f1"

    history = tuning["history"]
    assert history[0]["trial"] == 0, "trial 0 must be the LLM-proposed baseline"
    assert len(history) == tuning["trials_done"]
    # best_score must be monotone non-decreasing for a maximize metric and
    # never worse than the baseline (trial 0)
    best_scores = [h["best_score"] for h in history]
    assert best_scores == sorted(best_scores)
    assert best_scores[-1] >= history[0]["score"]
    assert isinstance(tuning["best_params"], dict)


def test_tuning_disabled_short_circuits(tmp_path):
    run_id = train_model.invoke(
        {
            "candidate_name": "untuned",
            "library": "sklearn",
            "estimator": "LogisticRegression",
            "hyperparams": {"max_iter": 200},
            "dataset_path": _dataset(tmp_path),
            "target_column": "target",
            "task_type": "classification",
            "cv_enabled": False,
            "tuning_enabled": False,
        }
    )
    result = _wait_for(run_id)

    assert result["status"] == "succeeded", result.get("error")
    assert result["tuning"]["enabled"] is False
    assert result["tuning"]["history"] == []


def test_tuning_skipped_when_nothing_to_tune(tmp_path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 100)})
    df["y"] = 2 * df["x"] + rng.normal(0, 0.1, 100)
    path = tmp_path / "reg.csv"
    df.to_csv(path, index=False)

    run_id = train_model.invoke(
        {
            "candidate_name": "linreg",
            "library": "sklearn",
            "estimator": "LinearRegression",
            "hyperparams": {},
            "dataset_path": str(path),
            "target_column": "y",
            "task_type": "regression",
            "cv_enabled": False,
            "tuning_enabled": True,
            "tuning_trials": 4,
            "tuning_metric": "rmse",
        }
    )
    result = _wait_for(run_id)

    assert result["status"] == "succeeded", result.get("error")
    assert result["tuning"]["enabled"] is False
    assert result["tuning"]["note"], "skipping tuning must carry an explanatory note"
