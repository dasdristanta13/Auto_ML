"""Statistical preprocessing (mean/median imputation, scaling, target
encoding) must be fit on the training fold only, inside the training job's
pipeline — never on the full dataset before the split. Fitting on the full
dataset leaks test-fold statistics, and full-dataset target encoding leaks
each row's own label into its features (inflating every reported metric)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.graph.nodes import apply_feature_plan_node
from src.state import new_state
from src.training.dispatch import _build_preprocessor, poll_training_job, train_model


def _wait_for(run_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = poll_training_job.invoke({"run_id": run_id})
        if result["status"] in ("succeeded", "failed"):
            return result
        time.sleep(0.2)
    raise TimeoutError(f"training job {run_id} did not finish in {timeout}s")


def test_apply_feature_plan_defers_statistical_steps(tmp_path):
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, None, 4.0] * 25,
            "cat": ["a", "b", "a", "b"] * 25,
            "junk": range(100),
            "target": [0, 1] * 50,
        }
    )
    dataset_path = tmp_path / "data.csv"
    df.to_csv(dataset_path, index=False)

    state = new_state(run_id="defer-test", dataset_path=str(dataset_path), use_case_description="test")
    state["task_spec"] = {"target_column": "target", "task_type": "classification", "metric": "f1"}
    state["feature_plan"] = {
        "steps": [
            {"op": "drop", "columns": ["junk"], "params": {}, "rationale": ""},
            {"op": "impute", "columns": ["num"], "params": {"strategy": "mean"}, "rationale": ""},
            {"op": "scale", "columns": ["num"], "params": {"method": "standard"}, "rationale": ""},
            {"op": "encode", "columns": ["cat"], "params": {"method": "target", "target_column": "target"}, "rationale": ""},
        ],
        "plan_rationale": "",
    }

    state = apply_feature_plan_node(state)

    assert state["feature_plan_valid"] is True
    transformed = pd.read_csv(state["transformed_dataset_path"])
    # stateless step applied upfront
    assert "junk" not in transformed.columns
    # statistical steps must NOT have touched the persisted dataset ...
    assert transformed["num"].isna().sum() > 0, "imputation must be deferred to the training fold"
    assert not pd.api.types.is_numeric_dtype(transformed["cat"]), "target encoding must be deferred to the training fold"
    assert set(transformed["cat"].unique()) == {"a", "b"}
    # ... and must instead be recorded for the training job
    deferred_ops = [(s["op"], s.get("params", {}).get("method") or s.get("params", {}).get("strategy")) for s in state["training_preprocess_steps"]]
    assert deferred_ops == [("impute", "mean"), ("scale", "standard"), ("encode", "target")]


def test_preprocessor_fits_imputation_on_training_fold_only():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan]})
    X_test = pd.DataFrame({"a": [np.nan, 100.0]})
    steps = [{"op": "impute", "columns": ["a"], "params": {"strategy": "mean"}}]

    prep = _build_preprocessor(steps, X_train)
    prep.fit(X_train, pd.Series([0, 1, 0, 1]))
    out = prep.transform(X_test)

    assert out[0][0] == 2.0, "test-fold NaN must be filled with the TRAINING fold mean"


def test_target_encoding_does_not_leak_labels(tmp_path):
    """Regression for the metric-inflation bug: a pure-noise high-cardinality
    categorical with full-dataset target encoding used to score near-perfectly
    (each rare category encoded its own rows' labels). Fit on the training
    fold only, a noise feature must score near chance on the holdout."""
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(
        {
            "cat": [f"c{i // 2}" for i in range(n)],  # 300 categories, 2 rows each
            "target": rng.choice([0, 1], n),
        }
    )
    dataset_path = tmp_path / "noise.csv"
    df.to_csv(dataset_path, index=False)

    run_id = train_model.invoke(
        {
            "candidate_name": "leak-check",
            "library": "sklearn",
            "estimator": "RandomForestClassifier",
            "hyperparams": {"n_estimators": 50, "random_state": 0},
            "dataset_path": str(dataset_path),
            "target_column": "target",
            "task_type": "classification",
            "cv_enabled": False,
            "preprocess_steps": [
                {"op": "encode", "columns": ["cat"], "params": {"method": "target", "target_column": "target"}}
            ],
        }
    )
    result = _wait_for(run_id)

    assert result["status"] == "succeeded", result.get("error")
    assert result["metrics"]["accuracy"] < 0.70, (
        f"holdout accuracy {result['metrics']['accuracy']:.3f} on pure noise implies target leakage"
    )


def test_predict_accepts_raw_categorical_input(tmp_path):
    """With preprocessing inside the model pipeline, the saved bundle takes
    raw feature values (including category strings), not pre-scaled numbers."""
    from src.training.dispatch import predict_one

    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            "num": rng.normal(10, 2, n),
            "city": rng.choice(["berlin", "tokyo", "lima"], n),
            "target": rng.choice([0, 1], n),
        }
    )
    dataset_path = tmp_path / "mixed.csv"
    df.to_csv(dataset_path, index=False)

    run_id = train_model.invoke(
        {
            "candidate_name": "raw-input",
            "library": "sklearn",
            "estimator": "LogisticRegression",
            "hyperparams": {"max_iter": 200},
            "dataset_path": str(dataset_path),
            "target_column": "target",
            "task_type": "classification",
            "cv_enabled": False,
            "preprocess_steps": [
                {"op": "scale", "columns": ["num"], "params": {"method": "standard"}},
                {"op": "encode", "columns": ["city"], "params": {"method": "target", "target_column": "target"}},
            ],
        }
    )
    result = _wait_for(run_id)
    assert result["status"] == "succeeded", result.get("error")

    out = predict_one(result["model_path"], {"num": 10.5, "city": "tokyo"})
    assert out["prediction"] in (0, 1)
