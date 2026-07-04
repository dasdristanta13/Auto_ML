"""Async training job dispatch + polling.

CLAUDE.md rule #4: training is async, never inline — `train_model` returns a
run_id immediately; polling happens in a separate loop with backoff.

Local dev stand-in for Celery/Ray: a bounded ThreadPoolExecutor plays the role
of the job queue, and an in-memory registry (keyed by run_id) plays the role
of the result backend. The call sites (`train_model` / `poll_training_job`)
are the real interface — swap this module's internals for a Celery task +
Redis-backed result store in production without touching agents/graph code.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
import yaml
from langchain_core.tools import tool
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

_RUNTIME_CONFIG_PATH = "config/runtime.yaml"
ARTIFACT_DIR = Path("artifacts/models")

_registry: dict[str, dict[str, Any]] = {}
_executor: Optional[ThreadPoolExecutor] = None
_futures: dict[str, Future] = {}


def _runtime_config() -> dict[str, Any]:
    with open(_RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["training"]


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=_runtime_config()["max_concurrent_jobs"])
    return _executor


def _build_estimator(library: str, estimator: str, hyperparams: dict[str, Any]):
    if library == "sklearn":
        import sklearn.ensemble as ens
        import sklearn.linear_model as lm

        registry = {
            "LogisticRegression": lm.LogisticRegression,
            "LinearRegression": lm.LinearRegression,
            "Ridge": lm.Ridge,
            "RandomForestClassifier": ens.RandomForestClassifier,
            "RandomForestRegressor": ens.RandomForestRegressor,
            "GradientBoostingClassifier": ens.GradientBoostingClassifier,
            "GradientBoostingRegressor": ens.GradientBoostingRegressor,
        }
    elif library == "xgboost":
        import xgboost as xgb

        registry = {"XGBClassifier": xgb.XGBClassifier, "XGBRegressor": xgb.XGBRegressor}
    elif library == "lightgbm":
        import lightgbm as lgb

        registry = {"LGBMClassifier": lgb.LGBMClassifier, "LGBMRegressor": lgb.LGBMRegressor}
    else:
        raise ValueError(f"unknown library '{library}'")

    if estimator not in registry:
        raise ValueError(f"unknown estimator '{estimator}' for library '{library}'")
    return registry[estimator](**hyperparams)


def _split(
    df: pd.DataFrame, target_column: str, task_type: str, time_column: Optional[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if time_column and time_column in df.columns:
        # chronological split avoids leakage on time-series data (CLAUDE.md
        # time-series fixture requirement) — never shuffle time-ordered rows.
        ordered = df.sort_values(time_column)
        split_idx = int(len(ordered) * 0.8)
        train_df, test_df = ordered.iloc[:split_idx], ordered.iloc[split_idx:]
        feature_cols = [c for c in df.columns if c not in (target_column, time_column)]
        return (
            train_df[feature_cols],
            test_df[feature_cols],
            train_df[target_column],
            test_df[target_column],
        )

    feature_cols = [c for c in df.columns if c != target_column]
    stratify = df[target_column] if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df[target_column], test_size=0.2, random_state=0, stratify=stratify
    )
    return X_train, X_test, y_train, y_test


def _evaluate(task_type: str, y_test: pd.Series, y_pred, y_proba=None) -> dict[str, float]:
    if task_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average="weighted")),
        }
        if y_proba is not None and len(set(y_test)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            except ValueError:
                pass
        return metrics
    return {
        "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }


def _run_job(
    run_id: str,
    dataset_path: str,
    target_column: str,
    task_type: str,
    library: str,
    estimator_name: str,
    hyperparams: dict[str, Any],
    time_column: Optional[str],
) -> None:
    start = time.monotonic()
    _registry[run_id]["status"] = "running"
    try:
        df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_parquet(dataset_path)
        df = df.dropna(subset=[target_column])

        label_encoder = None
        y_full = df[target_column]
        if task_type == "classification" and not pd.api.types.is_numeric_dtype(y_full):
            label_encoder = LabelEncoder()
            df = df.copy()
            df[target_column] = label_encoder.fit_transform(y_full.astype(str))

        X_train, X_test, y_train, y_test = _split(df, target_column, task_type, time_column)

        # numeric-only guard for this local reference implementation — the
        # feature engineering node is responsible for encoding categoricals
        # before this stage in the real pipeline.
        X_train_numeric = X_train.select_dtypes(include="number").fillna(0)
        X_test_numeric = X_test[X_train_numeric.columns].fillna(0)

        estimator = _build_estimator(library, estimator_name, hyperparams)
        estimator.fit(X_train_numeric, y_train)
        y_pred = estimator.predict(X_test_numeric)
        y_proba = None
        if task_type == "classification" and hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X_test_numeric)
            if proba.shape[1] == 2:
                y_proba = proba[:, 1]

        metrics = _evaluate(task_type, y_test, y_pred, y_proba)

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = ARTIFACT_DIR / f"{run_id}.joblib"
        joblib.dump({"estimator": estimator, "label_encoder": label_encoder, "feature_columns": list(X_train_numeric.columns)}, model_path)

        _registry[run_id].update(
            status="succeeded",
            metrics=metrics,
            duration_seconds=time.monotonic() - start,
            model_path=str(model_path),
        )
    except Exception as exc:  # noqa: BLE001 - failure surfaced via registry, not raised across the thread boundary
        _registry[run_id].update(
            status="failed",
            error=str(exc),
            duration_seconds=time.monotonic() - start,
        )


@tool
def train_model(
    candidate_name: str,
    library: str,
    estimator: str,
    hyperparams: dict[str, Any],
    dataset_path: str,
    target_column: str,
    task_type: str,
    time_column: Optional[str] = None,
) -> str:
    """Dispatch an async training job for one candidate model and return its
    run_id IMMEDIATELY (does not block on training completion). Use
    poll_training_job(run_id) to check status. `library` is one of "sklearn",
    "xgboost", "lightgbm"; `estimator` is the class name within that library
    (e.g. "RandomForestClassifier"). Pass `time_column` for time-series data
    so the train/test split is chronological rather than random.
    """
    run_id = str(uuid.uuid4())
    _registry[run_id] = {
        "run_id": run_id,
        "candidate_name": candidate_name,
        "status": "pending",
        "metrics": {},
        "error": None,
        "model_path": None,
    }
    future = _get_executor().submit(
        _run_job, run_id, dataset_path, target_column, task_type, library, estimator, hyperparams, time_column
    )
    _futures[run_id] = future
    return run_id


@tool
def poll_training_job(run_id: str) -> dict[str, Any]:
    """Return the current status snapshot for a previously dispatched training
    run_id: {run_id, candidate_name, status, metrics, error, model_path}.
    status is one of "pending", "running", "succeeded", "failed".
    """
    if run_id not in _registry:
        raise ValueError(f"unknown run_id '{run_id}'")
    return dict(_registry[run_id])
