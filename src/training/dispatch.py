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
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_validate, train_test_split
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
    # stratification requires every class to have >= 2 members; a target
    # column with singleton classes (wrong/high-cardinality column choice)
    # would otherwise raise and fail every candidate outright.
    can_stratify = task_type == "classification" and df[target_column].value_counts().min() >= 2
    stratify = df[target_column] if can_stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df[target_column], test_size=0.2, random_state=0, stratify=stratify
    )
    return X_train, X_test, y_train, y_test


_TOP_N_FEATURE_IMPORTANCE = 8


def _feature_importance(estimator, feature_names: list[str]) -> list[dict[str, Any]]:
    """Best-effort extraction for the report view's feature-importance chart
    (PRODUCT.md 3.4). Tree ensembles expose feature_importances_ directly;
    linear models expose coef_ (importance taken as |coef|, normalized to sum
    to 1 so it reads the same way across model types)."""
    if hasattr(estimator, "feature_importances_"):
        raw = list(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        raw = list(abs(coef[0]) if getattr(coef, "ndim", 1) > 1 else abs(coef))
    else:
        return []

    total = sum(raw) or 1.0
    ranked = sorted(zip(feature_names, raw), key=lambda pair: pair[1], reverse=True)
    return [{"feature": name, "importance": round(value / total, 4)} for name, value in ranked[:_TOP_N_FEATURE_IMPORTANCE]]


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


_CV_REGRESSION_SCORERS = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
_CV_REGRESSION_SIGN_FLIP = {"rmse", "mae"}


def _cross_validate(
    estimator, X: pd.DataFrame, y: pd.Series, task_type: str, time_column: Optional[str], requested_folds: int
) -> dict[str, Any]:
    """K-fold cross-validation on the training split, reported alongside the
    holdout metrics so a single lucky/unlucky split isn't mistaken for a
    reliable estimate. Uses TimeSeriesSplit (no shuffling, no leakage from
    the future) when the candidate is chronologically split, StratifiedKFold
    for classification, plain KFold for regression. Folds are auto-reduced
    (floor of 2) for small datasets or rare classes rather than raising; if
    even 2 folds aren't possible, CV is skipped with an explanatory note —
    never silently omitted without saying why."""
    n = len(X)
    if time_column:
        folds = min(requested_folds, n - 1)
        splitter = TimeSeriesSplit(n_splits=folds) if folds >= 2 else None
    elif task_type == "classification":
        min_class = int(y.value_counts().min())
        folds = min(requested_folds, min_class, n)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0) if folds >= 2 else None
    else:
        folds = min(requested_folds, n)
        splitter = KFold(n_splits=folds, shuffle=True, random_state=0) if folds >= 2 else None

    if splitter is None or folds < 2:
        return {"folds": 0, "metrics": {}, "note": "cross-validation skipped: not enough samples per class/fold"}

    if task_type == "classification":
        scoring = {"accuracy": "accuracy", "f1": "f1_weighted"}
        if y.nunique() == 2:
            scoring["roc_auc"] = "roc_auc"
        sign_flip: set[str] = set()
    else:
        scoring = _CV_REGRESSION_SCORERS
        sign_flip = _CV_REGRESSION_SIGN_FLIP

    scores = cross_validate(clone(estimator), X, y, cv=splitter, scoring=scoring)

    metrics: dict[str, Any] = {}
    for name in scoring:
        raw = scores[f"test_{name}"]
        values = -raw if name in sign_flip else raw
        metrics[name] = {"mean": float(values.mean()), "std": float(values.std())}
    return {"folds": folds, "metrics": metrics, "note": None}


def _build_resampler(method: str, y: pd.Series) -> tuple[Optional[Any], Optional[str], Optional[str]]:
    """Returns (resampler, applied_method, note). SMOTE needs k_neighbors <
    the minority class size; auto-reduces it and falls back to random
    oversampling if even 2 neighbors aren't available — same "auto-adjust
    and explain, never silently omit" pattern as the CV fold auto-reduction
    above. `applied_method` reflects what actually ran (may differ from the
    requested `method` after a fallback)."""
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    if method == "smote":
        minority_count = int(y.value_counts().min())
        k = min(5, minority_count - 1)
        if k < 1:
            note = (
                f"SMOTE needs at least 2 minority-class samples (had {minority_count}); used random "
                "oversampling instead."
            )
            return RandomOverSampler(random_state=0), "random_oversample", note
        return SMOTE(k_neighbors=k, random_state=0), "smote", None
    if method == "random_oversample":
        return RandomOverSampler(random_state=0), "random_oversample", None
    if method == "random_undersample":
        return RandomUnderSampler(random_state=0), "random_undersample", None
    return None, None, None


def _run_job(
    run_id: str,
    dataset_path: str,
    target_column: str,
    task_type: str,
    library: str,
    estimator_name: str,
    hyperparams: dict[str, Any],
    time_column: Optional[str],
    cv_enabled: bool,
    cv_folds: Optional[int],
    resampling_enabled: bool,
    resampling_method: str,
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

        resampler, resampling_applied, resampling_note = (None, None, None)
        if resampling_enabled and resampling_method != "none" and task_type == "classification":
            resampler, resampling_applied, resampling_note = _build_resampler(resampling_method, y_train)

        if resampler is not None:
            from imblearn.pipeline import Pipeline as ImbPipeline

            # imblearn's Pipeline only resamples during .fit() — .predict()
            # and cross_validate()'s held-out scoring pass through untouched,
            # which is exactly what keeps synthetic/duplicated rows out of
            # the test fold (no leakage across the train/test or CV boundary).
            fit_estimator = ImbPipeline([("resample", resampler), ("model", estimator)])
        else:
            fit_estimator = estimator

        if cv_enabled:
            cv_folds_requested = cv_folds if cv_folds is not None else _runtime_config()["cv_folds"]
            cv_result = _cross_validate(fit_estimator, X_train_numeric, y_train, task_type, time_column, cv_folds_requested)
        else:
            cv_result = {"folds": 0, "metrics": {}, "note": "cross-validation disabled for this run"}

        fit_estimator.fit(X_train_numeric, y_train)
        y_pred = fit_estimator.predict(X_test_numeric)
        y_proba = None
        if task_type == "classification" and hasattr(fit_estimator, "predict_proba"):
            proba = fit_estimator.predict_proba(X_test_numeric)
            if proba.shape[1] == 2:
                y_proba = proba[:, 1]

        metrics = _evaluate(task_type, y_test, y_pred, y_proba)
        fitted_model = fit_estimator.named_steps["model"] if resampler is not None else fit_estimator
        feature_importance = _feature_importance(fitted_model, list(X_train_numeric.columns))

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = ARTIFACT_DIR / f"{run_id}.joblib"
        joblib.dump({"estimator": fit_estimator, "label_encoder": label_encoder, "feature_columns": list(X_train_numeric.columns)}, model_path)

        _registry[run_id].update(
            status="succeeded",
            metrics=metrics,
            duration_seconds=time.monotonic() - start,
            model_path=str(model_path),
            feature_importance=feature_importance,
            cv_folds=cv_result["folds"],
            cv_metrics=cv_result["metrics"],
            cv_note=cv_result["note"],
            resampling_applied=resampling_applied,
            resampling_note=resampling_note,
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
    cv_enabled: bool = True,
    cv_folds: Optional[int] = None,
    resampling_enabled: bool = False,
    resampling_method: str = "none",
) -> str:
    """Dispatch an async training job for one candidate model and return its
    run_id IMMEDIATELY (does not block on training completion). Use
    poll_training_job(run_id) to check status. `library` is one of "sklearn",
    "xgboost", "lightgbm"; `estimator` is the class name within that library
    (e.g. "RandomForestClassifier"). Pass `time_column` for time-series data
    so the train/test split is chronological rather than random. `cv_enabled`/
    `cv_folds` are the user's choice at the confirm checkpoint (folds defaults
    to config/runtime.yaml's cv_folds when not given); set cv_enabled=False to
    skip k-fold cross-validation entirely for this run. `resampling_enabled`/
    `resampling_method` ("smote" | "random_oversample" | "random_undersample")
    are the user's class-balancing choice from the feature-approval checkpoint
    — applied to the training fold only (classification tasks only; ignored
    for regression), never to the held-out test/CV fold, so it can't leak.
    """
    run_id = str(uuid.uuid4())
    _registry[run_id] = {
        "run_id": run_id,
        "candidate_name": candidate_name,
        "status": "pending",
        "metrics": {},
        "error": None,
        "model_path": None,
        "feature_importance": [],
        "cv_folds": 0,
        "cv_metrics": {},
        "cv_note": None,
        "resampling_applied": None,
        "resampling_note": None,
    }
    future = _get_executor().submit(
        _run_job,
        run_id,
        dataset_path,
        target_column,
        task_type,
        library,
        estimator,
        hyperparams,
        time_column,
        cv_enabled,
        cv_folds,
        resampling_enabled,
        resampling_method,
    )
    _futures[run_id] = future
    return run_id


@tool
def poll_training_job(run_id: str) -> dict[str, Any]:
    """Return the current status snapshot for a previously dispatched training
    run_id: {run_id, candidate_name, status, metrics, error, model_path,
    feature_importance, cv_folds, cv_metrics, cv_note, resampling_applied,
    resampling_note}. status is one of "pending", "running", "succeeded",
    "failed". feature_importance is a best-effort ranked list (may be empty
    for estimators that expose neither feature_importances_ nor coef_).
    cv_metrics is {metric: {mean, std}} from k-fold cross-validation on the
    training split (cv_folds may be auto-reduced from config/runtime.yaml's
    requested value, or 0 with cv_note explaining why if cross-validation
    wasn't possible). resampling_applied is the class-balancing method that
    actually ran ("smote"/"random_oversample"/"random_undersample"), or None
    if resampling wasn't requested/applicable; resampling_note explains an
    auto-fallback (e.g. SMOTE -> random oversampling when the minority class
    was too small).
    """
    if run_id not in _registry:
        raise ValueError(f"unknown run_id '{run_id}'")
    return dict(_registry[run_id])


def load_model_schema(model_path: str) -> dict[str, Any]:
    """Feature columns the saved model expects — lets the frontend's 'test
    the model' tab build an input form without hardcoding anything."""
    bundle = joblib.load(model_path)
    return {"feature_columns": bundle["feature_columns"]}


def predict_one(model_path: str, values: dict[str, Any]) -> dict[str, Any]:
    """Score a single user-supplied row against a saved model bundle. Missing
    features are filled with 0, matching the fillna(0) convention used at
    training time (see _run_job above)."""
    bundle = joblib.load(model_path)
    estimator = bundle["estimator"]
    feature_columns = bundle["feature_columns"]
    label_encoder = bundle.get("label_encoder")

    row = pd.DataFrame([{col: values.get(col, 0) for col in feature_columns}])
    raw_prediction = estimator.predict(row)[0]
    prediction = (
        label_encoder.inverse_transform([int(raw_prediction)])[0] if label_encoder is not None else raw_prediction
    )

    result: dict[str, Any] = {"prediction": prediction}
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(row)[0]
        classes = estimator.classes_
        if label_encoder is not None:
            classes = label_encoder.inverse_transform(classes.astype(int))
        result["probabilities"] = {str(c): float(p) for c, p in zip(classes, proba)}
    return result
