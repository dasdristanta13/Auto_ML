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

import base64
import inspect
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
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
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler, TargetEncoder

from src.data_io import load_dataset

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


def _estimator_registry(library: str) -> dict[str, type]:
    """Single source of truth for valid estimator names per library — used
    by both training dispatch and model_selection_node's pre-dispatch
    validation. Lazily imports each library so an uninstalled optional
    dependency (xgboost/lightgbm) only fails when actually requested."""
    if library == "sklearn":
        import sklearn.ensemble as ens
        import sklearn.linear_model as lm

        return {
            "LogisticRegression": lm.LogisticRegression,
            "LinearRegression": lm.LinearRegression,
            "Ridge": lm.Ridge,
            "RandomForestClassifier": ens.RandomForestClassifier,
            "RandomForestRegressor": ens.RandomForestRegressor,
            "GradientBoostingClassifier": ens.GradientBoostingClassifier,
            "GradientBoostingRegressor": ens.GradientBoostingRegressor,
        }
    if library == "xgboost":
        import xgboost as xgb

        return {"XGBClassifier": xgb.XGBClassifier, "XGBRegressor": xgb.XGBRegressor}
    if library == "lightgbm":
        import lightgbm as lgb

        return {"LGBMClassifier": lgb.LGBMClassifier, "LGBMRegressor": lgb.LGBMRegressor}
    raise ValueError(f"unknown library '{library}'")


def known_estimators(library: str) -> set[str]:
    """Best-effort: returns an empty set if the library isn't installed in
    this environment, rather than raising. model_selection_node uses this to
    validate LLM-proposed estimator names before a candidate ever reaches
    dispatch — only called for libraries _library_available() already
    confirmed importable, so ImportError here is a defensive fallback, not
    the expected path."""
    try:
        return set(_estimator_registry(library))
    except (ImportError, ValueError):
        return set()


_DEPRECATED_HYPERPARAM_VALUES: dict[str, dict[Any, dict[str, Any]]] = {
    # param_name -> {llm_supplied_value: {"classifier": replacement, "regressor": replacement}}
    "max_features": {"auto": {"classifier": "sqrt", "regressor": None}},
}


def _sanitize_hyperparams(estimator_cls: type, hyperparams: dict[str, Any]) -> dict[str, Any]:
    """Defense-in-depth for CandidateModel.hyperparams (src/state.py), an
    untyped dict[str, Any] the LLM controls with no schema/enum validating
    individual names or values. Three independent protections:
    1. Key/value hygiene, applied to EVERY estimator: strip stray whitespace
       from keys and string values (an LLM emitting " reg_alpha" fails
       XGBoost's C++ parameter validation at fit time), and drop keys that
       aren't valid Python identifiers after stripping — no constructor or
       **kwargs consumer accepts those.
    2. Drop any key estimator_cls.__init__ doesn't accept, via signature
       introspection, instead of a typo'd/hallucinated name crashing
       construction with TypeError. Skipped when __init__ declares
       **kwargs (some XGBoost/LightGBM versions accept arbitrary extra
       keys), since nothing can be validated against an open signature.
    3. Translate known deprecated/renamed values (e.g. sklearn's
       max_features="auto", removed in 1.3) to their modern equivalent.
    """
    sanitized: dict[str, Any] = {}
    for key, value in hyperparams.items():
        key = key.strip() if isinstance(key, str) else key
        if not isinstance(key, str) or not key.isidentifier():
            continue
        if isinstance(value, str):
            value = value.strip()
        sanitized[key] = value

    params = inspect.signature(estimator_cls.__init__).parameters
    accepts_arbitrary_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if not accepts_arbitrary_kwargs:
        valid_names = set(params) - {"self"}
        sanitized = {k: v for k, v in sanitized.items() if k in valid_names}

    kind = "classifier" if estimator_cls.__name__.endswith("Classifier") else "regressor"
    for param, value_map in _DEPRECATED_HYPERPARAM_VALUES.items():
        if param in sanitized and sanitized[param] in value_map:
            sanitized[param] = value_map[sanitized[param]][kind]

    return sanitized


def _fix_target_incompatibilities(
    estimator_name: str, hyperparams: dict[str, Any], task_type: str, n_classes: int
) -> tuple[dict[str, Any], Optional[str]]:
    """Repairs hyperparameters that are individually valid but jointly invalid
    with THIS dataset's target — something _sanitize_hyperparams can't catch
    because it never sees the data. Returns (params, note); note explains any
    change so the run surfaces it instead of silently differing from the plan.

    Known case: LogisticRegression solver='liblinear' is binary-only (sklearn
    >= 1.7 removed its OvR multiclass path), so a plausible LLM proposal fails
    every fit on a 3+-class target. Swap to the closest multiclass-capable
    solver: 'saga' when the penalty needs it (l1/elasticnet), else 'lbfgs';
    'dual' is liblinear-specific and dropped alongside."""
    if estimator_name == "LogisticRegression" and task_type == "classification" and n_classes >= 3:
        if hyperparams.get("solver") == "liblinear":
            fixed = dict(hyperparams)
            fixed.pop("dual", None)
            fixed["solver"] = "saga" if fixed.get("penalty") in ("l1", "elasticnet") else "lbfgs"
            note = (
                f"solver 'liblinear' does not support {n_classes}-class classification; "
                f"switched to '{fixed['solver']}'"
            )
            return fixed, note
    return hyperparams, None


def _build_estimator(library: str, estimator: str, hyperparams: dict[str, Any]):
    registry = _estimator_registry(library)
    if estimator not in registry:
        raise ValueError(f"unknown estimator '{estimator}' for library '{library}'")
    estimator_cls = registry[estimator]
    hyperparams = _sanitize_hyperparams(estimator_cls, hyperparams)
    return estimator_cls(**hyperparams)


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


_SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


def _build_preprocessor(steps: list[dict[str, Any]], X: pd.DataFrame) -> ColumnTransformer:
    """Fold-safe preprocessing from the plan's deferred statistical steps
    (see src/graph/nodes._is_training_time_step). Fit on the training fold
    only — inside cross_validate it is re-fit per fold, so no test-fold
    statistic (or, for target encoding, any label) ever reaches the model.

    Built exclusively from sklearn estimators so the saved .joblib bundle
    stays loadable without this repo installed. Composition per column:
      - numeric: SimpleImputer (plan strategy, else constant-0 — the
        replacement for the old blanket fillna(0)) then the plan's scaler.
      - target-encoded (any dtype): sklearn TargetEncoder, which cross-fits
        internally during fit_transform and encodes unseen/missing
        categories with the global target mean.
      - all other non-numeric columns are dropped, matching the previous
        numeric-only training guard.
    """
    columns = list(X.columns)
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [
        c for c in columns if isinstance(X[c].dtype, pd.CategoricalDtype) or X[c].dtype in ("object", "string")
    ]

    impute_strategy: dict[str, str] = {}
    scale_method: dict[str, str] = {}
    target_encode_cols: list[str] = []
    for step in steps or []:
        op, params = step.get("op"), step.get("params", {})
        step_cols = [c for c in step.get("columns", []) if c in columns]
        if op == "impute":
            for col in step_cols:
                if col in numeric_cols:
                    impute_strategy[col] = params.get("strategy", "mean")
        elif op == "scale":
            for col in step_cols:
                if col in numeric_cols:
                    scale_method[col] = params.get("method", "standard")
        elif op == "encode" and params.get("method") == "target":
            for col in step_cols:
                if col not in target_encode_cols:
                    target_encode_cols.append(col)

    # group numeric columns sharing the same (impute, scale) recipe so wide
    # datasets don't produce hundreds of single-column transformers
    groups: dict[tuple[Optional[str], Optional[str]], list[str]] = {}
    for col in numeric_cols:
        if col in target_encode_cols:
            continue
        key = (impute_strategy.get(col), scale_method.get(col))
        groups.setdefault(key, []).append(col)

    transformers = []
    for i, ((strategy, scale), group_cols) in enumerate(groups.items()):
        stages: list[tuple[str, Any]] = []
        if strategy in ("mean", "median", "most_frequent"):
            stages.append(("impute", SimpleImputer(strategy=strategy)))
            if scale:
                stages.append(("scale", _SCALERS[scale]()))
        else:
            # sklearn scalers are NaN-tolerant during fit, so scale first and
            # zero-fill whatever remains (the old fillna(0) convention)
            if scale:
                stages.append(("scale", _SCALERS[scale]()))
            stages.append(("impute", SimpleImputer(strategy="constant", fill_value=0.0)))
        transformers.append((f"num_{i}", Pipeline(stages), group_cols))

    if not transformers and not target_encode_cols:
        # a dataset with no numeric features (all-categorical, e.g. mushroom-
        # style data) would otherwise produce a ZERO-column matrix and fail
        # every fit of every candidate ("model is misconfigured"). Fall back
        # to target-encoding the categorical columns instead of training on
        # nothing; TargetEncoder cross-fits internally so this stays
        # leakage-safe even for high-cardinality columns.
        target_encode_cols = list(categorical_cols)
        if not target_encode_cols:
            raise ValueError(
                "no usable feature columns: dataset has no numeric or categorical features to train on"
            )

    if target_encode_cols:
        # default cv/shuffle settings: sklearn 1.9 deprecated random_state
        # here in favor of passing a splitter, which sklearn 1.4 (our floor)
        # doesn't accept — the slight cross-fit shuffle nondeterminism is fine
        transformers.append(("target_encode", TargetEncoder(), target_encode_cols))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


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


def _explainability_config() -> dict[str, Any]:
    with open(_RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["explainability"]


def _reduce_shap_values(values: np.ndarray) -> np.ndarray:
    """Collapse a possible (samples, features, classes) SHAP output down to
    (samples, features): binary classification keeps the positive class'
    contribution (matching predict_one's proba[:, 1] convention elsewhere in
    this module); anything else averages |SHAP| across classes."""
    if values.ndim == 2:
        return values
    if values.shape[-1] == 2:
        return values[:, :, 1]
    return np.abs(values).mean(axis=2)


def _shap_method_label(explainer) -> str:
    name = type(explainer).__name__.lower()
    if "tree" in name:
        return "tree"
    if "linear" in name:
        return "linear"
    return "kernel"


def _build_shap_explainer(model, background: np.ndarray, feature_names: list[str]):
    """Mirrors _feature_importance's own dispatch: tree ensembles
    (feature_importances_) and linear models (coef_) get shap's fast, exact
    Tree/Linear explainer by passing the raw estimator; anything else falls
    back to a model-agnostic explainer over predict_proba/predict."""
    import shap

    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
        return shap.Explainer(model, background, feature_names=feature_names)
    predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    return shap.Explainer(predict_fn, background, feature_names=feature_names)


def _shap_background(transformed_dataset_path: str, feature_columns: list[str], max_rows: int) -> pd.DataFrame:
    df = load_dataset(transformed_dataset_path)
    n = min(max_rows, len(df))
    return df[feature_columns].sample(n=n, random_state=0)


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        plt.close(fig)


@contextmanager
def _close_new_figures():
    """Closes any matplotlib figure created inside the wrapped block, even if
    it raises partway through — shap's plotting functions create the figure
    early and do more work afterward, so a bare try/except around the call
    can leak a Figure on failure. Safe to combine with _fig_to_base64's own
    plt.close on the success path: closing an already-closed figure number
    is a no-op for plt.get_fignums()."""
    before = set(plt.get_fignums())
    try:
        yield
    finally:
        for num in set(plt.get_fignums()) - before:
            plt.close(num)


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
        with _close_new_figures():
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
        with _close_new_figures():
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
            with _close_new_figures():
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
        with _close_new_figures():
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


def _shap_fidelity_r2(
    model, background: np.ndarray, values: np.ndarray, base_values: np.ndarray
) -> Optional[float]:
    """R² between the model's actual output and the SHAP reconstruction
    (base value + sum of per-feature contributions) over the background
    sample — a diagnostic of how well these SHAP values explain the model's
    output. None when there's no single scalar output to compare against
    (e.g. multiclass classification), or on any failure — this is a
    diagnostic extra, never a reason to fail compute_explainability."""
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(background)
            if proba.ndim != 2 or proba.shape[1] != 2:
                return None
            model_output = proba[:, 1]
        else:
            model_output = model.predict(background)
        reconstructed = base_values + values.sum(axis=1)
        return float(r2_score(model_output, reconstructed))
    except Exception:  # noqa: BLE001 - fidelity is a diagnostic extra, never fatal
        return None


def compute_explainability(model_path: str, transformed_dataset_path: str) -> dict[str, Any]:
    """Best-effort aggregate SHAP feature impact for the winning model,
    computed once after training (explainability_node). Never raises —
    unsupported estimators or SHAP failures degrade to method="unavailable"
    with an explanatory note rather than failing the pipeline."""
    try:
        cfg = _explainability_config()
        bundle = joblib.load(model_path)
        fit_estimator = bundle["estimator"]
        prep = fit_estimator.named_steps["prep"]
        model = fit_estimator.named_steps["model"]
        feature_names = [str(n) for n in prep.get_feature_names_out()]

        sample = _shap_background(transformed_dataset_path, bundle["feature_columns"], cfg["max_background_rows"])
        background = np.asarray(prep.transform(sample))

        explainer = _build_shap_explainer(model, background, feature_names)
        shap_values = explainer(background)
        values = _reduce_shap_values(np.asarray(shap_values.values))
        base_values = _reduce_shap_base_values(np.asarray(shap_values.base_values))
        mean_abs = np.abs(values).mean(axis=0)
        mean_signed = values.mean(axis=0)
        fidelity_r2 = _shap_fidelity_r2(model, background, values, base_values)

        ranked = sorted(
            zip(feature_names, mean_abs, mean_signed), key=lambda triple: triple[1], reverse=True
        )
        top_n = cfg["top_n_features"]

        try:
            explanation = _shap_plot_explanation(values, background, feature_names)
            summary_plot = _render_beeswarm_plot(explanation)
            bar_plot = _render_bar_plot(explanation)
            dependence_plots = _render_dependence_plots(
                explanation, [name for name, _, _ in ranked], cfg["dependence_plot_top_n"]
            )
        except Exception:  # noqa: BLE001 - a plotting failure must never blank feature_impact/method
            summary_plot, bar_plot, dependence_plots = None, None, []

        return {
            "method": _shap_method_label(explainer),
            "feature_impact": [
                {"feature": name, "mean_abs_shap": round(float(abs_v), 6), "mean_signed_shap": round(float(signed_v), 6)}
                for name, abs_v, signed_v in ranked[:top_n]
            ],
            "narrative": None,
            "note": None,
            "summary_plot": summary_plot,
            "bar_plot": bar_plot,
            "dependence_plots": dependence_plots,
            "fidelity_r2": fidelity_r2,
            "background_sample_size": int(len(background)),
        }
    except Exception as exc:  # noqa: BLE001 - explainability is best-effort, never fatal
        return {
            "method": "unavailable",
            "feature_impact": [],
            "narrative": None,
            "note": f"SHAP explanation unavailable for this model: {exc}",
            "summary_plot": None,
            "bar_plot": None,
            "dependence_plots": [],
            "fidelity_r2": None,
            "background_sample_size": 0,
        }


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

    # error_score="raise" so a failing fold surfaces its REAL exception —
    # the default (nan) buries it under sklearn's "model is misconfigured"
    # warning and NaN metrics. The caller degrades gracefully: holdout
    # training still proceeds, with the cause recorded in cv_note.
    try:
        scores = cross_validate(clone(estimator), X, y, cv=splitter, scoring=scoring, error_score="raise")
    except Exception as exc:  # noqa: BLE001 - any fold error degrades to "CV unavailable", not a dead job
        return {"folds": 0, "metrics": {}, "note": f"cross-validation failed: {exc}"}

    metrics: dict[str, Any] = {}
    for name in scoring:
        raw = scores[f"test_{name}"]
        values = -raw if name in sign_flip else raw
        metrics[name] = {"mean": float(values.mean()), "std": float(values.std())}
    return {"folds": folds, "metrics": metrics, "note": None}


# ---------------------------------------------------------------------------
# Hyperparameter tuning (Optuna/TPE) — see
# docs/superpowers/specs/2026-07-04-hyperparameter-tuning-design.md.
# Trial 0 is always the LLM-proposed baseline, scored the same way, so the
# tuned model can never do worse in CV than the untuned one. Per-trial
# progress is written to the job registry as it happens; the 2s poll loop
# carries it into PipelineState so the UI can render live progress.
# ---------------------------------------------------------------------------


def _suggest_hyperparams(trial: Any, library: str, estimator: str) -> dict[str, Any]:
    """Search space per estimator. Returns {} when there is nothing to tune
    (plain LinearRegression), which callers treat as 'skip tuning'."""
    if estimator == "LogisticRegression":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "max_iter": 1000,
        }
    if estimator == "Ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}
    if estimator in ("RandomForestClassifier", "RandomForestRegressor"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    if estimator in ("GradientBoostingClassifier", "GradientBoostingRegressor"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    if estimator in ("XGBClassifier", "XGBRegressor"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if estimator in ("LGBMClassifier", "LGBMRegressor"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    return {}


_TUNABLE_ESTIMATORS = {
    "LogisticRegression",
    "Ridge",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "XGBClassifier",
    "XGBRegressor",
    "LGBMClassifier",
    "LGBMRegressor",
}


def _tuning_scoring(task_type: str, metric: Optional[str], y: pd.Series) -> tuple[str, str, bool]:
    """Maps the task-spec metric to (metric_name, sklearn scoring string,
    lower_is_better). Falls back to f1 / rmse when the metric is missing or
    not applicable (e.g. roc_auc on a multiclass target)."""
    if task_type == "classification":
        mapping = {
            "f1": "f1_weighted",
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
        }
        if metric == "roc_auc" and y.nunique() == 2:
            return "roc_auc", "roc_auc", False
        name = metric if metric in mapping else "f1"
        return name, mapping[name], False
    mapping_reg = {
        "rmse": ("neg_root_mean_squared_error", True),
        "mae": ("neg_mean_absolute_error", True),
        "r2": ("r2", False),
    }
    name = metric if metric in mapping_reg else "rmse"
    scoring, lower = mapping_reg[name]
    return name, scoring, lower


def _tuning_splitter(task_type: str, y: pd.Series, time_column: Optional[str]):
    """3-fold CV for the tuning objective, auto-reduced like _cross_validate;
    None means the data can't support even 2 folds and tuning is skipped."""
    if time_column:
        folds = min(3, len(y) - 1)
        return TimeSeriesSplit(n_splits=folds) if folds >= 2 else None
    if task_type == "classification":
        folds = min(3, int(y.value_counts().min()), len(y))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=0) if folds >= 2 else None
    folds = min(3, len(y))
    return KFold(n_splits=folds, shuffle=True, random_state=0) if folds >= 2 else None


def _tune_pipeline(
    run_id: str,
    make_pipeline: Any,  # (params: dict) -> unfitted pipeline
    baseline_params: dict[str, Any],
    library: str,
    estimator_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    metric: Optional[str],
    time_column: Optional[str],
    n_trials: int,
    budget_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (params_for_final_fit, tuning_info). tuning_info matches
    src/state.TuningInfo; it is also mirrored into the job registry after
    every completed trial so polling exposes live progress."""
    metric_name, scoring, lower_is_better = _tuning_scoring(task_type, metric, y)
    info: dict[str, Any] = {
        "enabled": False,
        "trials_total": 0,
        "trials_done": 0,
        "metric": metric_name,
        "lower_is_better": lower_is_better,
        "best_params": {},
        "history": [],
        "note": None,
    }

    if estimator_name not in _TUNABLE_ESTIMATORS:
        info["note"] = f"tuning skipped: {estimator_name} has no tunable hyperparameters"
        return dict(baseline_params), info

    splitter = _tuning_splitter(task_type, y, time_column)
    if splitter is None:
        info["note"] = "tuning skipped: not enough samples per class/fold for tuning CV"
        return dict(baseline_params), info

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _publish() -> None:
        _registry[run_id]["tuning"] = {**info, "history": [dict(h) for h in info["history"]]}

    def _cv_value(params: dict[str, Any]) -> float:
        # maximize orientation (sklearn neg_* scorers are already negated)
        scores = cross_val_score(make_pipeline(params), X, y, cv=splitter, scoring=scoring)
        return float(scores.mean())

    def _natural(value: float) -> float:
        return -value if lower_is_better else value

    info["enabled"] = True
    info["trials_total"] = n_trials
    _publish()

    # trial 0: the LLM-proposed baseline, scored identically to search trials
    best_value = _cv_value(baseline_params)
    best_params = dict(baseline_params)
    info["history"].append({"trial": 0, "score": _natural(best_value), "best_score": _natural(best_value)})
    info["trials_done"] = 1
    _publish()

    def _objective(trial: Any) -> float:
        suggested = _suggest_hyperparams(trial, library, estimator_name)
        return _cv_value({**baseline_params, **suggested})

    def _record(study: Any, trial: Any) -> None:
        nonlocal best_value, best_params
        if trial.value is None:
            return
        if trial.value > best_value:
            best_value = trial.value
            best_params = {**baseline_params, **trial.params}
        info["history"].append(
            {"trial": len(info["history"]), "score": _natural(trial.value), "best_score": _natural(best_value)}
        )
        info["trials_done"] = len(info["history"])
        _publish()

    if n_trials > 1:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(
            _objective,
            n_trials=n_trials - 1,
            timeout=budget_seconds,
            callbacks=[_record],
            catch=(Exception,),
        )

    info["best_params"] = dict(best_params)
    _publish()
    return best_params, info


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


def _load_training_frame(
    dataset_path: str, target_column: str, task_type: str
) -> tuple[pd.DataFrame, Optional[LabelEncoder]]:
    """Shared load + normalization path for anything that fits on the dataset
    (training jobs AND the one-shot feature-selection pass), so they can never
    drift apart on dtype handling. Returns (frame, label_encoder)."""
    df = load_dataset(dataset_path)
    if target_column not in df.columns:
        raise ValueError(
            f"target column '{target_column}' not found in dataset (available: {list(df.columns)[:30]})"
        )
    df = df.dropna(subset=[target_column])
    if df.empty:
        raise ValueError(f"target column '{target_column}' has no non-null values")

    # bool/int feature columns count as numeric to pandas but trip modern
    # sklearn dtype validation: SimpleImputer refuses bool input outright,
    # and refuses a float fill_value (the pipeline's constant-0.0 stage)
    # on int64 input. Cast both to float64 — estimators convert to float
    # internally anyway, and float holds NaN, so nullable bool/Int64
    # columns with missing values cast cleanly too. The target column is
    # left alone so classification labels stay as-is.
    cast_cols = [
        c
        for c in df.columns
        if c != target_column
        and (pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c]))
    ]
    if cast_cols:
        df = df.copy()
        df[cast_cols] = df[cast_cols].astype("float64")
    if pd.api.types.is_bool_dtype(df[target_column]):
        df = df.copy()
        df[target_column] = df[target_column].astype("int64")

    # ±inf (divide-by-zero artifacts, sentinel values) fails estimator
    # fit with "Input contains infinity" — treat as missing so the
    # pipeline's imputers handle it like any other NaN.
    float_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    if float_cols and np.isinf(df[float_cols].to_numpy()).any():
        df = df.copy()
        df[float_cols] = df[float_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[target_column])
        if df.empty:
            raise ValueError(f"target column '{target_column}' has no finite values")

    label_encoder = None
    y_full = df[target_column]
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y_full):
        label_encoder = LabelEncoder()
        df = df.copy()
        df[target_column] = label_encoder.fit_transform(y_full.astype(str))
    elif task_type == "regression" and not pd.api.types.is_numeric_dtype(y_full):
        # numbers frequently arrive as strings ("1,234", "$50.00", " 3.5 ");
        # strip common formatting and coerce rather than crashing the fit
        # with an opaque sklearn dtype error. Rows that still don't parse
        # are dropped; an entirely unparseable target is a clear error.
        coerced = pd.to_numeric(
            y_full.astype(str).str.strip().str.replace(r"[,$€£%]", "", regex=True), errors="coerce"
        )
        if coerced.notna().sum() == 0:
            raise ValueError(
                f"target column '{target_column}' is non-numeric and could not be parsed as numbers; "
                "a regression target must be numeric — pick a different column or task type"
            )
        df = df.copy()
        df[target_column] = coerced
        df = df.dropna(subset=[target_column])

    return df, label_encoder


_MIN_FEATURES_FOR_RFE = 5


def select_features(
    dataset_path: str,
    target_column: str,
    task_type: str,
    time_column: Optional[str],
    preprocess_steps: Optional[list[dict[str, Any]]],
    metric: Optional[str],
) -> dict[str, Any]:
    """One-shot recursive feature elimination with a very basic linear model
    (LogisticRegression / Ridge), run BEFORE candidates are dispatched; the
    selected subset is then shared by every candidate so all models train on
    the same feature space (see 2026-07-06-eda-drops-and-rfe-design.md,
    revision). Fits on the deterministic training split only — _split with
    the same arguments reproduces exactly the split each job will use, so
    selection never sees any job's held-out rows. Never raises: returns
    enabled=False with an explanatory note instead, and candidates fall back
    to training on all features."""
    info: dict[str, Any] = {
        "enabled": False,
        "basic_model": None,
        "n_features_total": None,
        "n_features_selected": None,
        "selected_features": [],
        "note": None,
    }
    try:
        df, _ = _load_training_frame(dataset_path, target_column, task_type)
        X_train, _, y_train, _ = _split(df, target_column, task_type, time_column)
        info["n_features_total"] = int(X_train.shape[1])

        if X_train.shape[1] < _MIN_FEATURES_FOR_RFE:
            info["note"] = (
                f"feature selection skipped: only {X_train.shape[1]} feature column(s) — too few to "
                "meaningfully eliminate"
            )
            return info
        splitter = _tuning_splitter(task_type, y_train, time_column)
        if splitter is None:
            info["note"] = "feature selection skipped: not enough samples per class/fold for its internal CV"
            return info
        _, scoring, _ = _tuning_scoring(task_type, metric, y_train)

        import sklearn.linear_model as lm

        if task_type == "classification":
            basic, info["basic_model"] = lm.LogisticRegression(max_iter=1000), "LogisticRegression"
        else:
            basic, info["basic_model"] = lm.Ridge(), "Ridge"

        prep = _build_preprocessor(preprocess_steps or [], X_train)
        pipe = Pipeline(
            [
                ("prep", prep),
                ("rfe", RFECV(estimator=basic, step=0.2, cv=splitter, scoring=scoring, min_features_to_select=1)),
            ]
        )
        pipe.fit(X_train, y_train)

        # preprocessor output names map 1:1 to raw column names (one-hot runs
        # upstream in apply_feature_plan; verbose_feature_names_out=False), so
        # the selected names are directly usable as job column subsets.
        names = [str(n) for n in pipe.named_steps["prep"].get_feature_names_out()]
        support = pipe.named_steps["rfe"].support_
        selected = [name for name, keep in zip(names, support) if keep]
        info["enabled"] = True
        info["n_features_total"] = len(names)
        info["n_features_selected"] = len(selected)
        info["selected_features"] = selected
    except Exception as exc:  # noqa: BLE001 - selection is an optimization, never a reason to block training
        info["note"] = f"feature selection skipped after error: {exc}"
    return info


def _run_job(
    run_id: str,
    dataset_path: str,
    target_column: str,
    task_type: str,
    library: str,
    estimator_name: str,
    hyperparams: dict[str, Any],
    time_column: Optional[str],
    preprocess_steps: list[dict[str, Any]],
    cv_enabled: bool,
    cv_folds: Optional[int],
    resampling_enabled: bool,
    resampling_method: str,
    tuning_enabled: bool,
    tuning_trials: Optional[int],
    tuning_metric: Optional[str],
    selected_features: Optional[list[str]] = None,
    feature_selection_note: Optional[str] = None,
) -> None:
    start = time.monotonic()
    _registry[run_id]["status"] = "running"
    try:
        df, label_encoder = _load_training_frame(dataset_path, target_column, task_type)

        if task_type == "classification":
            n_classes = int(df[target_column].nunique())
            hyperparams, compat_note = _fix_target_incompatibilities(
                estimator_name, hyperparams, task_type, n_classes
            )
            if compat_note:
                _registry[run_id]["hyperparam_note"] = compat_note

        X_train, X_test, y_train, y_test = _split(df, target_column, task_type, time_column)

        # feature selection ran ONCE upstream (select_features, basic linear
        # model) — every candidate trains on the same shared subset, applied
        # here right after the split. Columns missing from this job's frame
        # are ignored defensively rather than raising.
        n_features_before = int(X_train.shape[1])
        fs_applied: Optional[list[str]] = None
        if selected_features:
            keep = [c for c in X_train.columns if c in set(selected_features)]
            if keep:
                X_train, X_test = X_train[keep], X_test[keep]
                fs_applied = keep

        # statistical preprocessing (impute/scale/target-encode) lives INSIDE
        # the fitted pipeline so it is fit on the training fold only — and
        # re-fit per fold by cross_validate. Columns it doesn't cover are
        # zero-filled numerics; non-numeric leftovers are dropped (the same
        # numeric-only guard as before, now expressed in the transformer).
        resampler, resampling_applied, resampling_note = (None, None, None)
        if resampling_enabled and resampling_method != "none" and task_type == "classification":
            resampler, resampling_applied, resampling_note = _build_resampler(resampling_method, y_train)

        def _make_pipeline(params: dict[str, Any]):
            preprocessor = _build_preprocessor(preprocess_steps or [], X_train)
            estimator = _build_estimator(library, estimator_name, params)
            if resampler is not None:
                from imblearn.pipeline import Pipeline as ImbPipeline

                # imblearn's Pipeline only resamples during .fit() — .predict()
                # and cross_validate()'s held-out scoring pass through
                # untouched, which is exactly what keeps synthetic/duplicated
                # rows out of the test fold (no leakage across the train/test
                # or CV boundary).
                return ImbPipeline([("prep", preprocessor), ("resample", clone(resampler)), ("model", estimator)])
            return Pipeline([("prep", preprocessor), ("model", estimator)])

        if tuning_enabled:
            cfg = _runtime_config()
            try:
                final_params, tuning_info = _tune_pipeline(
                    run_id,
                    _make_pipeline,
                    hyperparams,
                    library,
                    estimator_name,
                    X_train,
                    y_train,
                    task_type,
                    tuning_metric,
                    time_column,
                    tuning_trials if tuning_trials is not None else cfg["tuning_trials"],
                    cfg["hyperparam_search_budget_seconds"],
                )
            except Exception as exc:  # noqa: BLE001 - tuning is an optimization, never a reason to fail the run
                final_params = dict(hyperparams)
                tuning_info = {
                    "enabled": False,
                    "trials_total": 0,
                    "trials_done": 0,
                    "metric": None,
                    "lower_is_better": False,
                    "best_params": {},
                    "history": [],
                    "note": f"tuning skipped after error, trained with proposed hyperparameters: {exc}",
                }
                _registry[run_id]["tuning"] = dict(tuning_info)
        else:
            final_params = dict(hyperparams)
            tuning_info = {
                "enabled": False,
                "trials_total": 0,
                "trials_done": 0,
                "metric": None,
                "lower_is_better": False,
                "best_params": {},
                "history": [],
                "note": "tuning disabled for this run",
            }
            _registry[run_id]["tuning"] = dict(tuning_info)

        fit_estimator = _make_pipeline(final_params)

        if cv_enabled:
            cv_folds_requested = cv_folds if cv_folds is not None else _runtime_config()["cv_folds"]
            cv_result = _cross_validate(fit_estimator, X_train, y_train, task_type, time_column, cv_folds_requested)
        else:
            cv_result = {"folds": 0, "metrics": {}, "note": "cross-validation disabled for this run"}

        fit_estimator.fit(X_train, y_train)
        y_pred = fit_estimator.predict(X_test)
        y_proba = None
        if task_type == "classification" and hasattr(fit_estimator, "predict_proba"):
            proba = fit_estimator.predict_proba(X_test)
            if proba.shape[1] == 2:
                y_proba = proba[:, 1]

        metrics = _evaluate(task_type, y_test, y_pred, y_proba)
        fitted_prep = fit_estimator.named_steps["prep"]
        feature_names = [str(name) for name in fitted_prep.get_feature_names_out()]

        fs_info: dict[str, Any] = {
            "enabled": fs_applied is not None,
            "n_features_total": n_features_before,
            "n_features_selected": len(fs_applied) if fs_applied is not None else None,
            "selected_features": fs_applied or [],
            "note": feature_selection_note,
        }

        # X_train is already the selected subset, so the fitted preprocessor's
        # output names ARE the selected feature space
        feature_importance = _feature_importance(fit_estimator.named_steps["model"], feature_names)

        # feature_columns/feature_types describe the RAW model inputs (the
        # pipeline transforms them itself) — this is what the predict form
        # and predict_one build a row from.
        feature_types = {
            col: ("numeric" if pd.api.types.is_numeric_dtype(X_train[col]) else "text") for col in X_train.columns
        }
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = ARTIFACT_DIR / f"{run_id}.joblib"
        joblib.dump(
            {
                "estimator": fit_estimator,
                "label_encoder": label_encoder,
                "feature_columns": list(X_train.columns),
                "feature_types": feature_types,
            },
            model_path,
        )

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
            tuning=tuning_info,
            feature_selection=fs_info,
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
    preprocess_steps: Optional[list[dict[str, Any]]] = None,
    cv_enabled: bool = True,
    cv_folds: Optional[int] = None,
    resampling_enabled: bool = False,
    resampling_method: str = "none",
    tuning_enabled: bool = True,
    tuning_trials: Optional[int] = None,
    tuning_metric: Optional[str] = None,
    selected_features: Optional[list[str]] = None,
    feature_selection_note: Optional[str] = None,
) -> str:
    """Dispatch an async training job for one candidate model and return its
    run_id IMMEDIATELY (does not block on training completion). Use
    poll_training_job(run_id) to check status. `library` is one of "sklearn",
    "xgboost", "lightgbm"; `estimator` is the class name within that library
    (e.g. "RandomForestClassifier"). Pass `time_column` for time-series data
    so the train/test split is chronological rather than random.
    `preprocess_steps` are the feature plan's statistical steps (impute with
    mean/median, scale, target encode) — they are fit on the training fold
    only, inside the model pipeline, so no test-fold statistic or label can
    leak into training. `cv_enabled`/`cv_folds` are the user's choice at the
    confirm checkpoint (folds defaults to config/runtime.yaml's cv_folds when
    not given); set cv_enabled=False to skip k-fold cross-validation entirely
    for this run. `resampling_enabled`/`resampling_method` ("smote" |
    "random_oversample" | "random_undersample") are the user's class-balancing
    choice from the feature-approval checkpoint — applied to the training fold
    only (classification tasks only; ignored for regression), never to the
    held-out test/CV fold, so it can't leak. `tuning_enabled` runs Optuna
    hyperparameter search per candidate (trials/timeout from
    config/runtime.yaml unless `tuning_trials` overrides; `tuning_metric`
    should be the task spec's metric): the proposed `hyperparams` are scored
    as the baseline trial 0 and the best configuration wins, with per-trial
    progress visible via poll_training_job's `tuning` field.
    `selected_features` is the shared feature subset chosen by a single
    upstream recursive-feature-elimination pass with a basic linear model
    (select_features) — when given, this job trains on exactly those columns
    so every candidate sees the same feature space; when None, all features
    are used. `feature_selection_note` carries the upstream pass's skip
    explanation, if any; both are echoed via poll_training_job's
    `feature_selection` field.
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
        "tuning": {
            "enabled": False,
            "trials_total": 0,
            "trials_done": 0,
            "metric": None,
            "lower_is_better": False,
            "best_params": {},
            "history": [],
            "note": None,
        },
        "feature_selection": {
            "enabled": False,
            "n_features_total": None,
            "n_features_selected": None,
            "selected_features": [],
            "note": None,
        },
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
        preprocess_steps or [],
        cv_enabled,
        cv_folds,
        resampling_enabled,
        resampling_method,
        tuning_enabled,
        tuning_trials,
        tuning_metric,
        selected_features,
        feature_selection_note,
    )
    _futures[run_id] = future
    return run_id


@tool
def poll_training_job(run_id: str) -> dict[str, Any]:
    """Return the current status snapshot for a previously dispatched training
    run_id: {run_id, candidate_name, status, metrics, error, model_path,
    feature_importance, cv_folds, cv_metrics, cv_note, resampling_applied,
    resampling_note, tuning}. status is one of "pending", "running",
    "succeeded", "failed". tuning is {enabled, trials_total, trials_done,
    metric, lower_is_better, best_params, history: [{trial, score,
    best_score}], note} and updates live while hyperparameter search runs
    (trial 0 is the proposed baseline). feature_importance is a best-effort ranked list (may be empty
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
    """Raw input columns (and their numeric/text kind) the saved pipeline
    expects — lets the frontend's 'test the model' tab build an input form
    without hardcoding anything."""
    bundle = joblib.load(model_path)
    return {
        "feature_columns": bundle["feature_columns"],
        "feature_types": bundle.get("feature_types", {}),
    }


def predict_one(model_path: str, values: dict[str, Any]) -> dict[str, Any]:
    """Score a single user-supplied row of RAW feature values (numbers as
    numbers, categories as strings) against a saved model bundle — the
    pipeline inside the bundle applies its own preprocessing. Missing values
    become NaN and are handled by the pipeline's imputers/encoders."""
    bundle = joblib.load(model_path)
    estimator = bundle["estimator"]
    feature_columns = bundle["feature_columns"]
    label_encoder = bundle.get("label_encoder")

    row = pd.DataFrame([{col: values.get(col, np.nan) for col in feature_columns}])
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
