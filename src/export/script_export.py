"""Generates a standalone, human-readable Python script that reproduces the
training run: load data -> apply the confirmed feature plan -> split ->
train the winning candidate -> evaluate -> save. This is what PRODUCT.md's
report-view "export pipeline as code" action returns.

The script has no dependency on this repo — it only imports pandas/numpy/
scikit-learn (plus xgboost/lightgbm if that candidate won) so a data
scientist can take it and run it standalone (PRD 2.2 secondary persona).

This is a deterministic transcription of the exact same logic already
executed by src/graph/nodes.py::_apply_builtin_step and
src/training/dispatch.py — if those change, this stays in sync only if
kept in step; there is no shared code path (by design, so the exported
script is dependency-free).
"""

from __future__ import annotations

import textwrap
from typing import Any

_ESTIMATOR_IMPORTS = {
    ("sklearn", "LogisticRegression"): "from sklearn.linear_model import LogisticRegression",
    ("sklearn", "LinearRegression"): "from sklearn.linear_model import LinearRegression",
    ("sklearn", "Ridge"): "from sklearn.linear_model import Ridge",
    ("sklearn", "RandomForestClassifier"): "from sklearn.ensemble import RandomForestClassifier",
    ("sklearn", "RandomForestRegressor"): "from sklearn.ensemble import RandomForestRegressor",
    ("sklearn", "GradientBoostingClassifier"): "from sklearn.ensemble import GradientBoostingClassifier",
    ("sklearn", "GradientBoostingRegressor"): "from sklearn.ensemble import GradientBoostingRegressor",
    ("xgboost", "XGBClassifier"): "from xgboost import XGBClassifier",
    ("xgboost", "XGBRegressor"): "from xgboost import XGBRegressor",
    ("lightgbm", "LGBMClassifier"): "from lightgbm import LGBMClassifier",
    ("lightgbm", "LGBMRegressor"): "from lightgbm import LGBMRegressor",
}


def _step_code(step: dict[str, Any], index: int) -> str:
    op = step.get("op")
    columns = step.get("columns", [])
    params = step.get("params", {})
    rationale = step.get("rationale", "")
    comment = f"    # {rationale}\n" if rationale else ""

    if op == "impute":
        strategy = params.get("strategy", "mean")
        lines = []
        for col in columns:
            if strategy == "mean":
                lines.append(f'    df["{col}"] = df["{col}"].fillna(df["{col}"].mean())')
            elif strategy == "median":
                lines.append(f'    df["{col}"] = df["{col}"].fillna(df["{col}"].median())')
            elif strategy == "most_frequent":
                lines.append(f'    df["{col}"] = df["{col}"].fillna(df["{col}"].mode().iloc[0])')
            elif strategy == "constant":
                lines.append(f'    df["{col}"] = df["{col}"].fillna({params.get("fill_value", 0)!r})')
        return comment + "\n".join(lines)

    if op == "encode":
        method = params.get("method", "onehot")
        if method == "onehot":
            return comment + f"    df = pd.get_dummies(df, columns={columns!r}, dummy_na=False)"
        if method == "ordinal":
            return (
                comment
                + f'    _encoder_{index} = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)\n'
                + f"    df[{columns!r}] = _encoder_{index}.fit_transform(df[{columns!r}].astype(str))"
            )
        if method == "target":
            target_col = params.get("target_column", "TARGET_COLUMN")
            lines = [f'    df["{c}"] = df.groupby("{c}")["{target_col}"].transform("mean")' for c in columns]
            return comment + "\n".join(lines)

    if op == "scale":
        method = params.get("method", "standard")
        scaler_cls = {"standard": "StandardScaler", "minmax": "MinMaxScaler", "robust": "RobustScaler"}[method]
        return (
            comment
            + f"    _scaler_{index} = {scaler_cls}()\n"
            + f"    df[{columns!r}] = _scaler_{index}.fit_transform(df[{columns!r}])"
        )

    if op == "bin":
        n_bins = params.get("n_bins", 5)
        lines = [f'    df["{c}"] = pd.cut(df["{c}"], bins={n_bins}, labels=False)' for c in columns]
        return comment + "\n".join(lines)

    if op == "datetime_decompose":
        lines = []
        for c in columns:
            lines.append(f'    _parsed = pd.to_datetime(df["{c}"], errors="coerce")')
            lines.append(f'    df["{c}_year"] = _parsed.dt.year')
            lines.append(f'    df["{c}_month"] = _parsed.dt.month')
            lines.append(f'    df["{c}_day"] = _parsed.dt.day')
            lines.append(f'    df["{c}_dayofweek"] = _parsed.dt.dayofweek')
            lines.append(f'    df = df.drop(columns=["{c}"])')
        return comment + "\n".join(lines)

    if op == "drop":
        return comment + f"    df = df.drop(columns=[c for c in {columns!r} if c in df.columns])"

    if op == "custom_code":
        code = step.get("code", "")
        # the LLM-authored function is inlined verbatim; it already passed
        # src/sandbox/validate.py's AST whitelist and a sandboxed dry-run
        # before this run was allowed to use it. It must be nested (indented
        # one level) so it becomes a local function inside build_features
        # rather than a top-level def that would dedent out of — and
        # truncate — build_features's own body.
        renamed = code.replace("def transform(", f"def _custom_transform_{index}(", 1)
        nested = textwrap.indent(renamed.rstrip("\n"), "    ")
        return f"{comment}{nested}\n    df = _custom_transform_{index}(df)"

    return f"    # unrecognized step op '{op}' — skipped"


def generate_training_script(state: dict[str, Any]) -> str:
    task_spec = state.get("task_spec", {}) or {}
    target_column = task_spec.get("target_column", "TARGET_COLUMN")
    task_type = task_spec.get("task_type", "classification")
    metric = task_spec.get("metric")

    feature_plan = state.get("feature_plan", {}) or {}
    steps = feature_plan.get("steps", [])

    best_model = state.get("best_model", {}) or {}
    candidate_name = best_model.get("candidate_name", "candidate")
    candidate_models = state.get("candidate_models", []) or []
    winning_candidate = next((c for c in candidate_models if c.get("name") == candidate_name), {})
    library = winning_candidate.get("library", "sklearn")
    estimator_name = winning_candidate.get("estimator", "RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor")
    # prefer the Optuna-tuned params the winning model was actually fit with;
    # fall back to the LLM-proposed hyperparams when tuning was off/skipped
    hyperparams = (best_model.get("tuning") or {}).get("best_params") or winning_candidate.get("hyperparams", {})

    estimator_import = _ESTIMATOR_IMPORTS.get((library, estimator_name), "# unknown estimator — add the right import here")
    hyperparams_repr = ", ".join(f"{k}={v!r}" for k, v in hyperparams.items())

    uses_encoder = any(s.get("op") == "encode" and s.get("params", {}).get("method") == "ordinal" for s in steps)
    uses_scaler = any(s.get("op") == "scale" for s in steps)

    step_blocks = "\n\n".join(_step_code(step, i) for i, step in enumerate(steps)) or "    pass  # no feature transformations were planned for this run"

    is_classification = task_type == "classification"
    metrics_import = (
        "from sklearn.metrics import accuracy_score, f1_score, roc_auc_score"
        if is_classification
        else "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score"
    )
    metrics_block = (
        '    print("accuracy:", accuracy_score(y_test, y_pred))\n'
        '    print("f1:", f1_score(y_test, y_pred, average="weighted"))'
        if is_classification
        else '    print("rmse:", mean_squared_error(y_test, y_pred) ** 0.5)\n'
        '    print("mae:", mean_absolute_error(y_test, y_pred))\n'
        '    print("r2:", r2_score(y_test, y_pred))'
    )

    return f'''"""
Auto-generated training script — reproduces run {state.get("run_id", "")}.

Use case: {state.get("use_case_description", "")}
Task type: {task_type} | Target column: {target_column} | Metric: {metric}
Winning candidate: {candidate_name} ({library}.{estimator_name})

This script has no dependency on the Agentic AutoML platform — only pandas,
numpy, and scikit-learn (plus xgboost/lightgbm if that's what won). Point
DATASET_PATH at your CSV and run it.

Caveats carried over from the report: target-leakage detection is heuristic
and may have missed cases — review the leakage flags in the report before
trusting this model in production.

Simplification note: this script applies all feature transforms before the
train/test split for readability. The platform itself fits statistical
transforms (imputation means, scalers, target encodings) on the training
fold only, so metrics printed by this script can differ slightly (and target
encoding here will look optimistic). Prefer the platform's reported metrics.
"""

import pandas as pd
import numpy as np
{"from sklearn.preprocessing import OrdinalEncoder" if uses_encoder else ""}
{"from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler" if uses_scaler else ""}
from sklearn.model_selection import train_test_split
{metrics_import}
{estimator_import}
import joblib

DATASET_PATH = "your_dataset.csv"  # point this at your copy of the dataset
TARGET_COLUMN = "{target_column}"
TASK_TYPE = "{task_type}"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduces the confirmed feature engineering plan, in order."""
{step_blocks}
    return df


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=[TARGET_COLUMN])
    df = build_features(df)

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    stratify = df[TARGET_COLUMN] if TASK_TYPE == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df[TARGET_COLUMN], test_size=0.2, random_state=0, stratify=stratify
    )

    # this reference pipeline trains on numeric columns only — extend
    # build_features() with an encoding step if you need categoricals too
    X_train = X_train.select_dtypes(include="number").fillna(0)
    X_test = X_test[X_train.columns].fillna(0)

    model = {estimator_name}({hyperparams_repr})
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

{metrics_block}

    joblib.dump({{"estimator": model, "feature_columns": list(X_train.columns)}}, "model.joblib")
    print("Saved model.joblib")


if __name__ == "__main__":
    main()
'''
