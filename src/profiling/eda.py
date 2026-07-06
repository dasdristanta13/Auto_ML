"""Deterministic exploratory data analysis (EDA).

Runs after leakage_check, before the LLM feature_engineering node. Produces
concrete, rule-based feature-step suggestions (impute/encode/scale/drop/
datetime_decompose) with a plain-language rationale for each, plus a
resampling suggestion for imbalanced classification targets — all computed
directly from the dataset (never an LLM call, so it's instant, free, and
fully auditable). These suggestions are:
  1. fed to the LLM feature_engineering node as strong prior context, and
  2. shown to the user for approval before anything is applied (CLAUDE.md:
     never silently auto-select on an ambiguous/consequential decision).

Reads the raw dataframe (like leakage_check_node already does) only to
compute aggregate statistics (skew, IQR outlier rate, cardinality) — nothing
raw is ever put in the returned dict, preserving the non-negotiable "no raw
data in an LLM context" rule.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.insights.auto_insights import profile_insights
from src.profiling.heuristics import IMBALANCE_THRESHOLD, iqr_outlier_mask, looks_like_identifier, minority_ratio

_MIN_ROWS_FOR_IDENTIFIER_CHECK = 20
_HIGH_CARDINALITY_MIN_UNIQUE = 20
_HIGH_SKEW = 1.0
_OUTLIER_HEAVY_RATE = 0.02
_DATETIME_NAME_HINTS = ("date", "time", "timestamp", "_at", "_dt")


def _step(op: str, columns: list[str], params: dict[str, Any], rationale: str) -> dict[str, Any]:
    return {"op": op, "columns": columns, "params": params, "code": None, "rationale": rationale, "source": "eda"}


def _iqr_outlier_rate(series: pd.Series) -> float:
    if len(series) < 5:
        return 0.0
    return float(iqr_outlier_mask(series).mean())


def _looks_datetime(name: str, series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not any(hint in name.lower() for hint in _DATETIME_NAME_HINTS):
        return False
    sample = series.dropna().head(50)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.9


def _suggest_feature_steps(
    df: pd.DataFrame,
    profile: dict[str, Any],
    task_spec: dict[str, Any],
    leakage_flags: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    target_column = task_spec.get("target_column")
    time_column = task_spec.get("time_column")
    pii_columns = set((profile.get("pii_report", {}) or {}).get("columns", {}) or {})
    row_count = len(df)

    suggestions: list[dict[str, Any]] = []
    numeric_cols_for_scaling: list[str] = []
    any_outlier_heavy = False

    # high-severity leakage flags (near-perfect target correlation, categories
    # mapping ~1:1 to the target) become concrete drop suggestions the user
    # approves/rejects at the feature checkpoint — previously they were
    # display-only, so leaky columns reached training unless dropped manually.
    leakage_drop_cols: list[str] = []
    for flag in leakage_flags or []:
        col = flag.get("column")
        if (
            flag.get("severity") == "high"
            and col in df.columns
            and col not in (target_column, time_column)
            and col not in leakage_drop_cols
        ):
            leakage_drop_cols.append(col)
            suggestions.append(
                _step(
                    "drop", [col], {},
                    f"'{col}' is a suspected target-leakage column ({flag.get('reason')}). A model trained on "
                    "it will score deceptively well offline but won't generalize — dropping it is strongly "
                    "recommended. (Heuristic flag, not a guarantee.)",
                )
            )

    for col in df.columns:
        # the designated time_column must survive untouched: training's
        # chronological split sorts by it (src/training/dispatch._split), so
        # decomposing or dropping it would silently disable that split.
        if col == target_column or col == time_column or col in pii_columns:
            continue
        if col in leakage_drop_cols:
            continue  # already suggested as a drop; don't also impute/encode it
        series = df[col]
        dtype = str(series.dtype)
        n_unique = int(series.nunique(dropna=True))
        null_rate = float(series.isna().mean())
        is_numeric = pd.api.types.is_numeric_dtype(series)

        # datetime check comes BEFORE the identifier heuristic — a daily date
        # column is near-unique by nature and would otherwise be misread as an
        # identifier and dropped instead of decomposed.
        if not is_numeric and _looks_datetime(col, series):
            suggestions.append(
                _step(
                    "datetime_decompose", [col], {},
                    f"'{col}' looks like a timestamp; decomposing into year/month/day/day-of-week usually "
                    "carries more model-usable signal than the raw value.",
                )
            )
            continue

        if row_count > _MIN_ROWS_FOR_IDENTIFIER_CHECK and looks_like_identifier(col, dtype, n_unique, row_count):
            suggestions.append(
                _step("drop", [col], {}, f"'{col}' looks like an identifier (near-unique per row) rather than a predictive feature.")
            )
            continue  # don't also suggest imputing/encoding a column we're dropping

        if is_numeric:
            if null_rate > 0:
                non_null = series.dropna()
                skew = float(non_null.skew()) if len(non_null) > 2 else 0.0
                strategy = "median" if abs(skew) > _HIGH_SKEW else "mean"
                symmetry = "skewed" if strategy == "median" else "roughly symmetric"
                suggestions.append(
                    _step(
                        "impute", [col], {"strategy": strategy},
                        f"'{col}' is missing in {null_rate:.0%} of rows; {strategy} imputation chosen because "
                        f"the distribution is {symmetry}.",
                    )
                )
            if _iqr_outlier_rate(series.dropna()) > _OUTLIER_HEAVY_RATE:
                any_outlier_heavy = True
            numeric_cols_for_scaling.append(col)
        else:
            if null_rate > 0:
                suggestions.append(
                    _step(
                        "impute", [col], {"strategy": "most_frequent"},
                        f"'{col}' is missing in {null_rate:.0%} of rows; filled with the most frequent category.",
                    )
                )
            if n_unique > _HIGH_CARDINALITY_MIN_UNIQUE:
                method = "target" if target_column else "ordinal"
                params = {"method": method, **({"target_column": target_column} if method == "target" else {})}
                suggestions.append(
                    _step(
                        "encode", [col], params,
                        f"'{col}' has {n_unique} distinct categories; one-hot encoding would explode "
                        f"dimensionality, so {method} encoding is suggested instead.",
                    )
                )
            elif n_unique > 1:
                suggestions.append(
                    _step(
                        "encode", [col], {"method": "onehot"},
                        f"'{col}' has {n_unique} categories; one-hot encoding preserves each as a distinct "
                        "signal without exploding dimensionality.",
                    )
                )

    if numeric_cols_for_scaling:
        method = "robust" if any_outlier_heavy else "standard"
        rationale = "Numeric features differ in scale; scaling helps distance/gradient-based models converge properly."
        if method == "robust":
            rationale += " Robust scaling was chosen because some columns have a notable share of outliers."
        suggestions.append(_step("scale", numeric_cols_for_scaling, {"method": method}, rationale))

    return suggestions


def _resampling_suggestion(profile: dict[str, Any], task_spec: dict[str, Any]) -> dict[str, Any]:
    if task_spec.get("task_type") != "classification":
        return {"suggested": False, "method": "none", "reason": None, "minority_ratio": None}

    target = task_spec.get("target_column")
    target_info = (profile.get("columns") or {}).get(target) if target else None
    ratio = minority_ratio(target_info)

    if ratio is not None and ratio < IMBALANCE_THRESHOLD:
        return {
            "suggested": True,
            "method": "smote",
            "reason": (
                f"the minority class is only {ratio:.0%} of rows — training on the raw distribution tends to "
                "bias the model toward the majority class. SMOTE oversamples the minority class (synthetically, "
                "only within the training fold) to counter that."
            ),
            "minority_ratio": ratio,
        }
    return {"suggested": False, "method": "none", "reason": None, "minority_ratio": ratio}


_RFE_RECOMMEND_MIN_FEATURES = 15


def run_eda(
    df: pd.DataFrame,
    profile: dict[str, Any],
    task_spec: dict[str, Any],
    leakage_flags: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Returns {"insights": [...], "suggested_steps": [...], "resampling_suggestion": {...}}."""
    insights = profile_insights(profile, task_spec)

    target_column = task_spec.get("target_column")
    n_features = sum(1 for c in df.columns if c != target_column)
    if n_features >= _RFE_RECOMMEND_MIN_FEATURES:
        insights.append(
            {
                "id": "rfe_recommended",
                "category": "modeling",
                "tone": "info",
                "message": (
                    f"This dataset has {n_features} candidate features — enabling feature selection (RFE) at "
                    "the confirm step lets each model recursively eliminate weak features and often yields a "
                    "simpler, better-generalizing model (at the cost of longer training)."
                ),
            }
        )

    return {
        "insights": insights,
        "suggested_steps": _suggest_feature_steps(df, profile, task_spec, leakage_flags),
        "resampling_suggestion": _resampling_suggestion(profile, task_spec),
    }
