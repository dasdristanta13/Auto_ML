"""Shared deterministic heuristics used by both the post-hoc auto-insights
generator (src/insights/auto_insights.py) and the pre-training EDA module
(src/profiling/eda.py) — kept in one place so the two can never drift apart
on what counts as "looks like an identifier" or "imbalanced"."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

ID_NAME_HINTS = ("id", "uuid", "guid", "key", "index")
IMBALANCE_THRESHOLD = 0.15
TARGET_CARDINALITY_RATIO_THRESHOLD = 0.5


def iqr_outlier_mask(series: pd.Series) -> pd.Series:
    """Boolean mask (aligned to series.index) of IQR-fence outliers. Shared
    by src/profiling/eda.py's outlier-rate check and src/profiling/preview.py's
    IQR outlier detector so both agree on what counts as an outlier."""
    non_null = series.dropna()
    if len(non_null) < 5:
        return series.notna() & False
    q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return series.notna() & False
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def looks_like_identifier(name: str, dtype: str, n_unique: int, row_count: int) -> bool:
    """Continuous numeric columns (floats: amounts, measurements) are
    naturally near-unique — that's not suspicious. Only flag integer/object
    columns, and only at a ratio strict enough that it's very unlikely to be
    a legitimate high-cardinality feature rather than an identifier."""
    if "float" in dtype.lower():
        return False
    ratio = n_unique / row_count
    if any(hint in name.lower() for hint in ID_NAME_HINTS):
        return ratio > 0.5
    return ratio > 0.98


def target_too_high_cardinality_for_classification(n_unique: int, row_count: int) -> bool:
    """True when a classification target has so many distinct values that,
    by pigeonhole, more than half the classes can only ever have a single
    example — making a stratified split, cross-validation, or even a plain
    holdout split structurally unable to generalize (and, for XGBoost
    specifically, guaranteed to fail its contiguous-label-range validation
    the moment a random split leaves a gap in the training labels).

    Unlike looks_like_identifier(), this is not a feature-exclusion
    heuristic — it checks fitness as a TARGET, so it does not exempt float
    dtypes: a continuous value is exactly as invalid a classification label
    as a high-cardinality string/int one."""
    if row_count <= 0:
        return False
    return (n_unique / row_count) > TARGET_CARDINALITY_RATIO_THRESHOLD


def minority_ratio(target_column_profile: Optional[dict[str, Any]]) -> Optional[float]:
    """The minority-class share of a classification target, derived from its
    profile entry (categorical top_values, or a 0/1-encoded numeric column's
    mean-as-positive-rate). None if it can't be determined from the profile
    alone (e.g. no target column, or a non-binary/non-categorical target)."""
    if not target_column_profile:
        return None
    if "top_values" in target_column_profile and isinstance(target_column_profile["top_values"], dict):
        counts = target_column_profile["top_values"]
        total = sum(counts.values())
        return (min(counts.values()) / total) if total else None
    if "numeric_summary" in target_column_profile:
        mean = target_column_profile["numeric_summary"].get("mean")
        if mean is not None and 0 <= mean <= 1:
            return min(mean, 1 - mean)
    return None
