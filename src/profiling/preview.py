"""Deterministic (non-LLM) helpers for the Dataset Preview "Data" tab.

Unlike src/profiling/profile.py, these functions operate on and return
row-level data — that's fine here because they serve the human-facing UI
via src/api/server.py's new dataset endpoints, never an LLM prompt.
CLAUDE.md's "raw data never enters an LLM context window" rule is about the
LLM boundary specifically; nothing in this module is wired into any agent,
tool, or prompt path. See
docs/superpowers/specs/2026-07-06-dataset-preview-design.md.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.profiling.heuristics import iqr_outlier_mask

MAX_PAGE_SIZE = 200
MAX_OUTLIER_EXAMPLES = 20
CORRELATION_MAX_COLUMNS = 50
HISTOGRAM_BINS = 20

VALID_CORRELATION_METHODS = ("pearson", "spearman", "kendall", "mutual_info")
VALID_OUTLIER_METHODS = ("iqr", "zscore", "isolation_forest", "lof")


def paginate_rows(
    df: pd.DataFrame,
    page: int,
    page_size: int,
    sort_by: Optional[str] = None,
    sort_dir: str = "asc",
    search: Optional[str] = None,
) -> dict[str, Any]:
    """Server-side page of raw rows. duplicate_row_indices are indices
    within the returned page only (cheap to compute, no full-dataset scan
    needed for a UI highlight)."""
    if page_size < 1 or page_size > MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}")
    if page < 1:
        raise ValueError("page must be >= 1")

    working = df
    if search:
        needle = search.lower()
        mask = working.apply(lambda row: needle in " ".join(str(v) for v in row.values).lower(), axis=1)
        working = working[mask]

    if sort_by:
        if sort_by not in working.columns:
            raise ValueError(f"unknown column '{sort_by}'")
        working = working.sort_values(by=sort_by, ascending=(sort_dir != "desc"), kind="mergesort")

    total_count = len(working)
    start = (page - 1) * page_size
    page_df = working.iloc[start : start + page_size]

    duplicate_mask = page_df.duplicated(keep=False)
    duplicate_row_indices = [int(i) for i in page_df.index[duplicate_mask]]

    display_df = page_df.reset_index().rename(columns={"index": "_row_index"})
    rows = display_df.to_dict(orient="records")
    # Convert NaN to None
    rows = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in rows]

    return {
        "rows": rows,
        "total_count": int(total_count),
        "page": page,
        "page_size": page_size,
        "duplicate_row_indices": duplicate_row_indices,
    }


def column_detail(
    df: pd.DataFrame,
    column: str,
    target_column: Optional[str] = None,
) -> dict[str, Any]:
    if column not in df.columns:
        raise ValueError(f"unknown column '{column}'")
    series = df[column]
    is_numeric = pd.api.types.is_numeric_dtype(series)
    non_null = series.dropna()

    result: dict[str, Any] = {"column": column, "dtype": str(series.dtype), "is_numeric": is_numeric}

    if is_numeric:
        if len(non_null) > 0:
            counts, edges = np.histogram(non_null, bins=HISTOGRAM_BINS)
            result["histogram"] = {"counts": [int(c) for c in counts], "bin_edges": [float(e) for e in edges]}
            result["stats"] = {
                "mean": float(non_null.mean()),
                "median": float(non_null.median()),
                "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "p25": float(non_null.quantile(0.25)),
                "p75": float(non_null.quantile(0.75)),
                "skew": float(non_null.skew()) if len(non_null) > 2 else 0.0,
                "kurtosis": float(non_null.kurt()) if len(non_null) > 3 else 0.0,
            }
            if target_column and target_column in df.columns and target_column != column:
                target = df[target_column]
                if pd.api.types.is_numeric_dtype(target):
                    paired = pd.concat([series, target], axis=1).dropna()
                    if len(paired) > 1:
                        result["correlation_with_target"] = float(paired[column].corr(paired[target_column]))
        # else: numeric but entirely NaN — leave result with just column/dtype/is_numeric;
        # no histogram/stats/top_values, so a consumer checking is_numeric gets a
        # consistent (if data-less) shape rather than a mismatched one
    else:
        counts = non_null.astype(str).value_counts()
        result["top_values"] = {str(k): int(v) for k, v in counts.head(10).items()}
        result["rare_values"] = {str(k): int(v) for k, v in counts.tail(10).items()} if len(counts) > 10 else {}
        sample_size = min(5, len(non_null))
        result["random_samples"] = (
            [str(v) for v in non_null.sample(sample_size, random_state=0)] if sample_size else []
        )

    return result


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> dict[str, Any]:
    if method not in VALID_CORRELATION_METHODS:
        raise ValueError(f"unknown correlation method '{method}'")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    truncated = False
    if len(numeric_cols) > CORRELATION_MAX_COLUMNS:
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = list(variances.head(CORRELATION_MAX_COLUMNS).index)
        truncated = True

    if len(numeric_cols) < 2:
        return {"method": method, "columns": numeric_cols, "matrix": [], "truncated": truncated}

    if method == "mutual_info":
        from sklearn.feature_selection import mutual_info_regression

        subset = df[numeric_cols].dropna()
        n = len(numeric_cols)
        matrix = [[0.0] * n for _ in range(n)]
        if len(subset) > 3:
            for i, col_a in enumerate(numeric_cols):
                other_cols = [c for c in numeric_cols if c != col_a]
                mi = mutual_info_regression(subset[other_cols], subset[col_a], random_state=0)
                for value, col_b in zip(mi, other_cols):
                    matrix[i][numeric_cols.index(col_b)] = float(value)
        for i in range(n):
            matrix[i][i] = 1.0
    else:
        corr = df[numeric_cols].corr(method=method).fillna(0.0)
        matrix = [[float(corr.loc[a, b]) for b in numeric_cols] for a in numeric_cols]

    return {"method": method, "columns": numeric_cols, "matrix": matrix, "truncated": truncated}


def missing_value_matrix(df: pd.DataFrame) -> dict[str, Any]:
    null_counts = df.isna().sum()
    columns_with_nulls = [c for c in df.columns if null_counts[c] > 0]

    per_column = [
        {
            "column": c,
            "null_count": int(null_counts[c]),
            "null_rate": float(null_counts[c] / len(df)) if len(df) else 0.0,
        }
        for c in df.columns
    ]

    if len(columns_with_nulls) >= 2:
        nullness_corr = df[columns_with_nulls].isna().corr().fillna(0.0)
        correlation = {
            "columns": columns_with_nulls,
            "matrix": [[float(nullness_corr.loc[a, b]) for b in columns_with_nulls] for a in columns_with_nulls],
        }
    else:
        correlation = {"columns": columns_with_nulls, "matrix": []}

    return {"per_column": per_column, "missing_correlation": correlation}


def detect_outliers(df: pd.DataFrame, method: str = "iqr") -> dict[str, Any]:
    if method not in VALID_OUTLIER_METHODS:
        raise ValueError(f"unknown outlier method '{method}'")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return {"method": method, "outlier_count": 0, "affected_columns": [], "example_row_indices": []}

    if method in ("iqr", "zscore"):
        outlier_mask = pd.Series(False, index=df.index)
        affected_columns: list[str] = []
        for col in numeric_cols:
            series = df[col]
            if method == "iqr":
                col_mask = iqr_outlier_mask(series)
            else:
                non_null = series.dropna()
                std = non_null.std()
                if len(non_null) < 2 or std == 0:
                    continue
                col_mask = ((series - non_null.mean()) / std).abs() > 3
            col_mask = col_mask.fillna(False)
            if col_mask.any():
                affected_columns.append(col)
            outlier_mask |= col_mask
    else:
        subset = df[numeric_cols].apply(lambda c: c.fillna(c.mean()))
        if len(subset) < 2:
            return {"method": method, "outlier_count": 0, "affected_columns": [], "example_row_indices": []}
        if method == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            detector = IsolationForest(random_state=0, contamination="auto")
        else:
            from sklearn.neighbors import LocalOutlierFactor

            detector = LocalOutlierFactor(novelty=False, n_neighbors=min(20, len(subset) - 1))

        predictions = detector.fit_predict(subset)
        outlier_mask = pd.Series(predictions == -1, index=df.index)
        affected_columns = numeric_cols

    outlier_indices = list(df.index[outlier_mask])
    return {
        "method": method,
        "outlier_count": len(outlier_indices),
        "affected_columns": affected_columns,
        "example_row_indices": [int(i) for i in outlier_indices[:MAX_OUTLIER_EXAMPLES]],
    }
