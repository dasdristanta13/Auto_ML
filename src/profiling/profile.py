"""Deterministic data profiling (non-LLM).

Produces the statistical summary that REPLACES raw data in the LLM context
(CLAUDE.md non-negotiable rule). Nothing here ever puts a full column of raw
values, or any value from a PII-flagged column, into the returned profile.

PII redaction runs first (rule #5) — profiling always operates on the
redacted frame for anything that could leak a raw value (sample rows, top
categorical values). The original frame is only used for shape-level
statistics (null rates, cardinality counts, correlation magnitudes) which do
not expose individual values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.pii.redact import redact_dataframe

WIDE_DATASET_COLUMN_THRESHOLD = 50
CORRELATION_CLUSTER_THRESHOLD = 0.8
MAX_TOP_CATEGORIES = 10
MAX_SAMPLE_ROWS = 5  # far below the 20-row tool cap; profiling needs even less


def _numeric_summary(series: pd.Series) -> dict[str, float]:
    described = series.describe()
    return {
        "mean": float(described.get("mean", np.nan)),
        "std": float(described.get("std", np.nan)),
        "min": float(described.get("min", np.nan)),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.5)),
        "p75": float(series.quantile(0.75)),
        "max": float(described.get("max", np.nan)),
    }


def _categorical_summary(series: pd.Series, is_pii: bool) -> dict[str, Any]:
    if is_pii:
        return {"top_values": "[REDACTED]", "n_unique": int(series.nunique())}
    counts = series.value_counts().head(MAX_TOP_CATEGORIES)
    return {
        "top_values": {str(k): int(v) for k, v in counts.items()},
        "n_unique": int(series.nunique()),
    }


def _cluster_numeric_columns(corr: pd.DataFrame, threshold: float = CORRELATION_CLUSTER_THRESHOLD) -> list[list[str]]:
    """Greedy correlation clustering so wide numeric blocks summarize as groups
    rather than exhaustive per-column output (FR-6)."""
    remaining = list(corr.columns)
    clusters: list[list[str]] = []
    while remaining:
        anchor = remaining.pop(0)
        cluster = [anchor]
        for col in list(remaining):
            if abs(corr.loc[anchor, col]) >= threshold:
                cluster.append(col)
                remaining.remove(col)
        clusters.append(cluster)
    return clusters


def _column_level_profile(df: pd.DataFrame, redacted: pd.DataFrame, pii_columns: dict[str, Any]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for col in df.columns:
        is_pii = col in pii_columns
        col_profile: dict[str, Any] = {
            "dtype": str(df[col].dtype),
            "null_rate": float(df[col].isna().mean()),
            "n_unique": int(df[col].nunique(dropna=True)),
            "is_pii": is_pii,
        }
        if is_pii:
            col_profile["pii_type"] = pii_columns[col]["pii_type"]
        if pd.api.types.is_numeric_dtype(df[col]) and not is_pii:
            col_profile["numeric_summary"] = _numeric_summary(df[col].dropna())
        elif not is_pii:
            col_profile.update(_categorical_summary(redacted[col], is_pii=False))
        else:
            col_profile.update(_categorical_summary(redacted[col], is_pii=True))
        columns[col] = col_profile
    return columns


def _clustered_numeric_profile(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
    corr = df[numeric_cols].corr().fillna(0.0)
    clusters = _cluster_numeric_columns(corr)
    cluster_summaries = []
    for cluster in clusters:
        representative = cluster[0]
        cluster_summaries.append(
            {
                "member_columns": cluster,
                "size": len(cluster),
                "representative_column": representative,
                "representative_summary": _numeric_summary(df[representative].dropna()),
            }
        )
    return {"numeric_clusters": cluster_summaries, "n_clusters": len(clusters)}


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Returns a JSON-serializable profile safe to place directly in an LLM prompt."""
    redacted, pii_report = redact_dataframe(df)
    is_wide = len(df.columns) > WIDE_DATASET_COLUMN_THRESHOLD

    profile: dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "is_wide_dataset": is_wide,
        "pii_report": pii_report,
    }

    if is_wide:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in pii_report["columns"]]
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
        profile["numeric_summary"] = (
            _clustered_numeric_profile(df, numeric_cols) if numeric_cols else {"numeric_clusters": [], "n_clusters": 0}
        )
        # still give full detail for non-numeric columns (typically far fewer in wide tabular data)
        profile["columns"] = _column_level_profile(
            df[non_numeric_cols], redacted[non_numeric_cols], pii_report["columns"]
        )
        # every numeric column still gets a basic entry (no numeric_summary —
        # that's the cluster summary's job): the confirm endpoint validates
        # the target against profile["columns"] and the UI builds its target
        # picker from it, so omitting these made numeric targets unselectable.
        # Low-cardinality columns (plausible classification targets) also get
        # top_values so imbalance detection (minority_ratio) keeps working.
        for col in numeric_cols:
            entry: dict[str, Any] = {
                "dtype": str(df[col].dtype),
                "null_rate": float(df[col].isna().mean()),
                "n_unique": int(df[col].nunique(dropna=True)),
                "is_pii": False,
            }
            if entry["n_unique"] <= MAX_TOP_CATEGORIES:
                counts = df[col].value_counts().head(MAX_TOP_CATEGORIES)
                entry["top_values"] = {str(k): int(v) for k, v in counts.items()}
            profile["columns"][col] = entry
    else:
        profile["columns"] = _column_level_profile(df, redacted, pii_report["columns"])
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in pii_report["columns"]]
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().fillna(0.0)
            # only surface strong pairs, not the full matrix, to keep the profile compact
            strong_pairs = []
            for i, a in enumerate(numeric_cols):
                for b in numeric_cols[i + 1 :]:
                    value = float(corr.loc[a, b])
                    if abs(value) >= 0.6:
                        strong_pairs.append({"a": a, "b": b, "correlation": round(value, 3)})
            profile["strong_correlations"] = strong_pairs

    profile["sample_rows"] = redacted.head(MAX_SAMPLE_ROWS).to_dict(orient="records")
    return profile
