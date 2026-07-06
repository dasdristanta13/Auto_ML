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

MAX_PAGE_SIZE = 200
MAX_OUTLIER_EXAMPLES = 20
CORRELATION_MAX_COLUMNS = 50
HISTOGRAM_BINS = 20
TEXT_MEAN_LENGTH_THRESHOLD = 30

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
    if page_size > MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be <= {MAX_PAGE_SIZE}")
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
