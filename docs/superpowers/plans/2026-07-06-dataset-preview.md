# Dataset Preview ("Data" Tab) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the "Data" tab of a dataset detail page (KPI cards, interactive row preview, column explorer, correlations, missing values, outliers) reachable from a new "Datasets" section in the sidebar, per `docs/superpowers/specs/2026-07-06-dataset-preview-design.md`.

**Architecture:** A "dataset" is a top-level run (no `source_run_id`) — no new persistent entity. New deterministic profiling functions in `src/profiling/preview.py` operate on a per-run in-memory DataFrame cache in `src/api/server.py`, exposed via new `/api/datasets` and `/api/runs/{id}/{preview,columns/{name},correlations,missing-values,outliers,dataset-summary}` endpoints. Frontend adds a Datasets list view and a dataset detail view (vanilla JS, no build step, matching the existing `frontend/app.js` pattern).

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn (`IsolationForest`, `LocalOutlierFactor`, `mutual_info_regression` — all already in `requirements.txt`), FastAPI, vanilla JS/CSS (no new frontend dependency).

## Global Constraints

- Type hints required on all public functions (CLAUDE.md conventions).
- Row-level API responses must enforce hard caps in code: page_size ≤ 200, outlier examples ≤ 20 (CLAUDE.md tools convention, applied here to UI-facing endpoints for consistency even though they're not LLM tools).
- All new `/api/*` endpoints require `Depends(require_session)`, matching every existing `/api/runs*` route.
- No raw dataset values may reach an LLM prompt — these new endpoints are UI-facing only; nothing here is wired into any agent/tool/prompt path.
- Follow existing repo conventions: one responsibility per new module, deterministic profiling stays non-LLM, tests live under `/tests` mirroring the source module name.
- Frontend stays vanilla JS/CSS with no build step, following `frontend/app.js`'s existing `$()`/`escapeHtml()`/`ICONS` conventions and `styles.css`'s existing CSS custom-property theme system.

---

## Task 1: Shared IQR outlier helper

**Files:**
- Modify: `src/profiling/heuristics.py`
- Modify: `src/profiling/eda.py:39-47` (replace `_iqr_outlier_rate` body)
- Test: `tests/test_heuristics_outliers.py` (new)

**Interfaces:**
- Produces: `iqr_outlier_mask(series: pd.Series) -> pd.Series` (boolean mask aligned to `series.index`) — consumed by Task 7 (`detect_outliers`) and by `eda.py`'s existing outlier-rate check.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_heuristics_outliers.py
"""iqr_outlier_mask is shared between src/profiling/eda.py and the new
src/profiling/preview.py so 'outlier' is defined identically in both
(src/profiling/heuristics.py module docstring)."""

from __future__ import annotations

import pandas as pd

from src.profiling.heuristics import iqr_outlier_mask


def test_flags_values_outside_iqr_fences():
    series = pd.Series([10, 11, 12, 13, 14, 15, 1000])
    mask = iqr_outlier_mask(series)
    assert mask.iloc[-1] is True or bool(mask.iloc[-1]) is True
    assert not mask.iloc[:-1].any()


def test_returns_all_false_for_short_series():
    series = pd.Series([1, 2, 3])
    mask = iqr_outlier_mask(series)
    assert not mask.any()
    assert len(mask) == 3


def test_returns_all_false_when_iqr_is_zero():
    series = pd.Series([5, 5, 5, 5, 5, 100])
    mask = iqr_outlier_mask(series)
    assert not mask.any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_heuristics_outliers.py -v`
Expected: FAIL with `ImportError: cannot import name 'iqr_outlier_mask'`

- [ ] **Step 3: Add `iqr_outlier_mask` to `heuristics.py`**

`src/profiling/heuristics.py` currently imports nothing beyond `typing` (it has no pandas dependency yet). Add `import pandas as pd` directly below the existing `from __future__ import annotations` / `from typing import ...` imports at the top of the file.

Then add this function anywhere among the other free functions:

```python
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
```

- [ ] **Step 4: Update `eda.py` to use the shared helper**

In `src/profiling/eda.py`, replace the `_iqr_outlier_rate` function (lines 39-47) with:

```python
def _iqr_outlier_rate(series: pd.Series) -> float:
    if len(series) < 5:
        return 0.0
    return float(iqr_outlier_mask(series).mean())
```

Add `iqr_outlier_mask` to the existing heuristics import at the top of `eda.py` (currently `from src.profiling.heuristics import IMBALANCE_THRESHOLD, looks_like_identifier, minority_ratio`), so it reads:

```python
from src.profiling.heuristics import IMBALANCE_THRESHOLD, iqr_outlier_mask, looks_like_identifier, minority_ratio
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_heuristics_outliers.py -v`
Expected: PASS (3 tests)

Run: `pytest tests/test_eda_and_resampling.py -v`
Expected: PASS (existing EDA tests unaffected — same outlier semantics, just refactored)

- [ ] **Step 6: Commit**

```bash
git add src/profiling/heuristics.py src/profiling/eda.py tests/test_heuristics_outliers.py
git commit -m "refactor: share IQR outlier detection between eda.py and the new preview module"
```

---

## Task 2: Dataset memory usage in `profile_dataset`

**Files:**
- Modify: `src/profiling/profile.py:112-117`
- Test: `tests/test_profile_quality.py` (extend)

**Interfaces:**
- Produces: `profile["memory_bytes"]: int` on every `profile_dataset()` result — consumed by Task 13 (`_run_summary`'s `profile_summary`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_profile_quality.py`:

```python
def test_profile_reports_memory_bytes():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    profile = profile_dataset(df)
    assert profile["memory_bytes"] > 0
    assert profile["memory_bytes"] == int(df.memory_usage(deep=True).sum())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_profile_quality.py::test_profile_reports_memory_bytes -v`
Expected: FAIL with `KeyError: 'memory_bytes'`

- [ ] **Step 3: Add the field**

In `src/profiling/profile.py`, change the `profile` dict initializer (lines 112-117):

```python
    profile: dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "is_wide_dataset": is_wide,
        "pii_report": pii_report,
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_profile_quality.py -v`
Expected: PASS (all tests in file)

- [ ] **Step 5: Commit**

```bash
git add src/profiling/profile.py tests/test_profile_quality.py
git commit -m "feat: report dataset memory usage in profile_dataset"
```

---

## Task 3: `preview.py` — `paginate_rows`

**Files:**
- Create: `src/profiling/preview.py`
- Test: `tests/test_preview.py` (new)

**Interfaces:**
- Produces: `paginate_rows(df: pd.DataFrame, page: int, page_size: int, sort_by: Optional[str] = None, sort_dir: str = "asc", search: Optional[str] = None) -> dict[str, Any]` returning `{"rows": list[dict], "total_count": int, "page": int, "page_size": int, "duplicate_row_indices": list[int]}`. Raises `ValueError` on `page_size > MAX_PAGE_SIZE`, `page < 1`, or an unknown `sort_by` column. Consumed by Task 9 (`GET /api/runs/{id}/preview`).
- Also produces module constants `MAX_PAGE_SIZE = 200`, `MAX_OUTLIER_EXAMPLES = 20`, `CORRELATION_MAX_COLUMNS = 50`, `HISTOGRAM_BINS = 20`, `VALID_CORRELATION_METHODS`, `VALID_OUTLIER_METHODS` used by later tasks in this file.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preview.py
"""src/profiling/preview.py: deterministic, non-LLM helpers for the Dataset
Preview 'Data' tab. Unlike src/profiling/profile.py this module returns
row-level data — that's fine because it serves the human-facing UI, not an
LLM prompt (CLAUDE.md's raw-data rule is about the LLM boundary only). See
docs/superpowers/specs/2026-07-06-dataset-preview-design.md."""

from __future__ import annotations

import pandas as pd
import pytest

from src.profiling.preview import MAX_PAGE_SIZE, paginate_rows


def _df():
    return pd.DataFrame({"a": [3, 1, 2, 1], "b": ["x", "y", "z", "y"]})


def test_paginate_returns_requested_page():
    result = paginate_rows(_df(), page=1, page_size=2)
    assert result["total_count"] == 4
    assert len(result["rows"]) == 2
    assert result["page"] == 1
    assert result["page_size"] == 2


def test_paginate_second_page():
    result = paginate_rows(_df(), page=2, page_size=2)
    assert len(result["rows"]) == 2


def test_paginate_sorts_ascending():
    result = paginate_rows(_df(), page=1, page_size=10, sort_by="a", sort_dir="asc")
    values = [row["a"] for row in result["rows"]]
    assert values == sorted(values)


def test_paginate_rejects_unknown_sort_column():
    with pytest.raises(ValueError):
        paginate_rows(_df(), page=1, page_size=10, sort_by="nope")


def test_paginate_rejects_oversized_page():
    with pytest.raises(ValueError):
        paginate_rows(_df(), page=1, page_size=MAX_PAGE_SIZE + 1)


def test_paginate_search_filters_rows():
    result = paginate_rows(_df(), page=1, page_size=10, search="x")
    assert result["total_count"] == 1
    assert result["rows"][0]["b"] == "x"


def test_paginate_flags_duplicate_rows_within_page():
    result = paginate_rows(_df(), page=1, page_size=10)
    # rows at original index 1 ("1","y") and 3 ("1","y") are duplicates
    assert set(result["duplicate_row_indices"]) == {1, 3}


def test_paginate_converts_nan_to_none():
    df = pd.DataFrame({"a": [1, None]})
    result = paginate_rows(df, page=1, page_size=10)
    assert result["rows"][1]["a"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preview.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.profiling.preview'`

- [ ] **Step 3: Create `src/profiling/preview.py`**

```python
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

    display_df = page_df.where(pd.notnull(page_df), None)
    rows = display_df.reset_index().rename(columns={"index": "_row_index"}).to_dict(orient="records")

    return {
        "rows": rows,
        "total_count": int(total_count),
        "page": page,
        "page_size": page_size,
        "duplicate_row_indices": duplicate_row_indices,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preview.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/profiling/preview.py tests/test_preview.py
git commit -m "feat: add paginate_rows for the Dataset Preview Data tab"
```

---

## Task 4: `preview.py` — `column_detail`

**Files:**
- Modify: `src/profiling/preview.py`
- Test: `tests/test_preview.py` (extend)

**Interfaces:**
- Produces: `column_detail(df: pd.DataFrame, column: str, target_column: Optional[str] = None) -> dict[str, Any]`. Raises `ValueError` on unknown column. Returns, for numeric columns: `{"column", "dtype", "is_numeric": True, "histogram": {"counts": [...], "bin_edges": [...]}, "stats": {mean, median, std, min, max, p25, p75, skew, kurtosis}, "correlation_with_target"?: float}`. For non-numeric columns: `{"column", "dtype", "is_numeric": False, "top_values": dict, "rare_values": dict, "random_samples": list}`. Consumed by Task 10 (`GET /api/runs/{id}/columns/{name}`) and the frontend Column Explorer (Task 19).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_preview.py`:

```python
from src.profiling.preview import column_detail


def test_column_detail_numeric_has_histogram_and_stats():
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0, 4.0, 5.0]})
    detail = column_detail(df, "amount")
    assert detail["is_numeric"] is True
    assert len(detail["histogram"]["counts"]) > 0
    assert detail["stats"]["mean"] == 3.0
    assert detail["stats"]["min"] == 1.0
    assert detail["stats"]["max"] == 5.0


def test_column_detail_categorical_has_top_values():
    df = pd.DataFrame({"plan": ["a", "a", "b", "c"]})
    detail = column_detail(df, "plan")
    assert detail["is_numeric"] is False
    assert detail["top_values"]["a"] == 2


def test_column_detail_computes_correlation_with_numeric_target():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    detail = column_detail(df, "x", target_column="y")
    assert detail["correlation_with_target"] == pytest.approx(1.0)


def test_column_detail_rejects_unknown_column():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        column_detail(df, "nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preview.py -v -k column_detail`
Expected: FAIL with `ImportError: cannot import name 'column_detail'`

- [ ] **Step 3: Add `column_detail` to `preview.py`**

Append to `src/profiling/preview.py`:

```python
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

    if is_numeric and len(non_null) > 0:
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
    else:
        counts = non_null.astype(str).value_counts()
        result["top_values"] = {str(k): int(v) for k, v in counts.head(10).items()}
        result["rare_values"] = {str(k): int(v) for k, v in counts.tail(10).items()} if len(counts) > 10 else {}
        sample_size = min(5, len(non_null))
        result["random_samples"] = (
            [str(v) for v in non_null.sample(sample_size, random_state=0)] if sample_size else []
        )

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preview.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/profiling/preview.py tests/test_preview.py
git commit -m "feat: add column_detail for the Column Explorer panel"
```

---

## Task 5: `preview.py` — `correlation_matrix`

**Files:**
- Modify: `src/profiling/preview.py`
- Test: `tests/test_preview.py` (extend)

**Interfaces:**
- Produces: `correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> dict[str, Any]` returning `{"method": str, "columns": list[str], "matrix": list[list[float]], "truncated": bool}`. Raises `ValueError` on unknown method. Consumed by Task 11 and the Correlations sub-tab (Task 21).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_preview.py`:

```python
from src.profiling.preview import correlation_matrix


def test_correlation_matrix_pearson_perfect_correlation():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8], "z": ["a", "b", "c", "d"]})
    result = correlation_matrix(df, method="pearson")
    assert result["columns"] == ["x", "y"]
    x_idx, y_idx = 0, 1
    assert result["matrix"][x_idx][y_idx] == pytest.approx(1.0)
    assert result["matrix"][x_idx][x_idx] == pytest.approx(1.0)
    assert result["truncated"] is False


def test_correlation_matrix_rejects_unknown_method():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        correlation_matrix(df, method="bogus")


def test_correlation_matrix_mutual_info_diagonal_is_one():
    df = pd.DataFrame({"x": range(20), "y": list(range(10)) * 2})
    result = correlation_matrix(df, method="mutual_info")
    assert result["matrix"][0][0] == 1.0
    assert result["matrix"][1][1] == 1.0


def test_correlation_matrix_handles_fewer_than_two_numeric_columns():
    df = pd.DataFrame({"x": [1, 2, 3], "z": ["a", "b", "c"]})
    result = correlation_matrix(df, method="pearson")
    assert result["matrix"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preview.py -v -k correlation_matrix`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add `correlation_matrix` to `preview.py`**

Append to `src/profiling/preview.py`:

```python
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
        if len(subset) >= 2:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preview.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/profiling/preview.py tests/test_preview.py
git commit -m "feat: add correlation_matrix (pearson/spearman/kendall/mutual_info)"
```

---

## Task 6: `preview.py` — `missing_value_matrix`

**Files:**
- Modify: `src/profiling/preview.py`
- Test: `tests/test_preview.py` (extend)

**Interfaces:**
- Produces: `missing_value_matrix(df: pd.DataFrame) -> dict[str, Any]` returning `{"per_column": [{"column", "null_count", "null_rate"}], "missing_correlation": {"columns": list[str], "matrix": list[list[float]]}}`. Consumed by Task 12 and the Missing Values sub-tab (Task 22).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_preview.py`:

```python
from src.profiling.preview import missing_value_matrix


def test_missing_value_matrix_reports_per_column_rates():
    df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
    result = missing_value_matrix(df)
    per_col = {row["column"]: row for row in result["per_column"]}
    assert per_col["a"]["null_count"] == 2
    assert per_col["a"]["null_rate"] == 0.5
    assert per_col["b"]["null_count"] == 0


def test_missing_value_matrix_correlation_only_over_columns_with_nulls():
    df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, None, 3, None], "c": [1, 2, 3, 4]})
    result = missing_value_matrix(df)
    assert set(result["missing_correlation"]["columns"]) == {"a", "b"}
    assert len(result["missing_correlation"]["matrix"]) == 2


def test_missing_value_matrix_empty_correlation_when_fewer_than_two_null_columns():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    result = missing_value_matrix(df)
    assert result["missing_correlation"]["matrix"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preview.py -v -k missing_value_matrix`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add `missing_value_matrix` to `preview.py`**

Append to `src/profiling/preview.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preview.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/profiling/preview.py tests/test_preview.py
git commit -m "feat: add missing_value_matrix"
```

---

## Task 7: `preview.py` — `detect_outliers`

**Files:**
- Modify: `src/profiling/preview.py`
- Test: `tests/test_preview.py` (extend)

**Interfaces:**
- Consumes: `iqr_outlier_mask` from `src.profiling.heuristics` (Task 1).
- Produces: `detect_outliers(df: pd.DataFrame, method: str = "iqr") -> dict[str, Any]` returning `{"method", "outlier_count", "affected_columns", "example_row_indices"}` (capped at `MAX_OUTLIER_EXAMPLES`). Raises `ValueError` on unknown method. Consumed by Task 12 and the Outliers sub-tab (Task 23).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_preview.py`:

```python
from src.profiling.preview import MAX_OUTLIER_EXAMPLES, detect_outliers


def test_detect_outliers_iqr_flags_extreme_value():
    df = pd.DataFrame({"amount": [10, 11, 12, 13, 14, 15, 1000]})
    result = detect_outliers(df, method="iqr")
    assert result["outlier_count"] == 1
    assert "amount" in result["affected_columns"]
    assert result["example_row_indices"] == [6]


def test_detect_outliers_zscore_flags_extreme_value():
    # A single extreme value among only 6 close values would inflate std
    # enough to mask itself (classic z-score weakness on tiny samples) — use
    # a large tight cluster so the outlier's z-score clears the threshold.
    df = pd.DataFrame({"amount": [50] * 30 + [1000]})
    result = detect_outliers(df, method="zscore")
    assert result["outlier_count"] == 1
    assert result["example_row_indices"] == [30]


def test_detect_outliers_isolation_forest_runs_on_multivariate_data():
    df = pd.DataFrame({"x": list(range(30)) + [500], "y": list(range(30)) + [500]})
    result = detect_outliers(df, method="isolation_forest")
    assert result["outlier_count"] >= 0
    assert result["method"] == "isolation_forest"


def test_detect_outliers_lof_runs_on_multivariate_data():
    df = pd.DataFrame({"x": list(range(30)) + [500], "y": list(range(30)) + [500]})
    result = detect_outliers(df, method="lof")
    assert result["method"] == "lof"


def test_detect_outliers_rejects_unknown_method():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        detect_outliers(df, method="bogus")


def test_detect_outliers_caps_examples_at_configured_max():
    values = [10] * 50 + [10_000] * 30
    df = pd.DataFrame({"amount": values})
    result = detect_outliers(df, method="iqr")
    assert len(result["example_row_indices"]) <= MAX_OUTLIER_EXAMPLES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preview.py -v -k detect_outliers`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add `detect_outliers` to `preview.py`**

Add `from src.profiling.heuristics import iqr_outlier_mask` to the imports at the top of `src/profiling/preview.py`. Then append:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preview.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/profiling/preview.py tests/test_preview.py
git commit -m "feat: add detect_outliers (iqr/zscore/isolation_forest/lof)"
```

---

## Task 8: `preview.py` — `feature_type_counts` and `ml_readiness_score`

**Files:**
- Modify: `src/profiling/preview.py`
- Test: `tests/test_preview.py` (extend)

**Interfaces:**
- Produces: `feature_type_counts(df: pd.DataFrame) -> dict[str, int]` (keys `numeric`, `categorical`, `datetime`, `text`, `boolean`) and `ml_readiness_score(profile: dict[str, Any], leakage_flags: Optional[list[dict[str, Any]]] = None) -> float` (0.0–1.0). Both consumed by Task 12's `GET /api/runs/{id}/dataset-summary`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_preview.py`:

```python
from src.profiling.preview import feature_type_counts, ml_readiness_score


def test_feature_type_counts_classifies_columns():
    df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "plan": ["basic", "pro", "basic"],
            "signed_up_at": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "is_active": [True, False, True],
            "bio": ["a" * 50, "b" * 60, "c" * 45],
        }
    )
    counts = feature_type_counts(df)
    assert counts["numeric"] == 1
    assert counts["categorical"] == 1
    assert counts["datetime"] == 1
    assert counts["boolean"] == 1
    assert counts["text"] == 1


def test_ml_readiness_score_is_bounded():
    profile = {
        "quality": {"completeness": 0.9, "uniqueness": 0.95},
        "column_count": 10,
        "is_wide_dataset": False,
    }
    score = ml_readiness_score(profile, leakage_flags=[])
    assert 0.0 <= score <= 1.0


def test_ml_readiness_score_penalizes_high_severity_leakage():
    profile = {"quality": {"completeness": 1.0, "uniqueness": 1.0}, "column_count": 10, "is_wide_dataset": False}
    clean_score = ml_readiness_score(profile, leakage_flags=[])
    leaky_score = ml_readiness_score(profile, leakage_flags=[{"severity": "high"}] * 5)
    assert leaky_score < clean_score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preview.py -v -k "feature_type_counts or ml_readiness_score"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add both functions to `preview.py`**

Append to `src/profiling/preview.py`:

```python
def feature_type_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = {"numeric": 0, "categorical": 0, "datetime": 0, "text": 0, "boolean": 0}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            counts["boolean"] += 1
        elif pd.api.types.is_datetime64_any_dtype(series):
            counts["datetime"] += 1
        elif pd.api.types.is_numeric_dtype(series):
            counts["numeric"] += 1
        else:
            non_null = series.dropna().astype(str)
            mean_len = non_null.str.len().mean() if len(non_null) else 0.0
            if mean_len > TEXT_MEAN_LENGTH_THRESHOLD:
                counts["text"] += 1
            else:
                counts["categorical"] += 1
    return counts


def ml_readiness_score(profile: dict[str, Any], leakage_flags: Optional[list[dict[str, Any]]] = None) -> float:
    """Heuristic composite score, NOT a guarantee (same caveat convention as
    src/profiling/leakage.py::detect_target_leakage)."""
    quality = profile.get("quality", {}) or {}
    completeness = float(quality.get("completeness", 1.0))
    uniqueness = float(quality.get("uniqueness", 1.0))

    flags = leakage_flags or []
    high_severity = sum(1 for f in flags if f.get("severity") == "high")
    total_columns = max(profile.get("column_count", 1), 1)
    leakage_ratio = min(high_severity / total_columns, 1.0)

    wide_penalty = 0.5 if profile.get("is_wide_dataset") else 0.0

    score = 0.4 * completeness + 0.3 * uniqueness + 0.2 * (1 - leakage_ratio) + 0.1 * (1 - wide_penalty)
    return round(max(0.0, min(1.0, score)), 4)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preview.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add src/profiling/preview.py tests/test_preview.py
git commit -m "feat: add feature_type_counts and ml_readiness_score"
```

---

## Task 9: API — dataset DataFrame cache + `GET /api/datasets`

**Files:**
- Modify: `src/api/server.py:1-60` (imports + cache), and after `list_runs` (around line 498)
- Test: `tests/test_api_datasets.py` (new)

**Interfaces:**
- Consumes: `load_dataset` from `src.data_io` (already imported project-wide; not yet imported in `server.py`).
- Produces: `_get_cached_df(run_id: str, dataset_path: str) -> pd.DataFrame`, `_dataset_df_for_run(run_id: str, entry: dict[str, Any]) -> pd.DataFrame` (404s via `HTTPException` if the file is missing) — consumed by every task from here on that needs the raw dataset. Produces route `GET /api/datasets -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_datasets.py
"""GET /api/datasets lists top-level runs (no source_run_id) as 'datasets' —
a re-run experiment reuses its source's file and isn't a separate dataset
(docs/superpowers/specs/2026-07-06-dataset-preview-design.md)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from src.api import server


def _entry(filename: str, source_run_id: str | None = None) -> dict:
    now = time.time()
    return {
        "state": {"profile": {"row_count": 100, "column_count": 5, "quality": {"overall": 0.9}}},
        "status": "completed",
        "events": [],
        "filename": filename,
        "created_at": now,
        "finished_at": now,
        "cancel_requested": False,
        "chat_history": [],
        **({"source_run_id": source_run_id} if source_run_id else {}),
    }


def test_list_datasets_excludes_rerun_experiments(monkeypatch):
    client = TestClient(server.app)
    monkeypatch.setitem(server._runs, "top-level-run", _entry("churn.csv"))
    monkeypatch.setitem(server._runs, "rerun-run", _entry("churn.csv", source_run_id="top-level-run"))

    datasets = client.get("/api/datasets").json()
    run_ids = {d["run_id"] for d in datasets}
    assert "top-level-run" in run_ids
    assert "rerun-run" not in run_ids


def test_list_datasets_includes_row_and_quality_info(monkeypatch):
    client = TestClient(server.app)
    monkeypatch.setitem(server._runs, "ds-1", _entry("sales.csv"))

    dataset = next(d for d in client.get("/api/datasets").json() if d["run_id"] == "ds-1")
    assert dataset["row_count"] == 100
    assert dataset["column_count"] == 5
    assert dataset["quality_score"] == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_datasets.py -v`
Expected: FAIL with 404 (route doesn't exist yet)

- [ ] **Step 3: Add the cache and route to `server.py`**

Add `import pandas as pd` near the other imports at the top of `src/api/server.py` (alongside the existing `import yaml`), and `from src.data_io import load_dataset` alongside the other `from src....` imports (e.g. near the `from src.state import PipelineState, new_state` line).

After the existing `_lock = threading.Lock()` (line 55), add:

```python
_dataset_df_cache: dict[str, pd.DataFrame] = {}
_dataset_df_cache_lock = threading.Lock()


def _get_cached_df(run_id: str, dataset_path: str) -> pd.DataFrame:
    """Runs are immutable once created (their CSV never changes), so this
    cache never needs invalidation — only lazy population."""
    with _dataset_df_cache_lock:
        if run_id not in _dataset_df_cache:
            _dataset_df_cache[run_id] = load_dataset(dataset_path)
        return _dataset_df_cache[run_id]


def _dataset_df_for_run(run_id: str, entry: dict[str, Any]) -> pd.DataFrame:
    dataset_path = entry["state"].get("dataset_path")
    if not dataset_path or not Path(dataset_path).exists():
        raise HTTPException(status_code=404, detail="dataset file is no longer available for this run")
    return _get_cached_df(run_id, dataset_path)
```

After the existing `list_runs` endpoint (ends around line 498), add:

```python
@app.get("/api/datasets")
def list_datasets(_session: dict[str, Any] = Depends(require_session)) -> list[dict[str, Any]]:
    """A 'dataset' is a top-level run (no source_run_id) — a re-run
    experiment reuses its source's dataset file and is not listed separately
    (see docs/superpowers/specs/2026-07-06-dataset-preview-design.md)."""
    with _lock:
        return [
            {
                "run_id": run_id,
                "filename": entry["filename"],
                "status": entry["status"],
                "created_at": entry["created_at"],
                "row_count": entry["state"].get("profile", {}).get("row_count"),
                "column_count": entry["state"].get("profile", {}).get("column_count"),
                "quality_score": (entry["state"].get("profile", {}).get("quality") or {}).get("overall"),
            }
            for run_id, entry in sorted(_runs.items(), key=lambda kv: -kv[1]["created_at"])
            if not entry.get("source_run_id")
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_datasets.py -v`
Expected: PASS (2 tests)

Run: `pytest tests/ -v -k api`
Expected: PASS (no regressions in other API test modules)

- [ ] **Step 5: Commit**

```bash
git add src/api/server.py tests/test_api_datasets.py
git commit -m "feat: add GET /api/datasets and the per-run DataFrame cache"
```

---

## Task 10: API — `GET /api/runs/{id}/preview`

**Files:**
- Modify: `src/api/server.py` (after the `get_trace` endpoint, ~line 699, before the `ChatRequest` section)
- Test: `tests/test_api_preview.py` (new)

**Interfaces:**
- Consumes: `_dataset_df_for_run` (Task 9), `preview.paginate_rows` (Task 3).
- Produces: route `GET /api/runs/{run_id}/preview?page=&page_size=&sort_by=&sort_dir=&search=` returning `paginate_rows`'s dict plus a `"pii_columns": list[str]` key.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_preview.py
"""GET /api/runs/{id}/preview: paginated raw rows for the Dataset Preview
Data tab, plus which columns are PII (badged, not redacted — this is the
owning user viewing their own upload, not an LLM context; see
docs/superpowers/specs/2026-07-06-dataset-preview-design.md)."""

from __future__ import annotations

import time

import pandas as pd
from fastapi.testclient import TestClient

from src.api import server


def _make_run(tmp_path, monkeypatch):
    df = pd.DataFrame({"amount": range(10), "email": [f"user{i}@example.com" for i in range(10)]})
    dataset_path = tmp_path / "data.csv"
    df.to_csv(dataset_path, index=False)
    now = time.time()
    monkeypatch.setitem(
        server._runs,
        "run-1",
        {
            "state": {
                "dataset_path": str(dataset_path),
                "profile": {"pii_report": {"columns": {"email": {"pii_type": "email"}}}},
            },
            "status": "completed",
            "events": [],
            "filename": "data.csv",
            "created_at": now,
            "finished_at": now,
            "cancel_requested": False,
            "chat_history": [],
        },
    )


def test_preview_returns_page_of_rows(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/preview?page=1&page_size=5").json()
    assert result["total_count"] == 10
    assert len(result["rows"]) == 5
    assert result["pii_columns"] == ["email"]


def test_preview_rejects_oversized_page_size(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    res = client.get("/api/runs/run-1/preview?page=1&page_size=9999")
    assert res.status_code == 400


def test_preview_404s_for_unknown_run():
    client = TestClient(server.app)
    res = client.get("/api/runs/unknown-run/preview")
    assert res.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_preview.py -v`
Expected: FAIL with 404 (route doesn't exist yet)

- [ ] **Step 3: Add the endpoint to `server.py`**

Add `from src.profiling import preview` to the imports at the top of `src/api/server.py`, alongside the other `from src.profiling...` import (`from src.profiling.heuristics import target_too_high_cardinality_for_classification`).

After `get_trace` (ends around line 699, right before `class ChatRequest(BaseModel):`), add:

```python
@app.get("/api/runs/{run_id}/preview")
def get_dataset_preview(
    run_id: str,
    page: int = 1,
    page_size: int = 50,
    sort_by: Optional[str] = None,
    sort_dir: str = "asc",
    search: Optional[str] = None,
    _session: dict[str, Any] = Depends(require_session),
) -> dict[str, Any]:
    entry = _get_entry(run_id)
    df = _dataset_df_for_run(run_id, entry)
    try:
        result = preview.paginate_rows(
            df, page=page, page_size=page_size, sort_by=sort_by, sort_dir=sort_dir, search=search
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    pii_columns = (entry["state"].get("profile", {}).get("pii_report", {}) or {}).get("columns", {}) or {}
    result["pii_columns"] = sorted(pii_columns)
    return _json_safe(result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_preview.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/api/server.py tests/test_api_preview.py
git commit -m "feat: add GET /api/runs/{id}/preview"
```

---

## Task 11: API — `GET /api/runs/{id}/columns/{name}`

**Files:**
- Modify: `src/api/server.py` (immediately after the preview endpoint from Task 10)
- Test: `tests/test_api_column_detail.py` (new)

**Interfaces:**
- Consumes: `_dataset_df_for_run` (Task 9), `preview.column_detail` (Task 4).
- Produces: route `GET /api/runs/{run_id}/columns/{column}` returning `column_detail`'s dict plus an `"ml_insights"` key: `{"analyzed": True, "recommended_steps": [...], "leakage_flags": [...]}` when the run's `eda_report`/`feature_plan`/`leakage_flags` mention that column, else `{"analyzed": False}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_column_detail.py
"""GET /api/runs/{id}/columns/{name}: column-level stats for the Column
Explorer panel, plus already-computed EDA/leakage insights for that column
when the run has progressed far enough to have them."""

from __future__ import annotations

import time

import pandas as pd
from fastapi.testclient import TestClient

from src.api import server


def _make_run(tmp_path, monkeypatch, extra_state=None):
    df = pd.DataFrame({"amount": [1, 2, 3, 4, 5]})
    dataset_path = tmp_path / "data.csv"
    df.to_csv(dataset_path, index=False)
    now = time.time()
    state = {"dataset_path": str(dataset_path)}
    state.update(extra_state or {})
    monkeypatch.setitem(
        server._runs,
        "run-1",
        {
            "state": state,
            "status": "completed",
            "events": [],
            "filename": "data.csv",
            "created_at": now,
            "finished_at": now,
            "cancel_requested": False,
            "chat_history": [],
        },
    )


def test_column_detail_returns_stats(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/columns/amount").json()
    assert result["is_numeric"] is True
    assert result["ml_insights"] == {"analyzed": False}


def test_column_detail_includes_eda_recommendation_when_present(tmp_path, monkeypatch):
    _make_run(
        tmp_path,
        monkeypatch,
        extra_state={
            "feature_plan": {"steps": [{"op": "scale", "columns": ["amount"], "rationale": "wide range"}]},
        },
    )
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/columns/amount").json()
    assert result["ml_insights"]["analyzed"] is True
    assert result["ml_insights"]["recommended_steps"][0]["op"] == "scale"


def test_column_detail_404s_for_unknown_column(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    res = client.get("/api/runs/run-1/columns/nope")
    assert res.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_column_detail.py -v`
Expected: FAIL with 404 (route doesn't exist yet)

- [ ] **Step 3: Add the endpoint to `server.py`**

Add, right after the `get_dataset_preview` endpoint from Task 10:

```python
@app.get("/api/runs/{run_id}/columns/{column}")
def get_column_detail(
    run_id: str, column: str, _session: dict[str, Any] = Depends(require_session)
) -> dict[str, Any]:
    entry = _get_entry(run_id)
    df = _dataset_df_for_run(run_id, entry)
    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"unknown column '{column}'")
    task_spec = entry["state"].get("task_spec") or {}
    result = preview.column_detail(df, column, target_column=task_spec.get("target_column"))

    feature_plan = entry["state"].get("feature_plan") or {}
    matching_steps = [s for s in feature_plan.get("steps", []) if column in (s.get("columns") or [])]
    leakage_flags = [f for f in entry["state"].get("leakage_flags", []) if f.get("column") == column]
    if matching_steps or leakage_flags:
        result["ml_insights"] = {
            "analyzed": True,
            "recommended_steps": [{"op": s.get("op"), "rationale": s.get("rationale")} for s in matching_steps],
            "leakage_flags": leakage_flags,
        }
    else:
        result["ml_insights"] = {"analyzed": False}
    return _json_safe(result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_column_detail.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/api/server.py tests/test_api_column_detail.py
git commit -m "feat: add GET /api/runs/{id}/columns/{name}"
```

---

## Task 12: API — correlations, missing-values, outliers, dataset-summary

**Files:**
- Modify: `src/api/server.py` (immediately after the column-detail endpoint from Task 11)
- Test: `tests/test_api_analytics.py` (new)

**Interfaces:**
- Consumes: `_dataset_df_for_run` (Task 9), `preview.correlation_matrix`/`missing_value_matrix`/`detect_outliers`/`feature_type_counts`/`ml_readiness_score` (Tasks 5-8).
- Produces routes: `GET /api/runs/{id}/correlations?method=`, `GET /api/runs/{id}/missing-values`, `GET /api/runs/{id}/outliers?method=`, `GET /api/runs/{id}/dataset-summary`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_analytics.py
"""GET /api/runs/{id}/{correlations,missing-values,outliers,dataset-summary}
— the analytics sub-tabs and KPI row of the Dataset Preview Data tab."""

from __future__ import annotations

import time

import pandas as pd
from fastapi.testclient import TestClient

from src.api import server


def _make_run(tmp_path, monkeypatch):
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    dataset_path = tmp_path / "data.csv"
    df.to_csv(dataset_path, index=False)
    now = time.time()
    monkeypatch.setitem(
        server._runs,
        "run-1",
        {
            "state": {
                "dataset_path": str(dataset_path),
                "profile": {
                    "row_count": 5,
                    "column_count": 2,
                    "quality": {"completeness": 1.0, "uniqueness": 1.0},
                    "is_wide_dataset": False,
                },
                "leakage_flags": [],
            },
            "status": "completed",
            "events": [],
            "filename": "data.csv",
            "created_at": now,
            "finished_at": now,
            "cancel_requested": False,
            "chat_history": [],
        },
    )


def test_correlations_endpoint(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/correlations?method=pearson").json()
    assert result["columns"] == ["x", "y"]


def test_correlations_endpoint_rejects_unknown_method(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    res = client.get("/api/runs/run-1/correlations?method=bogus")
    assert res.status_code == 400


def test_missing_values_endpoint(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/missing-values").json()
    assert len(result["per_column"]) == 2


def test_outliers_endpoint(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/outliers?method=iqr").json()
    assert result["method"] == "iqr"


def test_dataset_summary_endpoint(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/dataset-summary").json()
    assert result["feature_type_counts"]["numeric"] == 2
    assert 0.0 <= result["ml_readiness_score"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_analytics.py -v`
Expected: FAIL with 404 for all four routes

- [ ] **Step 3: Add the four endpoints to `server.py`**

Add, right after the `get_column_detail` endpoint from Task 11:

```python
@app.get("/api/runs/{run_id}/correlations")
def get_correlations(
    run_id: str, method: str = "pearson", _session: dict[str, Any] = Depends(require_session)
) -> dict[str, Any]:
    entry = _get_entry(run_id)
    df = _dataset_df_for_run(run_id, entry)
    try:
        result = preview.correlation_matrix(df, method=method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _json_safe(result)


@app.get("/api/runs/{run_id}/missing-values")
def get_missing_values(run_id: str, _session: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
    entry = _get_entry(run_id)
    df = _dataset_df_for_run(run_id, entry)
    return _json_safe(preview.missing_value_matrix(df))


@app.get("/api/runs/{run_id}/outliers")
def get_outliers(
    run_id: str, method: str = "iqr", _session: dict[str, Any] = Depends(require_session)
) -> dict[str, Any]:
    entry = _get_entry(run_id)
    df = _dataset_df_for_run(run_id, entry)
    try:
        result = preview.detect_outliers(df, method=method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _json_safe(result)


@app.get("/api/runs/{run_id}/dataset-summary")
def get_dataset_summary(run_id: str, _session: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
    entry = _get_entry(run_id)
    df = _dataset_df_for_run(run_id, entry)
    profile = entry["state"].get("profile", {}) or {}
    leakage_flags = entry["state"].get("leakage_flags", [])
    return _json_safe(
        {
            "feature_type_counts": preview.feature_type_counts(df),
            "ml_readiness_score": preview.ml_readiness_score(profile, leakage_flags),
        }
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_analytics.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/api/server.py tests/test_api_analytics.py
git commit -m "feat: add correlations, missing-values, outliers, and dataset-summary endpoints"
```

---

## Task 13: Expose `memory_bytes` on `/api/runs/{id}`'s `profile_summary`

**Files:**
- Modify: `src/api/server.py:385-392` (the `profile_summary` dict inside `_run_summary`)
- Test: `tests/test_api_run_listing.py` (extend)

**Interfaces:**
- Modifies existing `_run_summary(run_id, entry)` output: `profile_summary.memory_bytes: Optional[int]`, consumed by the KPI row (Task 16).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api_run_listing.py`:

```python
def test_run_summary_includes_memory_bytes(monkeypatch):
    client = TestClient(server.app)
    state = {"profile": {"row_count": 10, "column_count": 2, "memory_bytes": 4096}}
    monkeypatch.setitem(server._runs, "fake-run-3", _entry(state, "completed"))

    run = client.get("/api/runs/fake-run-3").json()
    assert run["profile_summary"]["memory_bytes"] == 4096
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_run_listing.py -v -k memory_bytes`
Expected: FAIL with `KeyError: 'memory_bytes'`

- [ ] **Step 3: Add the field**

In `src/api/server.py`, change the `profile_summary` dict inside `_run_summary` (lines 385-392) from:

```python
            "profile_summary": {
                "row_count": state.get("profile", {}).get("row_count"),
                "column_count": state.get("profile", {}).get("column_count"),
                "pii_columns_detected": state.get("profile", {})
                .get("pii_report", {})
                .get("pii_columns_detected"),
                "quality": state.get("profile", {}).get("quality"),
            },
```

to:

```python
            "profile_summary": {
                "row_count": state.get("profile", {}).get("row_count"),
                "column_count": state.get("profile", {}).get("column_count"),
                "pii_columns_detected": state.get("profile", {})
                .get("pii_report", {})
                .get("pii_columns_detected"),
                "quality": state.get("profile", {}).get("quality"),
                "memory_bytes": state.get("profile", {}).get("memory_bytes"),
            },
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_run_listing.py -v`
Expected: PASS (all tests in file)

- [ ] **Step 5: Commit**

```bash
git add src/api/server.py tests/test_api_run_listing.py
git commit -m "feat: expose memory_bytes in run profile_summary"
```

---

## Task 14: Frontend — enable the Datasets nav item + Datasets list view

**Files:**
- Modify: `frontend/index.html:36` (nav item), and after the run-view `</main>` (currently line 457, before the closing `</div>` of `.main` at line 458)
- Modify: `frontend/app.js` (nav wiring near lines 116-154)
- Modify: `frontend/styles.css` (append a new section)

**Interfaces:**
- Produces: `showDatasetsView()` function and `#nav-datasets` click handler — consumed by Task 15 (dataset detail navigates back here).
- Consumes: `GET /api/datasets` (Task 9), `authFetch`, `$`, `escapeHtml`, `relativeTime` (all already defined in `app.js`).

- [ ] **Step 1: Replace the disabled nav item in `index.html`**

In `frontend/index.html`, replace line 36:

```html
      <span class="nav-item disabled" title="Not available in this local build"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>Datasets <em>soon</em></span>
```

with:

```html
      <button class="nav-item" id="nav-datasets" type="button">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>
        Datasets
      </button>
```

- [ ] **Step 2: Add the Datasets list view markup**

In `frontend/index.html`, immediately after the run-view's closing `</main>` (the line reading `    </main>` right before the final `  </div>\n</div>` at the end of the `.main` column — currently line 457), insert:

```html
    <!-- ============ datasets list ============ -->
    <main class="content hidden" id="datasets-view">
      <section class="card">
        <div class="card-head"><h3>Datasets</h3><span class="muted small" id="datasets-sub"></span></div>
        <div class="datasets-list" id="datasets-list"></div>
      </section>
    </main>
```

- [ ] **Step 3: Add nav wiring and rendering to `app.js`**

In `frontend/app.js`, add alongside the existing nav handlers (near line 116-121, right after `$("new-run-btn").addEventListener("click", showIntakeView);`):

```javascript
$("nav-datasets").addEventListener("click", showDatasetsView);
```

Add a new `showDatasetsView` function near the existing `showIntakeView`/`showRunView` functions (after `showRunView`, around line 154):

```javascript
function showDatasetsView() {
  stopPolling();
  currentRunId = null;
  $("intake-view").classList.add("hidden");
  $("run-view").classList.add("hidden");
  $("dataset-detail-view").classList.add("hidden");
  $("datasets-view").classList.remove("hidden");
  setActiveNav("nav-datasets");
  loadDatasetsList();
}

async function loadDatasetsList() {
  const box = $("datasets-list");
  let datasets = [];
  try {
    datasets = await (await authFetch("/api/datasets")).json();
  } catch {
    box.innerHTML = `<p class="muted small">Could not load datasets.</p>`;
    return;
  }
  $("datasets-sub").textContent = `${datasets.length} dataset${datasets.length === 1 ? "" : "s"}`;
  if (!datasets.length) {
    box.innerHTML = `<p class="muted small">No datasets yet — start an experiment to upload one.</p>`;
    return;
  }
  box.innerHTML = datasets
    .map(
      (d) => `
    <button type="button" class="dataset-row" data-run-id="${d.run_id}">
      <span class="dataset-row-main">
        <span class="dataset-row-name">${escapeHtml(d.filename)}</span>
        <span class="muted small">${d.row_count != null ? Number(d.row_count).toLocaleString() + " rows" : "…"} · ${d.column_count ?? "?"} columns</span>
      </span>
      <span class="dataset-row-meta">
        ${d.quality_score != null ? `<span class="chip detected">${Math.round(d.quality_score * 100)}% quality</span>` : ""}
        <span class="status-badge ${d.status}">${d.status.replaceAll("_", " ")}</span>
        <span class="muted small">${relativeTime(d.created_at)}</span>
      </span>
    </button>`
    )
    .join("");
  box.querySelectorAll(".dataset-row").forEach((el) => {
    el.addEventListener("click", () => openDatasetDetail(el.dataset.runId));
  });
}
```

`openDatasetDetail` is defined in Task 15 — leave it as a forward reference for now (it will exist by the time this file is loaded in the browser since all functions in the file are hoisted/defined before any click can fire).

- [ ] **Step 4: Add styles**

Append to `frontend/styles.css`:

```css
/* ================= datasets list ================= */

.datasets-list { display: grid; gap: 6px; }
.dataset-row {
  display: flex; justify-content: space-between; align-items: center; gap: var(--sp-2);
  font: inherit; text-align: left; width: 100%; cursor: pointer;
  background: var(--bg-surface-raised); border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm); padding: 10px 12px; color: var(--text-primary);
}
.dataset-row:hover { border-color: var(--accent-primary); }
.dataset-row-main { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.dataset-row-name { font-weight: 650; font-size: var(--text-sm); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dataset-row-meta { display: flex; align-items: center; gap: var(--sp-2); flex-shrink: 0; flex-wrap: wrap; justify-content: flex-end; }
```

- [ ] **Step 5: Manual verification**

Run: `python run_server.py` (or the project's existing dev-server entrypoint), log in, click "Datasets" in the sidebar.
Expected: the Datasets view shows, listing any existing runs' datasets (or an empty-state message), with no console errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: add Datasets list view to the sidebar"
```

---

## Task 15: Frontend — dataset detail shell (breadcrumb, disabled tabs, panel scaffolding)

**Files:**
- Modify: `frontend/index.html` (after the `#datasets-view` markup added in Task 14)
- Modify: `frontend/app.js`
- Modify: `frontend/styles.css`

**Interfaces:**
- Produces: `openDatasetDetail(runId: string)` and `showDatasetDetailView()` — consumed by Task 14's dataset-row click handler and later tasks' "back" navigation. Produces `currentDatasetRunId` module-level variable, read by Tasks 16-23 to know which dataset is open.
- Consumes: `GET /api/runs/{id}` (existing endpoint, already returns `filename`, `profile_summary`, etc.)

- [ ] **Step 1: Add the dataset detail view markup**

In `frontend/index.html`, immediately after the `#datasets-view` `</main>` added in Task 14, insert:

```html
    <!-- ============ dataset detail ============ -->
    <main class="content hidden" id="dataset-detail-view">
      <div class="breadcrumb">
        <button type="button" class="breadcrumb-link" id="dataset-breadcrumb-back">Datasets</button>
        <span class="breadcrumb-sep">/</span>
        <span id="dataset-breadcrumb-name"></span>
      </div>

      <div class="tab-bar dataset-tab-bar" role="tablist">
        <button class="tab-btn active" id="dtab-data-btn" type="button" role="tab" aria-selected="true">Data</button>
        <span class="tab-btn disabled" title="Not available in this local build">Overview <em>soon</em></span>
        <span class="tab-btn disabled" title="Not available in this local build">Profile <em>soon</em></span>
        <span class="tab-btn disabled" title="Not available in this local build">Features <em>soon</em></span>
        <span class="tab-btn disabled" title="Not available in this local build">Relationships <em>soon</em></span>
        <span class="tab-btn disabled" title="Not available in this local build">Versions <em>soon</em></span>
        <span class="tab-btn disabled" title="Not available in this local build">Settings <em>soon</em></span>
      </div>

      <section class="stat-row" id="dataset-kpi-row"></section>

      <section class="card">
        <div class="card-head">
          <h3>Data Preview</h3>
          <div class="preview-toolbar">
            <input type="text" id="preview-search" class="preview-search" placeholder="Search…" />
            <div class="preview-colvis" id="preview-colvis"></div>
          </div>
        </div>
        <div class="preview-table-scroll" id="preview-table-scroll">
          <table class="preview-table" id="preview-table"></table>
        </div>
        <div class="preview-pager" id="preview-pager"></div>
      </section>

      <section class="card">
        <div class="tab-bar" role="tablist">
          <button class="tab-btn active" id="ptab-summary-btn" type="button" role="tab" aria-selected="true">Column Summary</button>
          <button class="tab-btn" id="ptab-correlations-btn" type="button" role="tab" aria-selected="false">Correlations</button>
          <button class="tab-btn" id="ptab-missing-btn" type="button" role="tab" aria-selected="false">Missing Values</button>
          <button class="tab-btn" id="ptab-outliers-btn" type="button" role="tab" aria-selected="false">Outliers</button>
        </div>
        <div id="ptab-summary-panel"></div>
        <div id="ptab-correlations-panel" class="hidden"></div>
        <div id="ptab-missing-panel" class="hidden"></div>
        <div id="ptab-outliers-panel" class="hidden"></div>
      </section>

      <div class="column-explorer hidden" id="column-explorer">
        <div class="column-explorer-head">
          <h3 id="column-explorer-name"></h3>
          <button type="button" class="btn ghost" id="column-explorer-close">Close</button>
        </div>
        <div id="column-explorer-body"></div>
      </div>
    </main>
```

- [ ] **Step 2: Add navigation JS**

In `frontend/app.js`, add near the other module-level state variables (after `let predictFormLoadedFor = null;`):

```javascript
let currentDatasetRunId = null;
```

Add, right after `loadDatasetsList` (from Task 14):

```javascript
function showDatasetDetailView() {
  $("datasets-view").classList.add("hidden");
  $("dataset-detail-view").classList.remove("hidden");
  setActiveNav("nav-datasets");
}

async function openDatasetDetail(runId) {
  currentDatasetRunId = runId;
  showDatasetDetailView();
  $("column-explorer").classList.add("hidden");
  let run;
  try {
    run = await (await authFetch(`/api/runs/${runId}`)).json();
  } catch {
    $("dataset-breadcrumb-name").textContent = "Could not load dataset";
    return;
  }
  $("dataset-breadcrumb-name").textContent = run.filename;
}

$("dataset-breadcrumb-back").addEventListener("click", showDatasetsView);
```

- [ ] **Step 3: Add styles**

Append to `frontend/styles.css`:

```css
/* ================= dataset detail ================= */

.breadcrumb { display: flex; align-items: center; gap: 8px; font-size: var(--text-sm); margin-bottom: var(--sp-2); }
.breadcrumb-link { font: inherit; background: none; border: none; color: var(--accent-primary); cursor: pointer; padding: 0; }
.breadcrumb-sep { color: var(--text-secondary); }
#dataset-breadcrumb-name { font-weight: 650; }

.dataset-tab-bar { margin-bottom: 0; }
.dataset-tab-bar .tab-btn.disabled { cursor: default; opacity: 0.45; }
.dataset-tab-bar .tab-btn.disabled:hover { color: var(--text-secondary); }
.dataset-tab-bar em { font-style: normal; font-size: 10px; margin-left: 4px; border: 1px solid currentColor; border-radius: 999px; padding: 1px 6px; opacity: 0.8; }

.column-explorer {
  position: fixed; top: 0; right: 0; height: 100vh; width: min(420px, 100vw);
  background: var(--bg-surface); border-left: 1px solid var(--border-subtle);
  box-shadow: var(--shadow); padding: var(--sp-4); overflow-y: auto; z-index: 20;
}
.column-explorer-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--sp-3); }
```

- [ ] **Step 4: Manual verification**

Run the dev server, open Datasets, click a dataset row.
Expected: the detail view shows the breadcrumb with the dataset's filename, a "Data" tab plus six disabled "soon" tabs, and empty KPI row / table / sub-tab placeholders (filled in by later tasks) with no console errors. Clicking "Datasets" in the breadcrumb returns to the list.

- [ ] **Step 5: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/styles.css
git commit -m "feat: add dataset detail view shell with breadcrumb and disabled tabs"
```

---

## Task 16: Frontend — KPI row

**Files:**
- Modify: `frontend/app.js` (inside `openDatasetDetail`, from Task 15)
- Modify: `frontend/styles.css`

**Interfaces:**
- Consumes: `GET /api/runs/{id}` (`profile_summary`, `profile_columns`, `task_spec`), `GET /api/runs/{id}/dataset-summary` (Task 12). Reuses existing `ICONS`, `formatDuration`-style helpers, `.stat-row`/`.stat-card` CSS from the run dashboard (`styles.css:291-320`).
- Produces: `renderDatasetKpis(run, summary)` and `formatBytes(bytes)` helpers, called from `openDatasetDetail`.

- [ ] **Step 1: Add `formatBytes` and `renderDatasetKpis` to `app.js`**

Add near the other formatting helpers (after `formatDuration`, around line 1436):

```javascript
function formatBytes(bytes) {
  if (bytes == null) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}
```

Add near the datasets functions (after `openDatasetDetail`):

```javascript
function renderDatasetKpis(run, summary) {
  const profile = run.profile_summary || {};
  const counts = summary.feature_type_counts || {};
  const cards = [
    { icon: "db", tint: "violet", label: "Total Rows", value: profile.row_count != null ? Number(profile.row_count).toLocaleString() : "—" },
    { icon: "grid", tint: "violet", label: "Total Columns", value: String(profile.column_count ?? "—") },
    { icon: "warning", tint: "amber", label: "Missing Values", value: profile.quality ? `${(100 - profile.quality.completeness * 100).toFixed(1)}%` : "—" },
    { icon: "layers", tint: "amber", label: "Duplicate Rows", value: profile.quality ? String(profile.quality.duplicate_row_count) : "—" },
    { icon: "file", tint: "violet", label: "Memory Usage", value: formatBytes(profile.memory_bytes) },
    { icon: "sliders", tint: "green", label: "Numeric Features", value: String(counts.numeric ?? "—") },
    { icon: "grid", tint: "green", label: "Categorical Features", value: String(counts.categorical ?? "—") },
    { icon: "clock", tint: "green", label: "Datetime Features", value: String(counts.datetime ?? "—") },
    { icon: "file", tint: "green", label: "Text Features", value: String(counts.text ?? "—") },
    { icon: "shield", tint: "violet", label: "Data Quality Score", value: profile.quality ? `${Math.round(profile.quality.overall * 100)}%` : "—" },
    { icon: "shield", tint: "violet", label: "ML Readiness Score", value: `${Math.round((summary.ml_readiness_score ?? 0) * 100)}%` },
  ];
  const targetCol = (run.task_spec || {}).target_column;
  if (targetCol) {
    cards.push({ icon: "trophy", tint: "violet", label: "Target Column", value: targetCol });
  }
  $("dataset-kpi-row").innerHTML = cards
    .map(
      (c) => `
      <div class="stat-card">
        <span class="stat-icon ${c.tint}">${ICONS[c.icon]}</span>
        <div class="stat-body">
          <div class="stat-label">${c.label}</div>
          <div class="stat-value">${escapeHtml(c.value)}</div>
        </div>
      </div>`
    )
    .join("");
}
```

- [ ] **Step 2: Wire it into `openDatasetDetail`**

In `frontend/app.js`, modify `openDatasetDetail` (from Task 15) to fetch the summary and render KPIs. Replace:

```javascript
  $("dataset-breadcrumb-name").textContent = run.filename;
}
```

with:

```javascript
  $("dataset-breadcrumb-name").textContent = run.filename;

  let summary = { feature_type_counts: {}, ml_readiness_score: 0 };
  try {
    summary = await (await authFetch(`/api/runs/${runId}/dataset-summary`)).json();
  } catch { /* KPI row degrades gracefully to "—" placeholders */ }
  renderDatasetKpis(run, summary);
}
```

- [ ] **Step 3: Manual verification**

Open a dataset's detail page.
Expected: the KPI row shows Total Rows/Columns/Missing/Duplicates/Memory/feature-type counts/quality/readiness scores populated from real numbers, not placeholders (assuming at least one run exists with a computed profile).

- [ ] **Step 4: Commit**

```bash
git add frontend/app.js
git commit -m "feat: render dataset KPI row from profile_summary and dataset-summary"
```

---

## Task 17: Frontend — interactive preview table

**Files:**
- Modify: `frontend/app.js`
- Modify: `frontend/styles.css`

**Interfaces:**
- Consumes: `GET /api/runs/{id}/preview` (Task 10), `run.profile_columns` (existing, from `GET /api/runs/{id}`).
- Produces: `loadPreviewTable(run)`, `fetchAndRenderPreviewPage()`, `renderPreviewTable(data)` (reads column metadata from the module-level `previewColumns` set by `loadPreviewTable`, not a parameter), module state `previewState = {page, pageSize, sortBy, sortDir, search}`. Called from `openDatasetDetail`.

**Scope note (deviates from the design doc):** this task implements sticky header/first column, single-column sort, search, and column *visibility* persisted to `localStorage`. It does **not** implement drag-based column resize or drag-based column reorder, or persisting width/order — those are materially more involved (drag-and-drop state machines, resize-handle hit-testing) than the rest of this task and add real risk of a half-working interaction. If you want them, treat as a follow-up task after this plan ships and the simpler table is verified working.

- [ ] **Step 1: Add preview state and fetch/render functions**

Add near `currentDatasetRunId` (Task 15):

```javascript
let previewState = { page: 1, pageSize: 50, sortBy: null, sortDir: "asc", search: "" };
let previewColumns = [];
```

Add near the dataset functions:

```javascript
function classifyPreviewType(dtype) {
  const d = String(dtype || "").toLowerCase();
  if (d.includes("bool")) return "boolean";
  if (d.includes("datetime") || d.includes("date")) return "datetime";
  if (d.includes("int") || d.includes("float")) return "numeric";
  return "categorical";
}

async function loadPreviewTable(run) {
  previewColumns = run.profile_columns || [];
  const layoutKey = `automl-preview-layout-${currentDatasetRunId}`;
  const savedLayout = JSON.parse(localStorage.getItem(layoutKey) || "{}");
  const hiddenColumns = new Set(savedLayout.hiddenColumns || []);

  $("preview-colvis").innerHTML = previewColumns
    .map(
      (c) => `<label class="colvis-item"><input type="checkbox" data-col="${escapeHtml(c.name)}" ${hiddenColumns.has(c.name) ? "" : "checked"}/> ${escapeHtml(c.name)}</label>`
    )
    .join("");
  $("preview-colvis").querySelectorAll("input[type=checkbox]").forEach((input) => {
    input.addEventListener("change", () => {
      const hidden = new Set(
        Array.from($("preview-colvis").querySelectorAll("input:not(:checked)")).map((el) => el.dataset.col)
      );
      localStorage.setItem(layoutKey, JSON.stringify({ hiddenColumns: [...hidden] }));
      fetchAndRenderPreviewPage();
    });
  });

  await fetchAndRenderPreviewPage();
}

async function fetchAndRenderPreviewPage() {
  const params = new URLSearchParams({
    page: String(previewState.page),
    page_size: String(previewState.pageSize),
    sort_dir: previewState.sortDir,
  });
  if (previewState.sortBy) params.set("sort_by", previewState.sortBy);
  if (previewState.search) params.set("search", previewState.search);

  let data;
  try {
    data = await (await authFetch(`/api/runs/${currentDatasetRunId}/preview?${params}`)).json();
  } catch {
    $("preview-table").innerHTML = `<tr><td>Could not load preview.</td></tr>`;
    return;
  }
  renderPreviewTable(data);
}

function renderPreviewTable(data) {
  const layoutKey = `automl-preview-layout-${currentDatasetRunId}`;
  const savedLayout = JSON.parse(localStorage.getItem(layoutKey) || "{}");
  const hiddenColumns = new Set(savedLayout.hiddenColumns || []);
  const visibleColumns = previewColumns.filter((c) => !hiddenColumns.has(c.name));
  const piiSet = new Set(data.pii_columns || []);
  const duplicateSet = new Set(data.duplicate_row_indices || []);

  const numericRanges = {};
  for (const col of visibleColumns) {
    if (classifyPreviewType(col.dtype) !== "numeric") continue;
    const values = data.rows.map((r) => r[col.name]).filter((v) => v != null);
    numericRanges[col.name] = values.length ? { min: Math.min(...values), max: Math.max(...values) } : null;
  }

  let html = "<tr>" + visibleColumns
    .map(
      (c) => `<th data-col="${escapeHtml(c.name)}" class="sortable">
        ${escapeHtml(c.name)} <span class="col-type-badge">${classifyPreviewType(c.dtype)}</span>
        ${piiSet.has(c.name) ? `<span class="chip flagged" title="PII column">PII</span>` : ""}
        <button type="button" class="col-profile-btn" data-col="${escapeHtml(c.name)}" title="Profile this column">${ICONS.search}</button>
      </th>`
    )
    .join("") + "</tr>";

  for (const row of data.rows) {
    const isDup = duplicateSet.has(row._row_index);
    html += `<tr class="${isDup ? "preview-row-duplicate" : ""}">`;
    for (const col of visibleColumns) {
      const value = row[col.name];
      const type = classifyPreviewType(col.dtype);
      if (value == null) {
        html += `<td class="preview-cell-missing">—</td>`;
      } else if (type === "numeric") {
        const range = numericRanges[col.name];
        const pct = range && range.max > range.min ? (value - range.min) / (range.max - range.min) : 0;
        html += `<td class="num preview-cell-numeric" style="background:linear-gradient(90deg, var(--accent-primary-soft) ${(pct * 100).toFixed(0)}%, transparent ${(pct * 100).toFixed(0)}%)">${escapeHtml(String(value))}</td>`;
      } else if (type === "boolean") {
        html += `<td><span class="chip ${value ? "detected" : "flagged"}">${String(value)}</span></td>`;
      } else if (type === "categorical") {
        html += `<td><span class="chip detected">${escapeHtml(String(value))}</span></td>`;
      } else {
        html += `<td title="${escapeHtml(String(value))}">${escapeHtml(String(value))}</td>`;
      }
    }
    html += "</tr>";
  }
  $("preview-table").innerHTML = html;

  $("preview-table").querySelectorAll("th.sortable").forEach((th) => {
    th.addEventListener("click", (e) => {
      if (e.target.closest(".col-profile-btn")) return;
      const col = th.dataset.col;
      previewState.sortDir = previewState.sortBy === col && previewState.sortDir === "asc" ? "desc" : "asc";
      previewState.sortBy = col;
      fetchAndRenderPreviewPage();
    });
  });
  $("preview-table").querySelectorAll(".col-profile-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      openColumnExplorer(btn.dataset.col);
    });
  });

  const totalPages = Math.max(1, Math.ceil(data.total_count / previewState.pageSize));
  $("preview-pager").innerHTML = `
    <button type="button" class="btn ghost" id="preview-prev" ${previewState.page <= 1 ? "disabled" : ""}>Prev</button>
    <span class="muted small">Page ${data.page} of ${totalPages} · ${data.total_count.toLocaleString()} rows</span>
    <button type="button" class="btn ghost" id="preview-next" ${previewState.page >= totalPages ? "disabled" : ""}>Next</button>`;
  $("preview-prev").addEventListener("click", () => { previewState.page -= 1; fetchAndRenderPreviewPage(); });
  $("preview-next").addEventListener("click", () => { previewState.page += 1; fetchAndRenderPreviewPage(); });
}

let previewSearchDebounce = null;
$("preview-search").addEventListener("input", (e) => {
  clearTimeout(previewSearchDebounce);
  previewSearchDebounce = setTimeout(() => {
    previewState.search = e.target.value.trim();
    previewState.page = 1;
    fetchAndRenderPreviewPage();
  }, 300);
});
```

`openColumnExplorer` is defined in Task 18 — same forward-reference note as Task 14.

- [ ] **Step 2: Wire it into `openDatasetDetail`**

In `frontend/app.js`, extend `openDatasetDetail` (from Tasks 15/16) — replace:

```javascript
  renderDatasetKpis(run, summary);
}
```

with:

```javascript
  renderDatasetKpis(run, summary);

  previewState = { page: 1, pageSize: 50, sortBy: null, sortDir: "asc", search: "" };
  $("preview-search").value = "";
  await loadPreviewTable(run);
}
```

- [ ] **Step 3: Add styles**

Append to `frontend/styles.css`:

```css
/* ================= interactive preview table ================= */

.preview-toolbar { display: flex; align-items: center; gap: var(--sp-2); }
.preview-search { font-size: var(--text-sm); min-width: 200px; }
.preview-colvis { position: relative; display: flex; flex-wrap: wrap; gap: 4px; max-width: 340px; }
.colvis-item { font-size: var(--text-xs); display: flex; align-items: center; gap: 4px; }

.preview-table-scroll { overflow: auto; max-height: 480px; border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); }
.preview-table { width: 100%; border-collapse: collapse; font-size: var(--text-sm); }
.preview-table th {
  position: sticky; top: 0; background: var(--bg-surface-raised); text-align: left;
  padding: 8px 10px; border-bottom: 1px solid var(--border-subtle); white-space: nowrap; cursor: pointer;
}
.preview-table th:first-child, .preview-table td:first-child { position: sticky; left: 0; background: var(--bg-surface); z-index: 1; }
.preview-table td { padding: 6px 10px; border-bottom: 1px solid var(--border-subtle); max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.preview-table td.num { text-align: right; font-family: var(--font-mono); }
.col-type-badge { font-size: 10px; color: var(--text-secondary); font-weight: 500; text-transform: uppercase; margin-left: 4px; }
.col-profile-btn { background: none; border: none; cursor: pointer; color: var(--text-secondary); margin-left: 4px; vertical-align: middle; }
.col-profile-btn:hover { color: var(--accent-primary); }
.preview-cell-missing { color: var(--text-secondary); font-style: italic; }
.preview-row-duplicate { border-left: 3px solid var(--accent-warning); }
.preview-pager { display: flex; align-items: center; justify-content: center; gap: var(--sp-3); margin-top: var(--sp-3); }
```

- [ ] **Step 4: Manual verification**

Open a dataset's detail page.
Expected: the preview table shows real rows, paginated at 50/page; clicking a column header sorts by it (toggling asc/desc); typing in the search box filters after a short debounce; unchecking a column in the visibility list hides it and persists across a page reload; missing cells show a muted "—"; numeric cells are right-aligned with a heatmap tint.

- [ ] **Step 5: Commit**

```bash
git add frontend/app.js frontend/styles.css
git commit -m "feat: add interactive paginated/sortable/searchable preview table"
```

---

## Task 18: Frontend — Column Explorer panel

**Files:**
- Modify: `frontend/app.js`

**Interfaces:**
- Consumes: `GET /api/runs/{id}/columns/{name}` (Task 11).
- Produces: `openColumnExplorer(columnName)` (referenced by Task 17's profile button).

- [ ] **Step 1: Add `openColumnExplorer`**

Add near the other dataset-detail functions in `frontend/app.js`:

```javascript
async function openColumnExplorer(columnName) {
  $("column-explorer-name").textContent = columnName;
  $("column-explorer").classList.remove("hidden");
  $("column-explorer-body").innerHTML = `<p class="muted small">Loading…</p>`;

  let detail;
  try {
    detail = await (await authFetch(`/api/runs/${currentDatasetRunId}/columns/${encodeURIComponent(columnName)}`)).json();
  } catch {
    $("column-explorer-body").innerHTML = `<p class="muted small">Could not load column details.</p>`;
    return;
  }

  let html = `<p class="muted small mono">${escapeHtml(detail.dtype)}</p>`;

  if (detail.is_numeric) {
    const hist = detail.histogram;
    const maxCount = Math.max(...hist.counts, 1);
    html += `<div class="explorer-histogram">${hist.counts
      .map((c) => `<span class="explorer-bar" style="height:${((c / maxCount) * 100).toFixed(0)}%" title="${c}"></span>`)
      .join("")}</div>`;
    const s = detail.stats;
    html += `<div class="explorer-stats">
      <div>Mean <strong>${s.mean.toFixed(2)}</strong></div>
      <div>Median <strong>${s.median.toFixed(2)}</strong></div>
      <div>Std Dev <strong>${s.std.toFixed(2)}</strong></div>
      <div>Min <strong>${s.min.toFixed(2)}</strong></div>
      <div>Max <strong>${s.max.toFixed(2)}</strong></div>
      <div>P25 <strong>${s.p25.toFixed(2)}</strong></div>
      <div>P75 <strong>${s.p75.toFixed(2)}</strong></div>
      <div>Skew <strong>${s.skew.toFixed(2)}</strong></div>
    </div>`;
    if (detail.correlation_with_target != null) {
      html += `<p class="muted small">Correlation with target: <strong>${detail.correlation_with_target.toFixed(3)}</strong></p>`;
    }
  } else {
    html += `<div class="field"><span>Top values</span><ul class="callout-list">${Object.entries(detail.top_values)
      .map(([k, v]) => `<li>${escapeHtml(k)} <span class="muted small">(${v})</span></li>`)
      .join("")}</ul></div>`;
  }

  if (detail.ml_insights.analyzed) {
    html += `<div class="field"><span>ML Insights</span><ul class="callout-list">${detail.ml_insights.recommended_steps
      .map((s) => `<li><span class="step-op">${escapeHtml(s.op)}</span><span class="step-rationale">${escapeHtml(s.rationale || "")}</span></li>`)
      .join("")}${detail.ml_insights.leakage_flags
      .map((f) => `<li>${ICONS.warning}<span>${escapeHtml(f.reason)}</span></li>`)
      .join("")}</ul></div>`;
  } else {
    html += `<p class="muted small">Run further into the pipeline to see ML insights for this column.</p>`;
  }

  $("column-explorer-body").innerHTML = html;
}

$("column-explorer-close").addEventListener("click", () => $("column-explorer").classList.add("hidden"));
```

- [ ] **Step 2: Add styles**

Append to `frontend/styles.css`:

```css
/* ================= column explorer ================= */

.explorer-histogram { display: flex; align-items: flex-end; gap: 2px; height: 100px; margin: var(--sp-3) 0; }
.explorer-bar { flex: 1; background: var(--accent-primary); border-radius: 2px 2px 0 0; min-height: 1px; }
.explorer-stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; font-size: var(--text-xs); color: var(--text-secondary); margin-bottom: var(--sp-3); }
.explorer-stats strong { color: var(--text-primary); font-family: var(--font-mono); }
```

- [ ] **Step 3: Manual verification**

Click the profile icon on a numeric column header, then on a categorical one.
Expected: a right-side panel opens with a histogram + stats for the numeric column, and top values for the categorical one; "Close" hides it.

- [ ] **Step 4: Commit**

```bash
git add frontend/app.js frontend/styles.css
git commit -m "feat: add Column Explorer panel"
```

---

## Task 19: Frontend — Column Summary sub-tab

**Files:**
- Modify: `frontend/app.js`
- Modify: `frontend/styles.css`

**Interfaces:**
- Consumes: `run.profile_columns` (existing, already fetched in `openDatasetDetail`).
- Produces: `renderColumnSummaryTab(run)`, and generic sub-tab switching wired to the four `ptab-*-btn` buttons from Task 15.

- [ ] **Step 1: Add sub-tab switching and the Column Summary renderer**

Add near the other dataset-detail functions:

```javascript
const PROFILING_SUBTABS = ["summary", "correlations", "missing", "outliers"];

function switchProfilingSubtab(name) {
  for (const tab of PROFILING_SUBTABS) {
    const isActive = tab === name;
    $(`ptab-${tab}-btn`).classList.toggle("active", isActive);
    $(`ptab-${tab}-btn`).setAttribute("aria-selected", String(isActive));
    $(`ptab-${tab}-panel`).classList.toggle("hidden", !isActive);
  }
  if (name === "correlations" && !$("ptab-correlations-panel").dataset.loaded) loadCorrelationsTab();
  if (name === "missing" && !$("ptab-missing-panel").dataset.loaded) loadMissingValuesTab();
  if (name === "outliers" && !$("ptab-outliers-panel").dataset.loaded) loadOutliersTab();
}
for (const tab of PROFILING_SUBTABS) {
  $(`ptab-${tab}-btn`).addEventListener("click", () => switchProfilingSubtab(tab));
}

function renderColumnSummaryTab(run) {
  const columns = run.profile_columns || [];
  let html = "<table class=\"results-table\"><tr><th>Column</th><th>Type</th><th>Missing %</th><th>Unique %</th><th>Cardinality</th></tr>";
  const rowCount = (run.profile_summary || {}).row_count || 1;
  for (const c of columns) {
    html += `<tr>
      <td>${escapeHtml(c.name)}</td>
      <td>${escapeHtml(c.dtype)}</td>
      <td class="num">${((c.null_rate || 0) * 100).toFixed(1)}%</td>
      <td class="num">${(((c.n_unique || 0) / rowCount) * 100).toFixed(1)}%</td>
      <td class="num">${c.n_unique ?? "—"}</td>
    </tr>`;
  }
  html += "</table>";
  $("ptab-summary-panel").innerHTML = html;
  $("ptab-summary-panel").dataset.loaded = "1";
}
```

`loadCorrelationsTab`/`loadMissingValuesTab`/`loadOutliersTab` are defined in Tasks 20-22 — same forward-reference note as prior tasks.

- [ ] **Step 2: Wire it into `openDatasetDetail`**

Extend `openDatasetDetail` (from Task 17) — replace:

```javascript
  previewState = { page: 1, pageSize: 50, sortBy: null, sortDir: "asc", search: "" };
  $("preview-search").value = "";
  await loadPreviewTable(run);
}
```

with:

```javascript
  previewState = { page: 1, pageSize: 50, sortBy: null, sortDir: "asc", search: "" };
  $("preview-search").value = "";
  await loadPreviewTable(run);

  for (const tab of PROFILING_SUBTABS) $(`ptab-${tab}-panel`).dataset.loaded = "";
  renderColumnSummaryTab(run);
  switchProfilingSubtab("summary");
}
```

- [ ] **Step 3: Manual verification**

Open a dataset's detail page.
Expected: the "Column Summary" sub-tab shows a table of every column's dtype/missing%/unique%/cardinality by default.

- [ ] **Step 4: Commit**

```bash
git add frontend/app.js
git commit -m "feat: add Column Summary profiling sub-tab"
```

---

## Task 20: Frontend — Correlations sub-tab

**Files:**
- Modify: `frontend/app.js`
- Modify: `frontend/styles.css`

**Interfaces:**
- Consumes: `GET /api/runs/{id}/correlations?method=` (Task 12).
- Produces: `loadCorrelationsTab()` (referenced by Task 19's `switchProfilingSubtab`), `renderCorrelationHeatmap(container, result)`.

- [ ] **Step 1: Add the loader and a shared heatmap renderer**

Add near the other sub-tab functions:

```javascript
function renderCorrelationHeatmap(container, result) {
  const { columns, matrix } = result;
  if (!columns.length) {
    container.innerHTML = `<p class="muted small">Not enough numeric columns for a correlation matrix.</p>`;
    return;
  }
  const cell = 34;
  let html = `<div class="heatmap-scroll"><svg width="${cell * (columns.length + 1)}" height="${cell * (columns.length + 1)}" role="img" aria-label="Correlation heatmap">`;
  columns.forEach((name, i) => {
    html += `<text x="${cell * (i + 1) + cell / 2}" y="${cell * 0.9}" font-size="9" text-anchor="middle" transform="rotate(-45 ${cell * (i + 1) + cell / 2} ${cell * 0.9})">${escapeHtml(name)}</text>`;
    html += `<text x="${cell * 0.95}" y="${cell * (i + 1) + cell / 2 + 3}" font-size="9" text-anchor="end">${escapeHtml(name)}</text>`;
  });
  matrix.forEach((row, i) => {
    row.forEach((value, j) => {
      const intensity = Math.min(Math.abs(value), 1);
      const color = value >= 0 ? `rgba(124, 58, 237, ${intensity})` : `rgba(220, 38, 38, ${intensity})`;
      html += `<rect x="${cell * (j + 1)}" y="${cell * (i + 1)}" width="${cell}" height="${cell}" fill="${color}"><title>${escapeHtml(columns[i])} × ${escapeHtml(columns[j])}: ${value.toFixed(2)}</title></rect>`;
    });
  });
  html += "</svg></div>";
  container.innerHTML = html;
}

async function loadCorrelationsTab() {
  const panel = $("ptab-correlations-panel");
  panel.innerHTML = `
    <div class="chip-row">
      <label class="field" style="max-width:200px">
        <span class="visually-hidden">Correlation method</span>
        <select id="correlation-method-select">
          <option value="pearson">Pearson</option>
          <option value="spearman">Spearman</option>
          <option value="kendall">Kendall</option>
          <option value="mutual_info">Mutual Information</option>
        </select>
      </label>
    </div>
    <div id="correlation-heatmap-box"><p class="muted small">Loading…</p></div>`;
  panel.dataset.loaded = "1";

  const fetchAndRender = async () => {
    const method = $("correlation-method-select").value;
    let result;
    try {
      result = await (await authFetch(`/api/runs/${currentDatasetRunId}/correlations?method=${method}`)).json();
    } catch {
      $("correlation-heatmap-box").innerHTML = `<p class="muted small">Could not load correlations.</p>`;
      return;
    }
    renderCorrelationHeatmap($("correlation-heatmap-box"), result);
  };
  $("correlation-method-select").addEventListener("change", fetchAndRender);
  await fetchAndRender();
}
```

- [ ] **Step 2: Add styles**

Append to `frontend/styles.css`:

```css
/* ================= correlations / missing-values heatmaps ================= */

.heatmap-scroll { overflow: auto; max-width: 100%; }
```

- [ ] **Step 3: Manual verification**

Open a dataset's detail page, click the "Correlations" sub-tab, switch the method dropdown.
Expected: an SVG heatmap renders for numeric columns, colored by correlation strength/sign, with a tooltip on hover; switching methods (pearson/spearman/kendall/mutual_info) re-fetches and re-renders.

- [ ] **Step 4: Commit**

```bash
git add frontend/app.js frontend/styles.css
git commit -m "feat: add Correlations profiling sub-tab with method selector"
```

---

## Task 21: Frontend — Missing Values sub-tab

**Files:**
- Modify: `frontend/app.js`

**Interfaces:**
- Consumes: `GET /api/runs/{id}/missing-values` (Task 12), `renderCorrelationHeatmap` (Task 20, reused for the nullness-correlation matrix).
- Produces: `loadMissingValuesTab()` (referenced by Task 19's `switchProfilingSubtab`).

- [ ] **Step 1: Add the loader**

Add near the other sub-tab functions:

```javascript
async function loadMissingValuesTab() {
  const panel = $("ptab-missing-panel");
  panel.innerHTML = `<p class="muted small">Loading…</p>`;
  panel.dataset.loaded = "1";

  let result;
  try {
    result = await (await authFetch(`/api/runs/${currentDatasetRunId}/missing-values`)).json();
  } catch {
    panel.innerHTML = `<p class="muted small">Could not load missing-value analysis.</p>`;
    return;
  }

  const rows = result.per_column.filter((r) => r.null_count > 0).sort((a, b) => b.null_rate - a.null_rate);
  let html = `<div class="quality-bars">${rows
    .map(
      (r) => `
    <div class="quality-row">
      <span class="quality-name">${escapeHtml(r.column)}</span>
      <span class="fi-track"><span class="fi-fill quality-fill" style="width:${(r.null_rate * 100).toFixed(1)}%;background:var(--accent-warning)"></span></span>
      <span class="quality-value mono">${(r.null_rate * 100).toFixed(1)}%</span>
    </div>`
    )
    .join("")}</div>`;
  if (!rows.length) html = `<p class="muted small">No missing values in this dataset.</p>`;

  html += `<h4 class="missing-corr-title">Which columns tend to be missing together</h4><div id="missing-corr-box"></div>`;
  panel.innerHTML = html;
  renderCorrelationHeatmap($("missing-corr-box"), { columns: result.missing_correlation.columns, matrix: result.missing_correlation.matrix });
}
```

- [ ] **Step 2: Add styles**

Append to `frontend/styles.css`:

```css
.missing-corr-title { font-size: var(--text-sm); font-weight: 650; margin: var(--sp-3) 0 var(--sp-2); }
```

- [ ] **Step 3: Manual verification**

Open the "Missing Values" sub-tab on a dataset with at least one nullable column.
Expected: a bar list of per-column missing rates, and (if 2+ columns have nulls) a heatmap of which columns tend to be null together.

- [ ] **Step 4: Commit**

```bash
git add frontend/app.js frontend/styles.css
git commit -m "feat: add Missing Values profiling sub-tab"
```

---

## Task 22: Frontend — Outliers sub-tab

**Files:**
- Modify: `frontend/app.js`
- Modify: `frontend/styles.css`

**Interfaces:**
- Consumes: `GET /api/runs/{id}/outliers?method=` (Task 12).
- Produces: `loadOutliersTab()` (referenced by Task 19's `switchProfilingSubtab`).

- [ ] **Step 1: Add the loader**

Add near the other sub-tab functions:

```javascript
async function loadOutliersTab() {
  const panel = $("ptab-outliers-panel");
  panel.innerHTML = `
    <div class="chip-row">
      <label class="field" style="max-width:200px">
        <span class="visually-hidden">Outlier detection method</span>
        <select id="outlier-method-select">
          <option value="iqr">IQR</option>
          <option value="zscore">Z-score</option>
          <option value="isolation_forest">Isolation Forest</option>
          <option value="lof">Local Outlier Factor</option>
        </select>
      </label>
    </div>
    <div id="outlier-result-box"><p class="muted small">Loading…</p></div>`;
  panel.dataset.loaded = "1";

  const fetchAndRender = async () => {
    const method = $("outlier-method-select").value;
    const box = $("outlier-result-box");
    box.innerHTML = `<p class="muted small">Detecting…</p>`;
    let result;
    try {
      result = await (await authFetch(`/api/runs/${currentDatasetRunId}/outliers?method=${method}`)).json();
    } catch {
      box.innerHTML = `<p class="muted small">Could not run outlier detection.</p>`;
      return;
    }
    box.innerHTML = `
      <div class="chips">
        <span class="chip flagged">${result.outlier_count} outlier row(s) detected</span>
        ${result.affected_columns.map((c) => `<span class="chip detected">${escapeHtml(c)}</span>`).join("")}
      </div>
      ${result.example_row_indices.length ? `<p class="muted small">Example row indices: ${result.example_row_indices.join(", ")}</p>` : ""}`;
  };
  $("outlier-method-select").addEventListener("change", fetchAndRender);
  await fetchAndRender();
}
```

- [ ] **Step 2: Manual verification**

Open the "Outliers" sub-tab, switch between all four methods.
Expected: each method shows an outlier count, affected columns as chips, and example row indices; switching methods re-runs detection without a page reload.

- [ ] **Step 3: Commit**

```bash
git add frontend/app.js
git commit -m "feat: add Outliers profiling sub-tab with method selector"
```

---

## Task 23: Full manual verification pass

**Files:** None (verification only).

- [ ] **Step 1: Run the full backend test suite**

Run: `pytest tests/ -v`
Expected: PASS — all pre-existing tests plus every test added in Tasks 1-13.

- [ ] **Step 2: Start the app and walk the full Data tab flow**

Run: `python run_server.py` (or the project's existing entrypoint), log in with the demo credentials.

Manually verify, per the `verify` skill's "drive the real flow" convention:
1. Upload a CSV and start a run (existing flow, unaffected).
2. Click "Datasets" in the sidebar — the new run's dataset appears in the list.
3. Open it — breadcrumb shows the filename; "Data" tab is active; the other six tabs are visibly disabled with "soon" badges.
4. KPI row shows real numbers (rows, columns, missing %, duplicates, memory, feature-type counts, quality score, ML readiness score).
5. Preview table: paginate forward/back, sort a column both directions, search for a value, hide/show a column (reload the page and confirm the hidden column stays hidden), confirm missing cells render distinctly and any duplicate rows are bordered.
6. Click a column's profile icon — the Column Explorer panel opens with a histogram (numeric) or top values (categorical), and either ML insights or the "run further into the pipeline" fallback message.
7. Column Summary / Correlations (all 4 methods) / Missing Values / Outliers (all 4 methods) sub-tabs each load without console errors.
8. Confirm no regressions: the existing run dashboard (Report/Test the model/AI Assistant) still works exactly as before.

- [ ] **Step 3: Fix any issues found, re-run steps 1-2 until clean**

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: address issues found in Dataset Preview manual verification"
```
