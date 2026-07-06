"""src/profiling/preview.py: deterministic, non-LLM helpers for the Dataset
Preview 'Data' tab. Unlike src/profiling/profile.py this module returns
row-level data — that's fine because it serves the human-facing UI, not an
LLM prompt (CLAUDE.md's raw-data rule is about the LLM boundary only). See
docs/superpowers/specs/2026-07-06-dataset-preview-design.md."""

from __future__ import annotations

import pandas as pd
import pytest

from src.profiling.preview import MAX_PAGE_SIZE, column_detail, correlation_matrix, paginate_rows, missing_value_matrix


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


def test_paginate_rejects_non_positive_page_size():
    with pytest.raises(ValueError):
        paginate_rows(_df(), page=1, page_size=0)


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


def test_column_detail_numeric_all_nan_does_not_fall_into_categorical_branch():
    df = pd.DataFrame({"amount": pd.Series([float("nan"), float("nan"), float("nan")], dtype="float64")})
    detail = column_detail(df, "amount")
    assert detail["is_numeric"] is True
    assert "top_values" not in detail


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


def test_correlation_matrix_mutual_info_degrades_gracefully_on_tiny_subset():
    # After dropna(), exactly 3 rows remain in common between x and y. That's
    # the actual crash window: the OLD guard (`len(subset) >= 2`) would have
    # let this through to mutual_info_regression, whose default n_neighbors=3
    # requires strictly more than 3 samples and raises. The NEW guard
    # (`len(subset) > 3`) correctly skips the computation instead.
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, None], "y": [10.0, 20.0, 30.0, None]})
    result = correlation_matrix(df, method="mutual_info")
    assert result["matrix"][0][0] == 1.0
    assert result["matrix"][1][1] == 1.0
    assert result["matrix"][0][1] == 0.0
    assert result["matrix"][1][0] == 0.0


def test_correlation_matrix_truncates_to_max_columns_by_variance():
    data = {f"col{i}": [i] * 5 for i in range(55)}
    # Give one column a much larger variance so it's guaranteed to survive truncation.
    data["high_var"] = [0, 1000, -1000, 500, -500]
    df = pd.DataFrame(data)
    result = correlation_matrix(df, method="pearson")
    assert result["truncated"] is True
    assert len(result["columns"]) == 50
    assert "high_var" in result["columns"]


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
