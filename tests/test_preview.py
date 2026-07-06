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
