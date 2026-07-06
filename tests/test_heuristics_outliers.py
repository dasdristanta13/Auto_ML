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
