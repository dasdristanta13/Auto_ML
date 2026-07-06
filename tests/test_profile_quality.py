"""profile_dataset must expose a deterministic quality block (completeness /
duplicates / uniqueness) for the dashboard's data-quality panel. Aggregates
only — no raw values (CLAUDE.md rule)."""

from __future__ import annotations

import pandas as pd

from src.profiling.profile import profile_dataset


def test_quality_block_reports_duplicates_and_completeness():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 1, None],  # null rate 0.2
            "b": ["x", "y", "z", "x", "y"],  # rows 0 and 3 duplicate each other
        }
    )
    quality = profile_dataset(df)["quality"]
    assert quality["duplicate_row_count"] == 1
    assert quality["duplicate_row_rate"] == 0.2
    assert quality["uniqueness"] == 0.8
    assert quality["completeness"] == 0.9  # 1 - mean null rate (0.2 + 0.0) / 2
    assert quality["overall"] == round((0.9 + 0.8) / 2, 4)


def test_quality_block_perfect_dataset():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    quality = profile_dataset(df)["quality"]
    assert quality["duplicate_row_count"] == 0
    assert quality["uniqueness"] == 1.0
    assert quality["completeness"] == 1.0
    assert quality["overall"] == 1.0


def test_profile_reports_memory_bytes():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    profile = profile_dataset(df)
    assert profile["memory_bytes"] > 0
    assert profile["memory_bytes"] == int(df.memory_usage(deep=True).sum())
