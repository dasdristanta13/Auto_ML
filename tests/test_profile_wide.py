"""PRD FR-6: wide datasets (500+ columns) must profile via clustering/summarization
rather than exhaustive per-column output."""

import numpy as np
import pandas as pd

from src.profiling.profile import WIDE_DATASET_COLUMN_THRESHOLD, profile_dataset


def test_wide_dataset_uses_clustered_summary_not_per_column_detail():
    rng = np.random.default_rng(0)
    n_cols = WIDE_DATASET_COLUMN_THRESHOLD + 20
    df = pd.DataFrame({f"feature_{i}": rng.normal(size=50) for i in range(n_cols)})

    profile = profile_dataset(df)

    assert profile["is_wide_dataset"] is True
    assert "numeric_clusters" in profile["numeric_summary"]
    # clustered summary must not enumerate full per-column detail for every numeric column
    assert len(profile["columns"]) < n_cols


def test_narrow_dataset_gets_full_per_column_detail():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    profile = profile_dataset(df)
    assert profile["is_wide_dataset"] is False
    assert set(profile["columns"].keys()) == {"a", "b"}
