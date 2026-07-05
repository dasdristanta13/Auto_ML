"""PRD FR-6: wide datasets (500+ columns) must profile via clustering/summarization
rather than exhaustive per-column output — but every column must still be
*present* in profile["columns"] with basic info, because the confirm endpoint
validates the target against it and the UI builds the target picker from it."""

import numpy as np
import pandas as pd

from src.profiling.heuristics import minority_ratio
from src.profiling.profile import WIDE_DATASET_COLUMN_THRESHOLD, profile_dataset


def test_wide_dataset_uses_clustered_summary_not_per_column_detail():
    rng = np.random.default_rng(0)
    n_cols = WIDE_DATASET_COLUMN_THRESHOLD + 20
    df = pd.DataFrame({f"feature_{i}": rng.normal(size=50) for i in range(n_cols)})

    profile = profile_dataset(df)

    assert profile["is_wide_dataset"] is True
    assert "numeric_clusters" in profile["numeric_summary"]
    # clustered summary replaces exhaustive per-column numeric stats
    assert all("numeric_summary" not in info for info in profile["columns"].values())


def test_wide_dataset_still_lists_every_column_with_basic_info():
    """Regression: numeric columns used to be omitted from profile["columns"]
    entirely in wide mode, which made a numeric target unselectable in the UI
    and rejected by the confirm endpoint's membership check."""
    rng = np.random.default_rng(0)
    n_cols = WIDE_DATASET_COLUMN_THRESHOLD + 5
    data = {f"feature_{i}": rng.normal(size=50) for i in range(n_cols)}
    data["segment"] = rng.choice(["a", "b"], size=50)
    data["target"] = rng.choice([0, 1], size=50, p=[0.9, 0.1])
    df = pd.DataFrame(data)

    profile = profile_dataset(df)

    assert profile["is_wide_dataset"] is True
    assert set(profile["columns"].keys()) == set(df.columns)
    target_info = profile["columns"]["target"]
    assert target_info["dtype"] == str(df["target"].dtype)
    assert target_info["n_unique"] == 2


def test_wide_dataset_low_cardinality_numeric_column_supports_imbalance_detection():
    """A binary numeric target in a wide dataset must carry enough info
    (top_values) for minority_ratio / the resampling suggestion to work."""
    rng = np.random.default_rng(1)
    n_cols = WIDE_DATASET_COLUMN_THRESHOLD + 5
    data = {f"feature_{i}": rng.normal(size=200) for i in range(n_cols)}
    data["target"] = np.array([1] * 10 + [0] * 190)
    df = pd.DataFrame(data)

    profile = profile_dataset(df)

    ratio = minority_ratio(profile["columns"]["target"])
    assert ratio is not None
    assert abs(ratio - 0.05) < 1e-9


def test_narrow_dataset_gets_full_per_column_detail():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    profile = profile_dataset(df)
    assert profile["is_wide_dataset"] is False
    assert set(profile["columns"].keys()) == {"a", "b"}
