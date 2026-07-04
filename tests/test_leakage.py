import numpy as np
import pandas as pd

from src.profiling.leakage import detect_target_leakage


def test_flags_near_perfect_numeric_correlation():
    target = np.arange(200, dtype=float)
    df = pd.DataFrame({"target": target, "leaky_copy": target * 2 + 0.001, "noise": np.random.default_rng(0).normal(size=200)})
    flags = detect_target_leakage(df, "target")
    flagged_cols = {f["column"] for f in flags}
    assert "leaky_copy" in flagged_cols
    assert "noise" not in flagged_cols


def test_flags_suspicious_column_name():
    df = pd.DataFrame({"target": [0, 1, 0, 1], "post_outcome_flag": [0, 1, 0, 1], "amount": [1, 2, 3, 4]})
    flags = detect_target_leakage(df, "target")
    flagged_cols = {f["column"] for f in flags}
    assert "post_outcome_flag" in flagged_cols


def test_flags_categorical_column_that_maps_1to1_to_target():
    rng = np.random.default_rng(0)
    categories = [f"cat_{i}" for i in range(100)]
    targets = rng.integers(0, 2, 100)
    df = pd.DataFrame({"target": targets, "leaky_category": categories, "amount": rng.normal(size=100)})
    flags = detect_target_leakage(df, "target")
    flagged_cols = {f["column"] for f in flags}
    assert "leaky_category" in flagged_cols
