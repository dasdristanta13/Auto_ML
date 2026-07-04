"""Unit tests for the deterministic EDA module (src/profiling/eda.py) and the
leakage-safe resampling helper (src/training/dispatch._build_resampler).

These are pure-function tests against real fixtures — no LLM, no server, no
graph — so they pin down the rule-based suggestion logic and the SMOTE
fallback behavior independently of the rest of the pipeline."""

from __future__ import annotations

import pandas as pd

from src.profiling.eda import run_eda
from src.profiling.profile import profile_dataset
from src.training.dispatch import _build_resampler


def test_eda_flags_identifier_and_imbalance_on_imbalanced_fixture():
    df = pd.read_csv("tests/fixtures/imbalanced_classification.csv")
    profile = profile_dataset(df)
    task_spec = {"target_column": "churned", "task_type": "classification", "metric": "f1"}

    result = run_eda(df, profile, task_spec)

    assert result["resampling_suggestion"]["suggested"] is True
    assert result["resampling_suggestion"]["method"] == "smote"

    drop_targets = {
        col for step in result["suggested_steps"] if step["op"] == "drop" for col in step["columns"]
    }
    assert "customer_id" in drop_targets


def test_eda_flags_identifier_and_high_cardinality_on_categorical_fixture():
    df = pd.read_csv("tests/fixtures/high_cardinality_categorical.csv")
    profile = profile_dataset(df)
    task_spec = {"target_column": "amount", "task_type": "regression", "metric": "rmse"}

    result = run_eda(df, profile, task_spec)

    drop_targets = {
        col for step in result["suggested_steps"] if step["op"] == "drop" for col in step["columns"]
    }
    assert "transaction_id" in drop_targets

    encode_targets = {
        col for step in result["suggested_steps"] if step["op"] == "encode" for col in step["columns"]
    }
    assert "merchant_id" in encode_targets

    # regression task: no resampling suggestion regardless of any column's distribution
    assert result["resampling_suggestion"]["suggested"] is False


def test_build_resampler_falls_back_from_smote_when_minority_class_too_small():
    y = pd.Series([0] * 99 + [1] * 1)
    resampler, applied, note = _build_resampler("smote", y)

    assert applied == "random_oversample"
    assert note and "SMOTE" in note
    assert type(resampler).__name__ == "RandomOverSampler"


def test_build_resampler_uses_smote_when_minority_class_is_large_enough():
    y = pd.Series([0] * 80 + [1] * 20)
    resampler, applied, note = _build_resampler("smote", y)

    assert applied == "smote"
    assert note is None
    assert type(resampler).__name__ == "SMOTE"
