"""High-severity leakage flags must become approve-able drop suggestions in
the EDA step list (docs/superpowers/specs/2026-07-06-eda-drops-and-rfe-
design.md); medium-severity name hints stay display-only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.profiling.eda import run_eda
from src.profiling.leakage import detect_target_leakage


def _df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    target = rng.random(n) * 100
    return pd.DataFrame(
        {
            "leaky": target * 1.001 + rng.random(n) * 0.01,  # near-perfect correlation
            "honest": rng.random(n),
            "post_outcome_flag": rng.random(n),  # medium: name hint only
            "target": target,
        }
    )


def _task_spec() -> dict:
    return {"target_column": "target", "task_type": "regression", "metric": "rmse"}


def test_high_severity_flag_becomes_drop_suggestion():
    df = _df()
    flags = detect_target_leakage(df, "target")
    assert any(f["column"] == "leaky" and f["severity"] == "high" for f in flags)

    result = run_eda(df, {"columns": {}}, _task_spec(), flags)
    drops = [s for s in result["suggested_steps"] if s["op"] == "drop"]
    leaky_drop = next((s for s in drops if s["columns"] == ["leaky"]), None)
    assert leaky_drop is not None
    assert "leakage" in leaky_drop["rationale"].lower()
    assert leaky_drop["source"] == "eda"


def test_medium_severity_name_hint_is_not_dropped():
    df = _df()
    flags = detect_target_leakage(df, "target")
    result = run_eda(df, {"columns": {}}, _task_spec(), flags)
    drops = [s for s in result["suggested_steps"] if s["op"] == "drop"]
    assert not any("post_outcome_flag" in s["columns"] for s in drops)


def test_leakage_dropped_column_gets_no_other_suggestions():
    df = _df()
    df.loc[df.index[:10], "leaky"] = np.nan  # would normally earn an impute suggestion
    flags = [{"column": "leaky", "reason": "near-perfect correlation with target (0.999)", "severity": "high"}]
    result = run_eda(df, {"columns": {}}, _task_spec(), flags)
    steps_for_leaky = [s for s in result["suggested_steps"] if "leaky" in s["columns"]]
    assert [s["op"] for s in steps_for_leaky] == ["drop"]


def test_target_and_time_columns_never_drop_suggested():
    df = _df()
    flags = [
        {"column": "target", "reason": "x", "severity": "high"},
        {"column": "honest", "reason": "x", "severity": "high"},
    ]
    spec = {**_task_spec(), "time_column": "honest"}
    result = run_eda(df, {"columns": {}}, spec, flags)
    drops = [s for s in result["suggested_steps"] if s["op"] == "drop"]
    assert not any(set(s["columns"]) & {"target", "honest"} for s in drops)


def test_no_flags_no_drop_regression():
    df = _df()[["honest", "target"]]
    result = run_eda(df, {"columns": {}}, _task_spec(), [])
    assert not any(s["op"] == "drop" for s in result["suggested_steps"])


def test_wide_data_recommends_rfe_insight():
    rng = np.random.default_rng(1)
    wide = pd.DataFrame({f"f{i}": rng.random(50) for i in range(16)})
    wide["target"] = rng.random(50)
    result = run_eda(wide, {"columns": {}}, _task_spec(), [])
    assert any(i["id"] == "rfe_recommended" for i in result["insights"])

    narrow = _df()
    result = run_eda(narrow, {"columns": {}}, _task_spec(), [])
    assert not any(i["id"] == "rfe_recommended" for i in result["insights"])
