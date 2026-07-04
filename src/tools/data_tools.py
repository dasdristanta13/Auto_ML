"""Typed tools exposed to LLM-backed agent nodes.

Every tool here is capped at config/runtime.yaml -> tools.max_sample_rows
(<= 20 rows) ENFORCED IN CODE, not by convention (CLAUDE.md rule + tests/
test_tool_caps.py asserts this). Docstrings are what the LLM sees, so they
must describe exactly what is returned.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import yaml
from langchain_core.tools import tool

from src.pii.redact import redact_dataframe
from src.profiling.leakage import detect_target_leakage
from src.profiling.profile import profile_dataset

_RUNTIME_CONFIG_PATH = "config/runtime.yaml"


def _max_sample_rows() -> int:
    with open(_RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["tools"]["max_sample_rows"]


def _load(dataset_path: str) -> pd.DataFrame:
    if dataset_path.endswith(".parquet"):
        return pd.read_parquet(dataset_path)
    if dataset_path.endswith(".json"):
        return pd.read_json(dataset_path)
    return pd.read_csv(dataset_path)


@tool
def get_dataset_profile(dataset_path: str) -> dict[str, Any]:
    """Return the deterministic statistical profile of the dataset at dataset_path:
    schema, dtypes, null rates, cardinality, numeric summaries, top categorical
    values, correlations (or correlation clusters for wide datasets), and a
    PII report. All PII-flagged columns and free-text values are pre-redacted.
    Never returns more than 5 raw sample rows, all PII-redacted.
    """
    df = _load(dataset_path)
    return profile_dataset(df)


@tool
def get_column_sample(dataset_path: str, column: str, n: int = 10) -> list[Any]:
    """Return up to `n` (hard-capped, see config/runtime.yaml tools.max_sample_rows)
    non-null values from a single column, after PII redaction. If the column is
    flagged as PII, every returned value is the literal string "[REDACTED]".
    Use this only to inspect a specific column's value distribution more closely
    than the profile provides — never to reconstruct row-level records.
    """
    df = _load(dataset_path)
    redacted, _ = redact_dataframe(df)
    capped_n = min(n, _max_sample_rows())
    if column not in redacted.columns:
        raise ValueError(f"column '{column}' not found in dataset")
    return redacted[column].dropna().head(capped_n).tolist()


@tool
def check_target_leakage(dataset_path: str, target_column: str) -> list[dict[str, Any]]:
    """Return a best-effort (NOT guaranteed complete) list of columns that may
    leak information about the target: {column, reason, severity}. False
    negatives are expected; this must be presented to the user as a heuristic
    flag, not a completeness guarantee.
    """
    df = _load(dataset_path)
    return detect_target_leakage(df, target_column)
