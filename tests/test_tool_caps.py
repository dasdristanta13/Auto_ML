"""CLAUDE.md: any change to a tool function requires a unit test asserting it
never returns more than the capped row/sample limit."""

import pandas as pd
import yaml

from src.tools.data_tools import get_column_sample


def test_get_column_sample_never_exceeds_configured_cap(tmp_path):
    with open("config/runtime.yaml", "r", encoding="utf-8") as f:
        cap = yaml.safe_load(f)["tools"]["max_sample_rows"]

    df = pd.DataFrame({"value": range(1000)})
    dataset_path = tmp_path / "data.csv"
    df.to_csv(dataset_path, index=False)

    result = get_column_sample.invoke({"dataset_path": str(dataset_path), "column": "value", "n": 10_000})

    assert len(result) <= cap


def test_get_column_sample_redacts_pii_columns(tmp_path):
    df = pd.DataFrame({"email": [f"user{i}@example.com" for i in range(20)]})
    dataset_path = tmp_path / "data.csv"
    df.to_csv(dataset_path, index=False)

    result = get_column_sample.invoke({"dataset_path": str(dataset_path), "column": "email", "n": 5})

    assert all(value == "[REDACTED]" for value in result)
