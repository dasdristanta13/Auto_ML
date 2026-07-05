"""time_column must flow from the task spec all the way into the training
job, because src/training/dispatch._split only does the chronological
(leakage-safe) split when it receives one — before this plumbing existed the
chronological path was dead code and time-series data was shuffled randomly."""

from __future__ import annotations

import pandas as pd

from src.graph import nodes
from src.profiling.eda import run_eda
from src.profiling.profile import profile_dataset
from src.state import TaskSpec, new_state


class _CapturingTool:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def invoke(self, args: dict) -> str:
        self.calls.append(args)
        return f"run-{len(self.calls)}"


def test_task_spec_has_time_column_field():
    spec = TaskSpec(task_type="forecasting", target_column="sales", time_column="date")
    assert spec.time_column == "date"
    assert TaskSpec().time_column is None


def test_dispatch_training_forwards_time_column(monkeypatch):
    capture = _CapturingTool()
    monkeypatch.setattr(nodes, "train_model", capture)

    state = new_state(run_id="t", dataset_path="unused.csv", use_case_description="forecast sales")
    state["task_spec"] = {"task_type": "forecasting", "target_column": "sales", "metric": "rmse", "time_column": "date"}
    state["transformed_dataset_path"] = "unused.csv"
    state["candidate_models"] = [
        {"name": "rf", "library": "sklearn", "estimator": "RandomForestRegressor", "hyperparams": {}}
    ]

    nodes.dispatch_training_node(state)

    assert capture.calls, "no training job was dispatched"
    assert capture.calls[0]["time_column"] == "date"


def test_eda_does_not_suggest_consuming_the_time_column():
    """datetime_decompose drops the source column — applied to the designated
    time_column it would silently disable the chronological split downstream."""
    df = pd.read_csv("tests/fixtures/time_series.csv")
    profile = profile_dataset(df)
    task_spec = {"task_type": "forecasting", "target_column": "sales", "metric": "rmse", "time_column": "date"}

    result = run_eda(df, profile, task_spec)

    touched = {col for step in result["suggested_steps"] for col in step["columns"]}
    assert "date" not in touched
