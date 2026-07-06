"""GET /api/runs/{id}/{correlations,missing-values,outliers,dataset-summary}
— the analytics sub-tabs and KPI row of the Dataset Preview Data tab."""

from __future__ import annotations

import time

import pandas as pd
from fastapi.testclient import TestClient

from src.api import server


def _make_run(tmp_path, monkeypatch):
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    dataset_path = tmp_path / "data.csv"
    df.to_csv(dataset_path, index=False)
    now = time.time()
    monkeypatch.setitem(
        server._runs,
        "run-1",
        {
            "state": {
                "dataset_path": str(dataset_path),
                "profile": {
                    "row_count": 5,
                    "column_count": 2,
                    "quality": {"completeness": 1.0, "uniqueness": 1.0},
                    "is_wide_dataset": False,
                },
                "leakage_flags": [],
            },
            "status": "completed",
            "events": [],
            "filename": "data.csv",
            "created_at": now,
            "finished_at": now,
            "cancel_requested": False,
            "chat_history": [],
        },
    )


def test_correlations_endpoint(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/correlations?method=pearson").json()
    assert result["columns"] == ["x", "y"]


def test_correlations_endpoint_rejects_unknown_method(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    res = client.get("/api/runs/run-1/correlations?method=bogus")
    assert res.status_code == 400


def test_missing_values_endpoint(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/missing-values").json()
    assert len(result["per_column"]) == 2


def test_outliers_endpoint(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/outliers?method=iqr").json()
    assert result["method"] == "iqr"


def test_dataset_summary_endpoint(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/dataset-summary").json()
    assert result["feature_type_counts"]["numeric"] == 2
    assert 0.0 <= result["ml_readiness_score"] <= 1.0
