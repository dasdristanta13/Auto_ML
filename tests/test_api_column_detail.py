"""GET /api/runs/{id}/columns/{name}: column-level stats for the Column
Explorer panel, plus already-computed EDA/leakage insights for that column
when the run has progressed far enough to have them."""

from __future__ import annotations

import time

import pandas as pd
from fastapi.testclient import TestClient

from src.api import server


def _make_run(tmp_path, monkeypatch, extra_state=None):
    df = pd.DataFrame({"amount": [1, 2, 3, 4, 5]})
    dataset_path = tmp_path / "data.csv"
    df.to_csv(dataset_path, index=False)
    now = time.time()
    state = {"dataset_path": str(dataset_path)}
    state.update(extra_state or {})
    monkeypatch.setitem(
        server._runs,
        "run-1",
        {
            "state": state,
            "status": "completed",
            "events": [],
            "filename": "data.csv",
            "created_at": now,
            "finished_at": now,
            "cancel_requested": False,
            "chat_history": [],
        },
    )


def test_column_detail_returns_stats(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/columns/amount").json()
    assert result["is_numeric"] is True
    assert result["ml_insights"] == {"analyzed": False}


def test_column_detail_includes_eda_recommendation_when_present(tmp_path, monkeypatch):
    _make_run(
        tmp_path,
        monkeypatch,
        extra_state={
            "feature_plan": {"steps": [{"op": "scale", "columns": ["amount"], "rationale": "wide range"}]},
        },
    )
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/columns/amount").json()
    assert result["ml_insights"]["analyzed"] is True
    assert result["ml_insights"]["recommended_steps"][0]["op"] == "scale"


def test_column_detail_404s_for_unknown_column(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    res = client.get("/api/runs/run-1/columns/nope")
    assert res.status_code == 404
