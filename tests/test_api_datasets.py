"""GET /api/datasets lists top-level runs (no source_run_id) as 'datasets' —
a re-run experiment reuses its source's file and isn't a separate dataset
(docs/superpowers/specs/2026-07-06-dataset-preview-design.md)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from src.api import server


def _entry(filename: str, source_run_id: str | None = None) -> dict:
    now = time.time()
    return {
        "state": {"profile": {"row_count": 100, "column_count": 5, "quality": {"overall": 0.9}}},
        "status": "completed",
        "events": [],
        "filename": filename,
        "created_at": now,
        "finished_at": now,
        "cancel_requested": False,
        "chat_history": [],
        **({"source_run_id": source_run_id} if source_run_id else {}),
    }


def test_list_datasets_excludes_rerun_experiments(monkeypatch):
    client = TestClient(server.app)
    monkeypatch.setitem(server._runs, "top-level-run", _entry("churn.csv"))
    monkeypatch.setitem(server._runs, "rerun-run", _entry("churn.csv", source_run_id="top-level-run"))

    datasets = client.get("/api/datasets").json()
    run_ids = {d["run_id"] for d in datasets}
    assert "top-level-run" in run_ids
    assert "rerun-run" not in run_ids


def test_list_datasets_includes_row_and_quality_info(monkeypatch):
    client = TestClient(server.app)
    monkeypatch.setitem(server._runs, "ds-1", _entry("sales.csv"))

    dataset = next(d for d in client.get("/api/datasets").json() if d["run_id"] == "ds-1")
    assert dataset["row_count"] == 100
    assert dataset["column_count"] == 5
    assert dataset["quality_score"] == 0.9
