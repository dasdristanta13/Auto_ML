# tests/test_api_preview.py
"""GET /api/runs/{id}/preview: paginated raw rows for the Dataset Preview
Data tab, plus which columns are PII (badged, not redacted — this is the
owning user viewing their own upload, not an LLM context; see
docs/superpowers/specs/2026-07-06-dataset-preview-design.md)."""

from __future__ import annotations

import time

import pandas as pd
from fastapi.testclient import TestClient

from src.api import server


def _make_run(tmp_path, monkeypatch):
    df = pd.DataFrame({"amount": range(10), "email": [f"user{i}@example.com" for i in range(10)]})
    dataset_path = tmp_path / "data.csv"
    df.to_csv(dataset_path, index=False)
    now = time.time()
    monkeypatch.setitem(
        server._runs,
        "run-1",
        {
            "state": {
                "dataset_path": str(dataset_path),
                "profile": {"pii_report": {"columns": {"email": {"pii_type": "email"}}}},
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


def test_preview_returns_page_of_rows(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/preview?page=1&page_size=5").json()
    assert result["total_count"] == 10
    assert len(result["rows"]) == 5
    assert result["pii_columns"] == ["email"]


def test_preview_rejects_oversized_page_size(tmp_path, monkeypatch):
    _make_run(tmp_path, monkeypatch)
    client = TestClient(server.app)
    res = client.get("/api/runs/run-1/preview?page=1&page_size=9999")
    assert res.status_code == 400


def test_preview_404s_for_unknown_run():
    client = TestClient(server.app)
    res = client.get("/api/runs/unknown-run/preview")
    assert res.status_code == 404
