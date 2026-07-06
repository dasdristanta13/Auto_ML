"""GET /api/runs must expose best_score/metric so the home view can show
scores without fetching every run's detail, and _profile_columns must pass
top_values through for the class-distribution panel (see
docs/superpowers/specs/2026-07-05-mockup-parity-ui-design.md)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from src.api import server


def _entry(state: dict, status: str) -> dict:
    now = time.time()
    return {
        "state": state,
        "status": status,
        "events": [],
        "filename": "churn.csv",
        "created_at": now,
        "finished_at": now if status in ("completed", "failed") else None,
        "cancel_requested": False,
    }


def test_list_runs_includes_best_score_and_metric(monkeypatch):
    client = TestClient(server.app)
    state = {
        "use_case_description": "predict churn",
        "task_spec": {"metric": "f1"},
        "best_model": {"metrics": {"f1": 0.8421}},
    }
    monkeypatch.setitem(server._runs, "fake-run", _entry(state, "completed"))

    runs = client.get("/api/runs").json()
    fake = next(r for r in runs if r["run_id"] == "fake-run")
    assert fake["best_score"] == 0.8421
    assert fake["metric"] == "f1"


def test_list_runs_scores_null_before_completion(monkeypatch):
    client = TestClient(server.app)
    state = {"use_case_description": "predict churn"}
    monkeypatch.setitem(server._runs, "fake-run-2", _entry(state, "profiling"))

    runs = client.get("/api/runs").json()
    fake = next(r for r in runs if r["run_id"] == "fake-run-2")
    assert fake["best_score"] is None
    assert fake["metric"] is None


def test_profile_columns_include_top_values():
    state = {
        "profile": {
            "columns": {
                "churned": {
                    "dtype": "int64",
                    "null_rate": 0.0,
                    "n_unique": 2,
                    "is_pii": False,
                    "top_values": {"0": 90, "1": 10},
                },
                "tenure": {
                    "dtype": "float64",
                    "null_rate": 0.0,
                    "n_unique": 87,
                    "is_pii": False,
                },
            }
        }
    }
    cols = {c["name"]: c for c in server._profile_columns(state)}
    assert cols["churned"]["top_values"] == {"0": 90, "1": 10}
    assert cols["tenure"]["top_values"] is None


def test_run_summary_includes_memory_bytes(monkeypatch):
    client = TestClient(server.app)
    state = {"profile": {"row_count": 10, "column_count": 2, "memory_bytes": 4096}}
    monkeypatch.setitem(server._runs, "fake-run-3", _entry(state, "completed"))

    run = client.get("/api/runs/fake-run-3").json()
    assert run["profile_summary"]["memory_bytes"] == 4096
