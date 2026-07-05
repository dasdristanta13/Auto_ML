"""API integration tests for the AI Assistant chat endpoint (see
docs/superpowers/specs/2026-07-05-ai-assistant-chat-design.md). Runs the
whole intake -> confirm -> approve-features -> train flow through the real
HTTP layer with AUTOML_MOCK_LLM=1 so no API keys/network are needed."""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import server


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("AUTOML_MOCK_LLM", "1")
    return TestClient(server.app)


def _small_csv_bytes() -> bytes:
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "tenure_months": rng.normal(12, 5, n),
            "monthly_spend": rng.normal(60, 15, n),
            "churned": rng.choice([0, 1], n, p=[0.8, 0.2]),
        }
    )
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _wait_for_status(client, run_id, statuses, timeout=30.0):
    deadline = time.monotonic() + timeout
    run = None
    while time.monotonic() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in statuses:
            return run
        time.sleep(0.2)
    raise TimeoutError(f"run {run_id} never reached {statuses}, last status was {run['status'] if run else '?'}")


def _create_run(client) -> str:
    res = client.post(
        "/api/runs",
        files={"file": ("churn.csv", _small_csv_bytes(), "text/csv")},
        data={"description": "predict which customers will churn"},
    )
    assert res.status_code == 200
    return res.json()["run_id"]


def test_chat_before_report_ready_returns_409(client):
    run_id = _create_run(client)

    res = client.post(f"/api/runs/{run_id}/chat", json={"question": "why?"})

    assert res.status_code == 409


def test_chat_round_trip_after_report_ready(client):
    run_id = _create_run(client)
    _wait_for_status(client, run_id, {"awaiting_confirmation"})

    confirm = client.post(
        f"/api/runs/{run_id}/confirm",
        json={
            "target_column": "churned",
            "task_type": "classification",
            "metric": "f1",
            "cv_enabled": False,
            "cv_folds": 2,
            "tuning_enabled": False,
        },
    )
    assert confirm.status_code == 200
    _wait_for_status(client, run_id, {"awaiting_feature_approval", "failed"})

    approve = client.post(
        f"/api/runs/{run_id}/approve-features",
        json={"approved_step_indices": [], "resampling_enabled": False, "resampling_method": "none"},
    )
    assert approve.status_code == 200
    run = _wait_for_status(client, run_id, {"completed", "failed"})
    assert run["status"] == "completed", run.get("errors")

    assert run["suggested_questions"]
    assert run["chat_history"] == []

    first = client.post(f"/api/runs/{run_id}/chat", json={"question": "why was this model chosen?"})
    assert first.status_code == 200
    answer_1 = first.json()["answer"]
    assert answer_1

    second = client.post(f"/api/runs/{run_id}/chat", json={"question": "what about caveats?"})
    assert second.status_code == 200

    run = client.get(f"/api/runs/{run_id}").json()
    history = run["chat_history"]
    assert [h["role"] for h in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == "why was this model chosen?"
    assert history[1]["content"] == answer_1
    assert history[2]["content"] == "what about caveats?"
