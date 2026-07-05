"""A classification target with near-one-unique-value-per-row (e.g. an ID or
free-text column picked by mistake) causes every candidate to fail or return
meaningless metrics: sklearn models "succeed" with ~0.0 accuracy because most
test-set classes were never seen in training, and XGBoost fails outright
because its sklearn wrapper requires the label range to be exactly contiguous
(0..n-1), which a random split can't guarantee once cardinality is this high.
POST /api/runs/{id}/confirm must reject this at the human checkpoint with a
clear message rather than letting the pipeline burn through every candidate.
See docs/superpowers/specs (chat/mockup specs) sibling investigation."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from src.profiling.heuristics import target_too_high_cardinality_for_classification
from src.api import server


def test_ratio_at_or_below_threshold_is_not_flagged():
    assert target_too_high_cardinality_for_classification(n_unique=2, row_count=1000) is False
    assert target_too_high_cardinality_for_classification(n_unique=500, row_count=1000) is False  # exactly 0.5


def test_ratio_above_threshold_is_flagged():
    assert target_too_high_cardinality_for_classification(n_unique=1598, row_count=1600) is True


def test_zero_row_count_is_not_flagged():
    assert target_too_high_cardinality_for_classification(n_unique=0, row_count=0) is False


def _entry(profile: dict) -> dict:
    now = time.time()
    return {
        "state": {"profile": profile, "task_spec": {}},
        "status": "awaiting_confirmation",
        "events": [],
        "filename": "data.csv",
        "created_at": now,
        "finished_at": None,
        "cancel_requested": False,
        "chat_history": [],
    }


def test_confirm_rejects_high_cardinality_classification_target(monkeypatch):
    client = TestClient(server.app)
    profile = {
        "row_count": 1600,
        "columns": {
            "customer_id": {"dtype": "int64", "null_rate": 0.0, "n_unique": 1598, "is_pii": False},
        },
    }
    monkeypatch.setitem(server._runs, "fake-run-hc", _entry(profile))

    res = client.post(
        "/api/runs/fake-run-hc/confirm",
        json={"target_column": "customer_id", "task_type": "classification", "metric": "f1"},
    )

    assert res.status_code == 400
    assert "customer_id" in res.json()["detail"]


def test_confirm_allows_low_cardinality_classification_target(monkeypatch):
    client = TestClient(server.app)
    profile = {
        "row_count": 1600,
        "columns": {
            "churned": {"dtype": "int64", "null_rate": 0.0, "n_unique": 2, "is_pii": False},
        },
    }
    monkeypatch.setitem(server._runs, "fake-run-lc", _entry(profile))

    res = client.post(
        "/api/runs/fake-run-lc/confirm",
        json={"target_column": "churned", "task_type": "classification", "metric": "f1"},
    )

    assert res.status_code == 200


def test_confirm_does_not_guard_regression_targets(monkeypatch):
    client = TestClient(server.app)
    profile = {
        "row_count": 1600,
        "columns": {
            "price": {"dtype": "float64", "null_rate": 0.0, "n_unique": 1598, "is_pii": False},
        },
    }
    monkeypatch.setitem(server._runs, "fake-run-reg", _entry(profile))

    res = client.post(
        "/api/runs/fake-run-reg/confirm",
        json={"target_column": "price", "task_type": "regression", "metric": "rmse"},
    )

    assert res.status_code == 200
