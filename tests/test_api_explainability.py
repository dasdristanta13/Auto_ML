"""GET /api/runs/{id}/explainability (reads the precomputed SHAP summary) and
the /predict endpoint's new `contributions` field (computed on demand)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api import server
from src.training.dispatch import _registry, _run_job


def _make_run_with_model(tmp_path, monkeypatch, run_id="run-1", explainability=None):
    rng = np.random.default_rng(2)
    n = 150
    df = pd.DataFrame({"x1": rng.random(n), "x2": rng.random(n), "target": rng.integers(0, 2, n)})
    dataset_path = tmp_path / "data.csv"
    df.to_csv(dataset_path, index=False)

    tag = f"api-{run_id}"
    _registry[tag] = {"status": "pending", "feature_selection": {"enabled": False}}
    _run_job(
        tag, str(dataset_path), "target", "classification", "sklearn",
        "RandomForestClassifier", {"n_estimators": 10, "max_depth": 3, "random_state": 0},
        None, [], False, None, False, "none", False, None, None,
    )
    trained = _registry[tag]
    assert trained["status"] == "succeeded", trained.get("error")

    best_model = dict(trained)
    if explainability is not None:
        best_model["explainability"] = explainability

    now = time.time()
    monkeypatch.setitem(
        server._runs,
        run_id,
        {
            "state": {
                "dataset_path": str(dataset_path),
                "transformed_dataset_path": str(dataset_path),
                "best_model": best_model,
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
    return str(dataset_path), best_model


def test_get_explainability_returns_precomputed_summary(tmp_path, monkeypatch):
    canned = {
        "method": "tree",
        "feature_impact": [{"feature": "x1", "mean_abs_shap": 0.3}],
        "narrative": "x1 drives this model.",
        "note": None,
    }
    _make_run_with_model(tmp_path, monkeypatch, explainability=canned)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/explainability").json()
    assert result == canned


def test_get_explainability_default_when_not_yet_computed(tmp_path, monkeypatch):
    _make_run_with_model(tmp_path, monkeypatch, explainability=None)
    client = TestClient(server.app)
    result = client.get("/api/runs/run-1/explainability").json()
    assert result["method"] == "unavailable"
    assert result["feature_impact"] == []


def test_get_explainability_404_without_trained_model(monkeypatch):
    monkeypatch.setitem(
        server._runs,
        "run-2",
        {
            "state": {"best_model": {}}, "status": "running", "events": [], "filename": "d.csv",
            "created_at": time.time(), "finished_at": None, "cancel_requested": False, "chat_history": [],
        },
    )
    client = TestClient(server.app)
    res = client.get("/api/runs/run-2/explainability")
    assert res.status_code == 404


def test_predict_endpoint_includes_contributions(tmp_path, monkeypatch):
    _make_run_with_model(tmp_path, monkeypatch)
    client = TestClient(server.app)
    result = client.post("/api/runs/run-1/predict", json={"values": {"x1": 0.5, "x2": 0.5}}).json()
    assert "prediction" in result
    assert isinstance(result["contributions"], list) and result["contributions"]
