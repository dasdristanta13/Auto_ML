"""POST /api/runs/{id}/experiments — new experiment on an existing run's
dataset without re-upload (docs/superpowers/specs/2026-07-06-multi-experiment-
design.md): reuses the dataset file and profile, records source_run_id, and
fails clearly when the source run or its dataset file is gone."""

from __future__ import annotations

import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import server
from src.graph.nodes import profile_node
from src.state import new_state


@pytest.fixture()
def no_intake_thread(monkeypatch):
    """The endpoint spawns the real intake graph on a thread — replace it with
    a no-op so tests assert on the seeded state, not LLM behavior."""
    started: list[str] = []
    monkeypatch.setattr(server, "_run_intake", lambda run_id: started.append(run_id))
    return started


def _seed_source_run(monkeypatch, tmp_path, with_profile: bool = True) -> str:
    csv_path = tmp_path / "source.csv"
    pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]}).to_csv(csv_path, index=False)
    state = new_state(run_id="src-run", dataset_path=str(csv_path), use_case_description="predict y")
    if with_profile:
        state["profile"] = {"row_count": 4, "column_count": 2, "columns": {"x": {}, "y": {}}}
    monkeypatch.setitem(
        server._runs,
        "src-run",
        {
            "state": state,
            "status": "completed",
            "events": [],
            "filename": "source.csv",
            "created_at": time.time(),
            "finished_at": time.time(),
            "cancel_requested": False,
            "chat_history": [],
        },
    )
    return "src-run"


def test_experiment_reuses_dataset_and_profile(monkeypatch, tmp_path, no_intake_thread):
    client = TestClient(server.app)
    source_id = _seed_source_run(monkeypatch, tmp_path)

    res = client.post(f"/api/runs/{source_id}/experiments", json={"description": "predict x instead"})
    assert res.status_code == 200
    new_id = res.json()["run_id"]
    assert new_id != source_id
    assert res.json()["status"] == "profiling"

    entry = server._runs[new_id]
    source_state = server._runs[source_id]["state"]
    assert entry["state"]["dataset_path"] == source_state["dataset_path"]
    assert entry["state"]["profile"] == source_state["profile"]
    assert entry["state"]["use_case_description"] == "predict x instead"
    assert entry["source_run_id"] == source_id
    assert entry["filename"] == "source.csv"
    assert no_intake_thread == [new_id]

    server._runs.pop(new_id, None)


def test_experiment_unknown_source_is_404(no_intake_thread):
    client = TestClient(server.app)
    res = client.post("/api/runs/nope/experiments", json={"description": "anything"})
    assert res.status_code == 404


def test_experiment_missing_dataset_file_is_409(monkeypatch, tmp_path, no_intake_thread):
    client = TestClient(server.app)
    source_id = _seed_source_run(monkeypatch, tmp_path)
    (tmp_path / "source.csv").unlink()

    res = client.post(f"/api/runs/{source_id}/experiments", json={"description": "anything"})
    assert res.status_code == 409
    assert "no longer available" in res.json()["detail"]


def test_experiment_blank_description_is_400(monkeypatch, tmp_path, no_intake_thread):
    client = TestClient(server.app)
    source_id = _seed_source_run(monkeypatch, tmp_path)

    res = client.post(f"/api/runs/{source_id}/experiments", json={"description": "   "})
    assert res.status_code == 400


def test_experiment_without_source_profile_falls_back(monkeypatch, tmp_path, no_intake_thread):
    """Source still mid-intake (no profile yet): the experiment starts anyway
    and will profile the file itself."""
    client = TestClient(server.app)
    source_id = _seed_source_run(monkeypatch, tmp_path, with_profile=False)

    res = client.post(f"/api/runs/{source_id}/experiments", json={"description": "new goal"})
    assert res.status_code == 200
    new_id = res.json()["run_id"]
    assert not server._runs[new_id]["state"].get("profile")
    server._runs.pop(new_id, None)


def test_list_and_detail_expose_source_run_id(monkeypatch, tmp_path, no_intake_thread):
    client = TestClient(server.app)
    source_id = _seed_source_run(monkeypatch, tmp_path)
    new_id = client.post(f"/api/runs/{source_id}/experiments", json={"description": "again"}).json()["run_id"]

    runs = {r["run_id"]: r for r in client.get("/api/runs").json()}
    assert runs[new_id]["source_run_id"] == source_id
    assert runs[source_id]["source_run_id"] is None

    detail = client.get(f"/api/runs/{new_id}").json()
    assert detail["source_run_id"] == source_id

    server._runs.pop(new_id, None)


def test_profile_node_skips_when_profile_present(tmp_path):
    """The reused profile must short-circuit re-profiling — dataset_path
    deliberately points at a nonexistent file so any read would raise."""
    state = new_state(run_id="r", dataset_path=str(tmp_path / "missing.csv"), use_case_description="d")
    state["profile"] = {"row_count": 4, "columns": {}}
    out = profile_node(state)
    assert out["profile"] == {"row_count": 4, "columns": {}}
