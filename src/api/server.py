"""FastAPI backend for the agentic AutoML platform.

Run flow (mirrors PRD 2.3 user journey):
  1. POST /api/runs            — upload CSV + describe use case; intake phase
                                 (profile -> understand_usecase) starts in a
                                 background thread.
  2. status "awaiting_confirmation" — the user ALWAYS confirms/corrects the
                                 inferred task spec in the UI before any
                                 compute-heavy work (PRD FR-9/FR-10); nothing
                                 is silently guessed.
  3. POST /api/runs/{id}/confirm — locks the task spec, main phase (leakage ->
                                 features -> training -> evaluate -> report)
                                 runs in a background thread.
  4. GET /api/runs/{id}        — poll status/progress; GET .../model downloads
                                 the trained artifact; GET .../trace returns
                                 the full LLM audit log (CLAUDE.md rule #7).

Local stand-in note: runs live in an in-memory dict guarded by a lock and
execute on daemon threads. Production would back this with Postgres + the
Celery/Ray job infra, per the architecture doc.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.graph.build_graph import build_intake_graph, build_main_graph
from src.llm.tracing import read_trace
from src.state import PipelineState, new_state

UPLOAD_DIR = Path("data/uploads")

app = FastAPI(title="Agentic AutoML")

_runs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()

_intake_graph = build_intake_graph()
_main_graph = build_main_graph()


def _json_safe(obj: Any) -> Any:
    """Profiles/metrics can contain numpy scalars; round-trip through json
    with a tolerant default so responses always serialize."""

    def _default(o: Any) -> Any:
        try:
            return float(o)
        except (TypeError, ValueError):
            return str(o)

    return json.loads(json.dumps(obj, default=_default))


def _record_event(entry: dict[str, Any], node: str) -> None:
    entry["events"].append({"node": node, "timestamp": time.time()})


def _stream_graph(run_id: str, graph, on_done_status) -> None:
    entry = _runs[run_id]
    try:
        for chunk in graph.stream(dict(entry["state"]), config={"recursion_limit": 500}, stream_mode="updates"):
            for node_name, update in chunk.items():
                with _lock:
                    if isinstance(update, dict):
                        entry["state"].update(update)
                    _record_event(entry, node_name)
    except Exception as exc:  # noqa: BLE001 - a failed run must end in a clear user-facing state, never a hang
        with _lock:
            entry["state"].setdefault("errors", []).append(str(exc))
            entry["status"] = "failed"
        return
    with _lock:
        entry["status"] = on_done_status(entry["state"])


def _run_intake(run_id: str) -> None:
    _stream_graph(run_id, _intake_graph, lambda state: "awaiting_confirmation")


def _run_main(run_id: str) -> None:
    _stream_graph(
        run_id,
        _main_graph,
        lambda state: "completed" if state.get("status") == "completed" else "failed",
    )


class ConfirmRequest(BaseModel):
    target_column: str
    task_type: str
    metric: str
    constraints: list[str] = []


def _get_entry(run_id: str) -> dict[str, Any]:
    entry = _runs.get(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown run '{run_id}'")
    return entry


def _profile_columns(state: PipelineState) -> list[dict[str, Any]]:
    columns = state.get("profile", {}).get("columns", {})
    return [
        {
            "name": name,
            "dtype": info.get("dtype"),
            "null_rate": info.get("null_rate"),
            "n_unique": info.get("n_unique"),
            "is_pii": info.get("is_pii", False),
        }
        for name, info in columns.items()
    ]


def _run_summary(run_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    state = entry["state"]
    stages_done: list[str] = []
    for event in entry["events"]:
        if event["node"] not in stages_done:
            stages_done.append(event["node"])
    return _json_safe(
        {
            "run_id": run_id,
            "filename": entry["filename"],
            "description": state.get("use_case_description"),
            "status": entry["status"],
            "created_at": entry["created_at"],
            "stages_done": stages_done,
            "task_spec": state.get("task_spec"),
            "profile_summary": {
                "row_count": state.get("profile", {}).get("row_count"),
                "column_count": state.get("profile", {}).get("column_count"),
                "pii_columns_detected": state.get("profile", {})
                .get("pii_report", {})
                .get("pii_columns_detected"),
            },
            "profile_columns": _profile_columns(state),
            "leakage_flags": state.get("leakage_flags", []),
            "feature_plan": state.get("feature_plan"),
            "training_results": state.get("training_results", []),
            "best_model": state.get("best_model"),
            "report": state.get("report", {}).get("narrative"),
            "errors": state.get("errors", []),
        }
    )


@app.post("/api/runs")
async def create_run(file: UploadFile = File(...), description: str = Form(...)) -> dict[str, Any]:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="only .csv uploads are supported in this local build")

    run_id = str(uuid.uuid4())[:8]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = UPLOAD_DIR / f"{run_id}.csv"
    dataset_path.write_bytes(await file.read())

    state = new_state(run_id=run_id, dataset_path=str(dataset_path), use_case_description=description)
    with _lock:
        _runs[run_id] = {
            "state": state,
            "status": "profiling",
            "events": [],
            "filename": file.filename,
            "created_at": time.time(),
        }

    threading.Thread(target=_run_intake, args=(run_id,), daemon=True).start()
    return {"run_id": run_id, "status": "profiling"}


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    with _lock:
        return [
            {
                "run_id": run_id,
                "filename": entry["filename"],
                "status": entry["status"],
                "created_at": entry["created_at"],
                "description": entry["state"].get("use_case_description"),
            }
            for run_id, entry in sorted(_runs.items(), key=lambda kv: -kv[1]["created_at"])
        ]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    entry = _get_entry(run_id)
    with _lock:
        return _run_summary(run_id, entry)


@app.post("/api/runs/{run_id}/confirm")
def confirm_run(run_id: str, body: ConfirmRequest) -> dict[str, Any]:
    entry = _get_entry(run_id)
    with _lock:
        if entry["status"] != "awaiting_confirmation":
            raise HTTPException(status_code=409, detail=f"run is '{entry['status']}', not awaiting confirmation")

        state = entry["state"]
        columns = state.get("profile", {}).get("columns", {})
        if columns and body.target_column not in columns:
            raise HTTPException(status_code=400, detail=f"'{body.target_column}' is not a column of this dataset")

        task_spec = dict(state.get("task_spec") or {})
        task_spec.update(
            target_column=body.target_column,
            task_type=body.task_type,
            metric=body.metric,
            constraints=body.constraints,
            is_ambiguous=False,
            ambiguity_reason=None,
        )
        state["task_spec"] = task_spec
        state["needs_human_confirmation"] = False
        state["human_confirmed"] = True
        entry["status"] = "running"

    threading.Thread(target=_run_main, args=(run_id,), daemon=True).start()
    return {"run_id": run_id, "status": "running"}


@app.get("/api/runs/{run_id}/model")
def download_model(run_id: str):
    entry = _get_entry(run_id)
    model_path = (entry["state"].get("best_model") or {}).get("model_path")
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="no trained model artifact for this run yet")
    return FileResponse(model_path, filename=f"automl_{run_id}.joblib", media_type="application/octet-stream")


@app.get("/api/runs/{run_id}/trace")
def get_trace(run_id: str) -> list[dict[str, Any]]:
    _get_entry(run_id)
    return _json_safe(read_trace(run_id))


# Serve the frontend last so /api/* wins routing.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
