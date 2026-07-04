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
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.export.script_export import generate_training_script
from src.graph.build_graph import build_intake_graph, build_main_graph
from src.insights.auto_insights import generate_insights
from src.llm.tracing import read_trace
from src.state import PipelineState, new_state
from src.training.dispatch import load_model_schema, predict_one

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
            with _lock:
                if entry.get("cancel_requested"):
                    entry["status"] = "cancelled"
                    entry["finished_at"] = time.time()
                    return
    except Exception as exc:  # noqa: BLE001 - a failed run must end in a clear user-facing state, never a hang
        with _lock:
            entry["state"].setdefault("errors", []).append(str(exc))
            entry["status"] = "failed"
            entry["finished_at"] = time.time()
        return
    with _lock:
        entry["status"] = on_done_status(entry["state"])
        entry["finished_at"] = time.time()


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


_STAGE_MESSAGES = {
    "profile": "Profiled the dataset — schema, null rates, cardinality, and PII scan complete.",
    "understand_usecase": "Interpreted the use case into a task specification.",
    "confirm": "Task specification confirmed — compute-heavy work unlocked.",
    "leakage_check": "Checked for columns that may leak information about the target.",
    "feature_engineering": "Planned feature transformations (imputation, encoding, scaling).",
    "apply_feature_plan": "Applied the feature plan to the dataset.",
    "model_selection": "Selected candidate models suited to this task and data.",
    "dispatch_training": "Dispatched training jobs for each candidate model.",
    "poll_training": "Training in progress.",
    "evaluate": "Evaluated all candidates and selected the best model.",
    "report": "Wrote the final report.",
}


def _stage_timeline(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-stage completion time + duration, deduped: repeated nodes (the
    poll_training loop) keep their first position but their latest timestamp.
    The human confirmation checkpoint is its own recorded event, so waiting-
    on-the-user time is attributed to 'confirm', not to the next stage."""
    timeline: list[dict[str, Any]] = []
    by_node: dict[str, dict[str, Any]] = {}
    for event in entry["events"]:
        node, timestamp = event["node"], event["timestamp"]
        if node in by_node:
            by_node[node]["completed_at"] = timestamp
        else:
            record = {"node": node, "completed_at": timestamp}
            by_node[node] = record
            timeline.append(record)
    previous = entry["created_at"]
    for record in timeline:
        record["duration_seconds"] = round(max(record["completed_at"] - previous, 0), 1)
        previous = record["completed_at"]
    return timeline


def _plain_language_events(state: PipelineState, stages_done: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for stage in stages_done:
        message = _STAGE_MESSAGES.get(stage, stage)
        if stage == "profile":
            profile = state.get("profile", {})
            pii_n = profile.get("pii_report", {}).get("pii_columns_detected", 0)
            if pii_n:
                message += f" Redacted {pii_n} PII column(s) before any AI-facing step."
        elif stage == "leakage_check":
            flags = state.get("leakage_flags", [])
            message = (
                f"Flagged {len(flags)} possible target-leakage column(s) — heuristic, not guaranteed complete."
                if flags
                else "No target-leakage signals detected (heuristic — not guaranteed complete)."
            )
        elif stage == "dispatch_training":
            n = len(state.get("candidate_models", []))
            message = f"Dispatched {n} training job(s) — running asynchronously."
        elif stage == "evaluate":
            best = state.get("best_model", {})
            if best.get("candidate_name"):
                message = f"'{best['candidate_name']}' selected as the best model."
        events.append({"stage": stage, "message": message})
    return events


def _run_summary(run_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    state = entry["state"]
    stages_done: list[str] = []
    for event in entry["events"]:
        if event["node"] not in stages_done:
            stages_done.append(event["node"])

    retry_count = state.get("retry_count", {})
    end_time = entry.get("finished_at") or time.time()

    timeline = _stage_timeline(entry)
    completed_at_by_node = {record["node"]: record["completed_at"] for record in timeline}
    events = _plain_language_events(state, stages_done)
    for event in events:
        event["timestamp"] = completed_at_by_node.get(event["stage"])

    return _json_safe(
        {
            "run_id": run_id,
            "filename": entry["filename"],
            "description": state.get("use_case_description"),
            "status": entry["status"],
            "created_at": entry["created_at"],
            "elapsed_seconds": round(end_time - entry["created_at"], 1),
            "llm_call_count": len(read_trace(run_id)),
            "stages_done": stages_done,
            "stage_timeline": timeline,
            "events": events,
            "retry_count": retry_count,
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
            "insights": generate_insights(state, stages_done),
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
            "finished_at": None,
            "cancel_requested": False,
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
        _record_event(entry, "confirm")

    threading.Thread(target=_run_main, args=(run_id,), daemon=True).start()
    return {"run_id": run_id, "status": "running"}


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict[str, Any]:
    """Best-effort cancellation (PRODUCT.md 3.3: 'always show a way to cancel
    a running pipeline'). Takes effect between graph steps — an in-flight
    node/training job already running is not interrupted mid-execution, but
    no further stages will start."""
    entry = _get_entry(run_id)
    with _lock:
        if entry["status"] not in ("profiling", "running", "awaiting_confirmation"):
            raise HTTPException(status_code=409, detail=f"run is '{entry['status']}', nothing to cancel")
        entry["cancel_requested"] = True
        if entry["status"] == "awaiting_confirmation":
            entry["status"] = "cancelled"
            entry["finished_at"] = time.time()
    return {"run_id": run_id, "status": "cancelling"}


def _require_model_path(entry: dict[str, Any]) -> str:
    model_path = (entry["state"].get("best_model") or {}).get("model_path")
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="no trained model artifact for this run yet")
    return model_path


@app.get("/api/runs/{run_id}/model")
def download_model(run_id: str):
    entry = _get_entry(run_id)
    model_path = _require_model_path(entry)
    return FileResponse(model_path, filename=f"automl_{run_id}.joblib", media_type="application/octet-stream")


@app.get("/api/runs/{run_id}/script")
def download_script(run_id: str) -> Response:
    """Standalone Python script reproducing this run's feature engineering +
    training (PRODUCT.md 3.4 'export pipeline as code'; PRD 2.3 step 6)."""
    entry = _get_entry(run_id)
    if not entry["state"].get("best_model"):
        raise HTTPException(status_code=404, detail="no winning model to export a script for yet")
    script = generate_training_script(entry["state"])
    return Response(
        content=script,
        media_type="text/x-python",
        headers={"Content-Disposition": f'attachment; filename="automl_{run_id}_train.py"'},
    )


@app.get("/api/runs/{run_id}/model/schema")
def get_model_schema(run_id: str) -> dict[str, Any]:
    """Feature columns + task info the frontend's 'test the model' tab needs
    to build an input form without hardcoding anything."""
    entry = _get_entry(run_id)
    model_path = _require_model_path(entry)
    schema = load_model_schema(model_path)
    task_spec = entry["state"].get("task_spec", {}) or {}
    best_model = entry["state"].get("best_model", {}) or {}
    return _json_safe(
        {
            "feature_columns": schema["feature_columns"],
            "task_type": task_spec.get("task_type"),
            "target_column": task_spec.get("target_column"),
            "candidate_name": best_model.get("candidate_name"),
        }
    )


class PredictRequest(BaseModel):
    values: dict[str, float]


@app.post("/api/runs/{run_id}/predict")
def predict(run_id: str, body: PredictRequest) -> dict[str, Any]:
    """Score one user-supplied row against the winning model — the
    'ready-made deployment to test the model' PRODUCT.md 3.4/3.5 gestures at,
    scoped to local interactive testing rather than a hosted endpoint."""
    entry = _get_entry(run_id)
    model_path = _require_model_path(entry)
    try:
        result = predict_one(model_path, body.values)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear 400, not a 500 stack trace
        raise HTTPException(status_code=400, detail=f"could not score this input: {exc}") from exc
    return _json_safe(result)


@app.get("/api/runs/{run_id}/trace")
def get_trace(run_id: str) -> list[dict[str, Any]]:
    _get_entry(run_id)
    return _json_safe(read_trace(run_id))


# Serve the frontend last so /api/* wins routing.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
