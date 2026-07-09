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

import os

import pandas as pd
import yaml
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.auth.session import create_session, destroy_session, get_session
from src.data_io import load_dataset
from src.export.script_export import generate_training_script
from src.graph.build_graph import build_intake_graph, build_prep_graph, build_train_graph
from src.agents.chat_node import answer_chat_question
from src.insights.auto_insights import generate_insights, suggested_questions
from src.profiling import preview
from src.profiling.heuristics import target_too_high_cardinality_for_classification
from src.llm.tracing import read_trace
from src.state import PipelineState, new_state
from src.training.dispatch import explain_prediction, load_model_schema, predict_one

UPLOAD_DIR = Path("data/uploads")

app = FastAPI(title="Agentic AutoML")

_runs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()

_dataset_df_cache: dict[str, pd.DataFrame] = {}
_dataset_df_cache_lock = threading.Lock()


def _get_cached_df(run_id: str, dataset_path: str) -> pd.DataFrame:
    """Runs are immutable once created (their CSV never changes), so this
    cache never needs invalidation — only lazy population."""
    with _dataset_df_cache_lock:
        if run_id not in _dataset_df_cache:
            _dataset_df_cache[run_id] = load_dataset(dataset_path)
        return _dataset_df_cache[run_id]


def _dataset_df_for_run(run_id: str, entry: dict[str, Any]) -> pd.DataFrame:
    dataset_path = entry["state"].get("dataset_path")
    if not dataset_path or not Path(dataset_path).exists():
        raise HTTPException(status_code=404, detail="dataset file is no longer available for this run")
    return _get_cached_df(run_id, dataset_path)


_intake_graph = build_intake_graph()
_prep_graph = build_prep_graph()
_train_graph = build_train_graph()

SESSION_COOKIE_NAME = "automl_session"


def _auth_config() -> dict[str, Any]:
    with open("config/runtime.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["auth"]


def _get_session_from_request(request: Request) -> dict[str, Any] | None:
    return get_session(request.cookies.get(SESSION_COOKIE_NAME))


def require_session(request: Request) -> dict[str, Any]:
    """FastAPI dependency: 401s any request without a valid, unexpired
    session cookie. Applied to every /api/runs* endpoint (Task 3) — this is
    a demo-credential gate for a single-user local tool, not production
    multi-tenant auth (see docs/superpowers/specs/2026-07-05-login-page-design.md)."""
    session = _get_session_from_request(request)
    if session is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return session


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/auth/login")
def login(body: LoginRequest, response: Response) -> dict[str, Any]:
    demo_email = os.environ.get("AUTOML_DEMO_EMAIL", "demo@automl.local")
    demo_password = os.environ.get("AUTOML_DEMO_PASSWORD", "demo123")
    if body.email.strip().lower() != demo_email.strip().lower() or body.password != demo_password:
        raise HTTPException(status_code=401, detail="invalid email or password")

    ttl_hours = _auth_config()["session_ttl_hours"]
    token = create_session(demo_email, ttl_hours)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=int(ttl_hours * 3600),
        httponly=True,
        samesite="lax",
        secure=False,  # local http dev (run_server.py); see design spec
        path="/",
    )
    return {"ok": True}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response) -> dict[str, Any]:
    destroy_session(request.cookies.get(SESSION_COOKIE_NAME))
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/auth/session")
def auth_session(request: Request) -> dict[str, Any]:
    session = _get_session_from_request(request)
    return {"authenticated": session is not None, "email": session["email"] if session else None}


@app.get("/api/auth/demo-credentials")
def demo_credentials() -> dict[str, Any]:
    """Unauthenticated by design — this is a demo credential for a
    single-user local tool, not a secret (see design spec)."""
    return {
        "email": os.environ.get("AUTOML_DEMO_EMAIL", "demo@automl.local"),
        "password": os.environ.get("AUTOML_DEMO_PASSWORD", "demo123"),
    }


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


def _run_prep(run_id: str) -> None:
    _stream_graph(
        run_id,
        _prep_graph,
        lambda state: "failed" if state.get("status") == "failed" else "awaiting_feature_approval",
    )


def _run_train(run_id: str) -> None:
    _stream_graph(
        run_id,
        _train_graph,
        lambda state: "completed" if state.get("status") == "completed" else "failed",
    )


class ConfirmRequest(BaseModel):
    target_column: str
    task_type: str
    metric: str
    # optional: order-by column for a chronological (leakage-safe) train/test
    # split on time-ordered data; None/empty means a random split.
    time_column: Optional[str] = None
    constraints: list[str] = []
    cv_enabled: bool = True
    cv_folds: int = 5
    tuning_enabled: bool = True
    feature_selection_enabled: bool = False


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
            # aggregate counts only (already PII-redacted upstream) — feeds
            # the class-distribution panel for the confirmed target column
            "top_values": info.get("top_values"),
            # distributional stats (mean/std/percentiles/min/max), no raw rows —
            # lets the class-distribution panel fall back to a 0/1-encoded
            # numeric target's mean-as-positive-rate when top_values wasn't
            # computed for it (mirrors minority_ratio() in profiling/heuristics.py)
            "numeric_summary": info.get("numeric_summary"),
        }
        for name, info in columns.items()
    ]


def _best_score(entry: dict[str, Any]) -> float | None:
    """Best model's score on the run's own success metric, for the run list."""
    state = entry["state"]
    metric = (state.get("task_spec") or {}).get("metric")
    metrics = (state.get("best_model") or {}).get("metrics") or {}
    if metric and metric in metrics:
        return round(float(metrics[metric]), 4)
    return None


_STAGE_MESSAGES = {
    "profile": "Profiled the dataset — schema, null rates, cardinality, and PII scan complete.",
    "understand_usecase": "Interpreted the use case into a task specification.",
    "confirm": "Task specification confirmed — compute-heavy work unlocked.",
    "leakage_check": "Checked for columns that may leak information about the target.",
    "eda": "Ran automated exploratory data analysis to ground the feature plan in this specific dataset.",
    "feature_engineering": "Planned feature transformations (imputation, encoding, scaling), informed by the EDA.",
    "feature_approval": "Feature engineering plan approved — proceeding to training.",
    "apply_feature_plan": "Applied the approved feature plan to the dataset.",
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


def _chat_context(state: PipelineState) -> dict[str, Any]:
    """Trimmed subset of _run_summary's already-redacted, already-computed
    fields for the chat prompt — skips event timelines/stage messages to
    keep the prompt compact (see the chat design spec)."""
    feature_plan = state.get("feature_plan") or {}
    return _json_safe(
        {
            "task_spec": state.get("task_spec"),
            "profile_summary": {
                "row_count": state.get("profile", {}).get("row_count"),
                "column_count": state.get("profile", {}).get("column_count"),
            },
            "eda_insights": (state.get("eda_report") or {}).get("insights", []),
            "leakage_flags": state.get("leakage_flags", []),
            "feature_plan_steps": [
                {"op": s.get("op"), "columns": s.get("columns"), "rationale": s.get("rationale")}
                for s in feature_plan.get("steps", [])
            ],
            "training_results": [
                {
                    "candidate_name": r.get("candidate_name"),
                    "status": r.get("status"),
                    "metrics": r.get("metrics"),
                    "tuning": r.get("tuning"),
                }
                for r in state.get("training_results", [])
            ],
            "best_model": state.get("best_model"),
            "report_narrative": (state.get("report") or {}).get("narrative"),
        }
    )


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

    insights = generate_insights(state, stages_done)

    return _json_safe(
        {
            "run_id": run_id,
            "filename": entry["filename"],
            "description": state.get("use_case_description"),
            "source_run_id": entry.get("source_run_id"),
            "status": entry["status"],
            "created_at": entry["created_at"],
            "elapsed_seconds": round(end_time - entry["created_at"], 1),
            "llm_call_count": len(read_trace(run_id)),
            "stages_done": stages_done,
            "stage_timeline": timeline,
            "events": events,
            "retry_count": retry_count,
            "task_spec": state.get("task_spec"),
            "cv_config": {
                "enabled": state.get("cv_enabled", True),
                "requested_folds": state.get("cv_folds", 5),
            },
            "tuning_config": {"enabled": state.get("tuning_enabled", True)},
            "feature_selection_config": {"enabled": state.get("feature_selection_enabled", False)},
            "feature_selection": state.get("feature_selection_result"),
            "eda_report": state.get("eda_report"),
            "resampling_suggestion": state.get("resampling_suggestion"),
            "resampling_plan": state.get("resampling_plan"),
            "profile_summary": {
                "row_count": state.get("profile", {}).get("row_count"),
                "column_count": state.get("profile", {}).get("column_count"),
                "pii_columns_detected": state.get("profile", {})
                .get("pii_report", {})
                .get("pii_columns_detected"),
                "quality": state.get("profile", {}).get("quality"),
                "memory_bytes": state.get("profile", {}).get("memory_bytes"),
            },
            "profile_columns": _profile_columns(state),
            "leakage_flags": state.get("leakage_flags", []),
            "feature_plan": state.get("feature_plan"),
            "training_results": state.get("training_results", []),
            "best_model": state.get("best_model"),
            "insights": insights,
            "report": state.get("report", {}).get("narrative"),
            "errors": state.get("errors", []),
            "chat_history": entry.get("chat_history", []),
            "suggested_questions": suggested_questions(
                insights, state.get("task_spec") or {}, state.get("best_model") or {}
            ),
        }
    )


@app.post("/api/runs")
async def create_run(
    file: UploadFile = File(...), description: str = Form(...), _session: dict[str, Any] = Depends(require_session)
) -> dict[str, Any]:
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
            "chat_history": [],
        }

    threading.Thread(target=_run_intake, args=(run_id,), daemon=True).start()
    return {"run_id": run_id, "status": "profiling"}


class ExperimentRequest(BaseModel):
    description: str


@app.post("/api/runs/{run_id}/experiments")
def create_experiment(
    run_id: str, body: ExperimentRequest, _session: dict[str, Any] = Depends(require_session)
) -> dict[str, Any]:
    """Start a new experiment on an existing run's dataset — no re-upload.
    Reuses the source run's already-computed profile when present (intake's
    profile_node skips re-profiling), re-infers the task spec from the new
    description, and pauses at the standard confirm checkpoint like any
    fresh run. See docs/superpowers/specs/2026-07-06-multi-experiment-design.md."""
    source = _get_entry(run_id)
    description = body.description.strip()
    if not description:
        raise HTTPException(status_code=400, detail="description must not be empty")
    dataset_path = source["state"].get("dataset_path")
    if not dataset_path or not Path(dataset_path).exists():
        raise HTTPException(
            status_code=409, detail="the source run's dataset file is no longer available — please re-upload it"
        )

    new_id = str(uuid.uuid4())[:8]
    state = new_state(run_id=new_id, dataset_path=dataset_path, use_case_description=description)
    source_profile = source["state"].get("profile")
    if source_profile:
        state["profile"] = source_profile
    with _lock:
        _runs[new_id] = {
            "state": state,
            "status": "profiling",
            "events": [],
            "filename": source["filename"],
            "created_at": time.time(),
            "finished_at": None,
            "cancel_requested": False,
            "chat_history": [],
            "source_run_id": run_id,
        }

    threading.Thread(target=_run_intake, args=(new_id,), daemon=True).start()
    return {"run_id": new_id, "status": "profiling"}


@app.get("/api/runs")
def list_runs(_session: dict[str, Any] = Depends(require_session)) -> list[dict[str, Any]]:
    with _lock:
        return [
            {
                "run_id": run_id,
                "filename": entry["filename"],
                "status": entry["status"],
                "created_at": entry["created_at"],
                "description": entry["state"].get("use_case_description"),
                "best_score": _best_score(entry),
                "metric": (entry["state"].get("task_spec") or {}).get("metric"),
                "source_run_id": entry.get("source_run_id"),
            }
            for run_id, entry in sorted(_runs.items(), key=lambda kv: -kv[1]["created_at"])
        ]


@app.get("/api/datasets")
def list_datasets(_session: dict[str, Any] = Depends(require_session)) -> list[dict[str, Any]]:
    """A 'dataset' is a top-level run (no source_run_id) — a re-run
    experiment reuses its source's dataset file and is not listed separately
    (see docs/superpowers/specs/2026-07-06-dataset-preview-design.md)."""
    with _lock:
        return [
            {
                "run_id": run_id,
                "filename": entry["filename"],
                "status": entry["status"],
                "created_at": entry["created_at"],
                "row_count": entry["state"].get("profile", {}).get("row_count"),
                "column_count": entry["state"].get("profile", {}).get("column_count"),
                "quality_score": (entry["state"].get("profile", {}).get("quality") or {}).get("overall"),
            }
            for run_id, entry in sorted(_runs.items(), key=lambda kv: -kv[1]["created_at"])
            if not entry.get("source_run_id")
        ]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, _session: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
    entry = _get_entry(run_id)
    with _lock:
        return _run_summary(run_id, entry)


@app.post("/api/runs/{run_id}/confirm")
def confirm_run(
    run_id: str, body: ConfirmRequest, _session: dict[str, Any] = Depends(require_session)
) -> dict[str, Any]:
    entry = _get_entry(run_id)
    with _lock:
        if entry["status"] != "awaiting_confirmation":
            raise HTTPException(status_code=409, detail=f"run is '{entry['status']}', not awaiting confirmation")

        state = entry["state"]
        columns = state.get("profile", {}).get("columns", {})
        if columns and body.target_column not in columns:
            raise HTTPException(status_code=400, detail=f"'{body.target_column}' is not a column of this dataset")
        time_column = body.time_column or None
        if time_column and columns and time_column not in columns:
            raise HTTPException(status_code=400, detail=f"'{time_column}' is not a column of this dataset")
        if time_column and time_column == body.target_column:
            raise HTTPException(status_code=400, detail="time_column cannot be the same as the target column")
        if body.task_type == "classification":
            target_info = columns.get(body.target_column, {})
            row_count = state.get("profile", {}).get("row_count")
            n_unique = target_info.get("n_unique")
            if row_count and n_unique and target_too_high_cardinality_for_classification(n_unique, row_count):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"'{body.target_column}' has {n_unique} unique values across {row_count} rows — "
                        "too high-cardinality to be a meaningful classification target (it looks like an "
                        "identifier or free-text column rather than a category). Pick a different target "
                        "column, or choose 'regression' if this is a continuous value."
                    ),
                )
        if body.cv_enabled and body.cv_folds < 2:
            raise HTTPException(status_code=400, detail="cv_folds must be at least 2 when cross-validation is enabled")

        task_spec = dict(state.get("task_spec") or {})
        task_spec.update(
            target_column=body.target_column,
            task_type=body.task_type,
            metric=body.metric,
            time_column=time_column,
            constraints=body.constraints,
            is_ambiguous=False,
            ambiguity_reason=None,
        )
        state["task_spec"] = task_spec
        state["cv_enabled"] = body.cv_enabled
        state["cv_folds"] = body.cv_folds
        state["tuning_enabled"] = body.tuning_enabled
        state["feature_selection_enabled"] = body.feature_selection_enabled
        state["needs_human_confirmation"] = False
        state["human_confirmed"] = True
        entry["status"] = "running"
        _record_event(entry, "confirm")

    threading.Thread(target=_run_prep, args=(run_id,), daemon=True).start()
    return {"run_id": run_id, "status": "running"}


class FeatureApprovalRequest(BaseModel):
    approved_step_indices: list[int]
    resampling_enabled: bool = False
    resampling_method: str = "none"


_VALID_RESAMPLING_METHODS = {"none", "smote", "random_oversample", "random_undersample"}


@app.post("/api/runs/{run_id}/approve-features")
def approve_features(
    run_id: str, body: FeatureApprovalRequest, _session: dict[str, Any] = Depends(require_session)
) -> dict[str, Any]:
    entry = _get_entry(run_id)
    with _lock:
        if entry["status"] != "awaiting_feature_approval":
            raise HTTPException(status_code=409, detail=f"run is '{entry['status']}', not awaiting feature approval")
        if body.resampling_method not in _VALID_RESAMPLING_METHODS:
            raise HTTPException(status_code=400, detail=f"unknown resampling_method '{body.resampling_method}'")

        state = entry["state"]
        plan = dict(state.get("feature_plan") or {})
        all_steps = plan.get("steps", [])
        approved = {i for i in body.approved_step_indices if 0 <= i < len(all_steps)}
        plan["steps"] = [step for i, step in enumerate(all_steps) if i in approved]
        state["feature_plan"] = plan
        state["feature_plan_approved"] = True
        state["needs_feature_approval"] = False
        state["resampling_plan"] = {
            "enabled": body.resampling_enabled and body.resampling_method != "none",
            "method": body.resampling_method if body.resampling_enabled else "none",
        }
        entry["status"] = "running"
        _record_event(entry, "feature_approval")

    threading.Thread(target=_run_train, args=(run_id,), daemon=True).start()
    return {"run_id": run_id, "status": "running"}


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str, _session: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
    """Best-effort cancellation (PRODUCT.md 3.3: 'always show a way to cancel
    a running pipeline'). Takes effect between graph steps — an in-flight
    node/training job already running is not interrupted mid-execution, but
    no further stages will start."""
    entry = _get_entry(run_id)
    with _lock:
        if entry["status"] not in ("profiling", "running", "awaiting_confirmation", "awaiting_feature_approval"):
            raise HTTPException(status_code=409, detail=f"run is '{entry['status']}', nothing to cancel")
        entry["cancel_requested"] = True
        if entry["status"] in ("awaiting_confirmation", "awaiting_feature_approval"):
            entry["status"] = "cancelled"
            entry["finished_at"] = time.time()
    return {"run_id": run_id, "status": "cancelling"}


def _require_model_path(entry: dict[str, Any]) -> str:
    model_path = (entry["state"].get("best_model") or {}).get("model_path")
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="no trained model artifact for this run yet")
    return model_path


@app.get("/api/runs/{run_id}/model")
def download_model(run_id: str, _session: dict[str, Any] = Depends(require_session)):
    entry = _get_entry(run_id)
    model_path = _require_model_path(entry)
    return FileResponse(model_path, filename=f"automl_{run_id}.joblib", media_type="application/octet-stream")


@app.get("/api/runs/{run_id}/script")
def download_script(run_id: str, _session: dict[str, Any] = Depends(require_session)) -> Response:
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
def get_model_schema(run_id: str, _session: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
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
            "feature_types": schema.get("feature_types", {}),
            "task_type": task_spec.get("task_type"),
            "target_column": task_spec.get("target_column"),
            "candidate_name": best_model.get("candidate_name"),
        }
    )


@app.get("/api/runs/{run_id}/explainability")
def get_explainability(run_id: str, _session: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
    """Precomputed SHAP feature impact + narrative for the winning model
    (explainability_node) — computed once at training time, not on request."""
    entry = _get_entry(run_id)
    _require_model_path(entry)
    best_model = entry["state"].get("best_model") or {}
    explainability = best_model.get("explainability") or {
        "method": "unavailable",
        "feature_impact": [],
        "narrative": None,
        "note": "explainability has not been computed for this run yet",
        "summary_plot": None,
        "bar_plot": None,
        "dependence_plots": [],
    }
    return _json_safe(explainability)


class PredictRequest(BaseModel):
    # raw feature values: numbers for numeric columns, strings for
    # categorical (e.g. target-encoded) columns — the model pipeline applies
    # its own preprocessing.
    values: dict[str, Any]


@app.post("/api/runs/{run_id}/predict")
def predict(
    run_id: str, body: PredictRequest, _session: dict[str, Any] = Depends(require_session)
) -> dict[str, Any]:
    """Score one user-supplied row against the winning model — the
    'ready-made deployment to test the model' PRODUCT.md 3.4/3.5 gestures at,
    scoped to local interactive testing rather than a hosted endpoint."""
    entry = _get_entry(run_id)
    model_path = _require_model_path(entry)
    try:
        result = predict_one(model_path, body.values)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear 400, not a 500 stack trace
        raise HTTPException(status_code=400, detail=f"could not score this input: {exc}") from exc
    transformed_dataset_path = entry["state"].get("transformed_dataset_path", "")
    explanation = explain_prediction(model_path, body.values, transformed_dataset_path)
    result["contributions"] = explanation["contributions"] if explanation else None
    result["waterfall_plot_base64"] = explanation["waterfall_plot_base64"] if explanation else None
    return _json_safe(result)


@app.get("/api/runs/{run_id}/trace")
def get_trace(run_id: str, _session: dict[str, Any] = Depends(require_session)) -> list[dict[str, Any]]:
    _get_entry(run_id)
    return _json_safe(read_trace(run_id))


@app.get("/api/runs/{run_id}/preview")
def get_dataset_preview(
    run_id: str,
    page: int = 1,
    page_size: int = 50,
    sort_by: Optional[str] = None,
    sort_dir: str = "asc",
    search: Optional[str] = None,
    _session: dict[str, Any] = Depends(require_session),
) -> dict[str, Any]:
    entry = _get_entry(run_id)
    df = _dataset_df_for_run(run_id, entry)
    try:
        result = preview.paginate_rows(
            df, page=page, page_size=page_size, sort_by=sort_by, sort_dir=sort_dir, search=search
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    pii_columns = (entry["state"].get("profile", {}).get("pii_report", {}) or {}).get("columns", {}) or {}
    result["pii_columns"] = sorted(pii_columns)
    return _json_safe(result)


@app.get("/api/runs/{run_id}/columns/{column}")
def get_column_detail(
    run_id: str, column: str, _session: dict[str, Any] = Depends(require_session)
) -> dict[str, Any]:
    entry = _get_entry(run_id)
    df = _dataset_df_for_run(run_id, entry)
    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"unknown column '{column}'")
    task_spec = entry["state"].get("task_spec") or {}
    result = preview.column_detail(df, column, target_column=task_spec.get("target_column"))

    feature_plan = entry["state"].get("feature_plan") or {}
    matching_steps = [s for s in feature_plan.get("steps", []) if column in (s.get("columns") or [])]
    leakage_flags = [f for f in entry["state"].get("leakage_flags", []) if f.get("column") == column]
    if matching_steps or leakage_flags:
        result["ml_insights"] = {
            "analyzed": True,
            "recommended_steps": [{"op": s.get("op"), "rationale": s.get("rationale")} for s in matching_steps],
            "leakage_flags": leakage_flags,
        }
    else:
        result["ml_insights"] = {"analyzed": False}
    return _json_safe(result)


@app.get("/api/runs/{run_id}/correlations")
def get_correlations(
    run_id: str, method: str = "pearson", _session: dict[str, Any] = Depends(require_session)
) -> dict[str, Any]:
    entry = _get_entry(run_id)
    df = _dataset_df_for_run(run_id, entry)
    try:
        result = preview.correlation_matrix(df, method=method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _json_safe(result)


@app.get("/api/runs/{run_id}/missing-values")
def get_missing_values(run_id: str, _session: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
    entry = _get_entry(run_id)
    df = _dataset_df_for_run(run_id, entry)
    return _json_safe(preview.missing_value_matrix(df))


@app.get("/api/runs/{run_id}/outliers")
def get_outliers(
    run_id: str, method: str = "iqr", _session: dict[str, Any] = Depends(require_session)
) -> dict[str, Any]:
    entry = _get_entry(run_id)
    df = _dataset_df_for_run(run_id, entry)
    try:
        result = preview.detect_outliers(df, method=method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _json_safe(result)


@app.get("/api/runs/{run_id}/dataset-summary")
def get_dataset_summary(run_id: str, _session: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
    entry = _get_entry(run_id)
    df = _dataset_df_for_run(run_id, entry)
    profile = entry["state"].get("profile", {}) or {}
    leakage_flags = entry["state"].get("leakage_flags", [])
    return _json_safe(
        {
            "feature_type_counts": preview.feature_type_counts(df),
            "ml_readiness_score": preview.ml_readiness_score(profile, leakage_flags),
        }
    )


class ChatRequest(BaseModel):
    question: str


@app.post("/api/runs/{run_id}/chat")
def chat(run_id: str, body: ChatRequest, _session: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
    """Answer a question about this run's already-computed results (see
    docs/superpowers/specs/2026-07-05-ai-assistant-chat-design.md). Only
    available once the report is ready — gated here, not just in the UI, so
    the contract holds regardless of client."""
    entry = _get_entry(run_id)
    with _lock:
        if entry["status"] not in ("completed", "failed"):
            raise HTTPException(status_code=409, detail=f"run is '{entry['status']}', not ready for questions yet")
        state = entry["state"]
        context = _chat_context(state)
        history_for_prompt = [{"role": h["role"], "content": h["content"]} for h in entry["chat_history"]]

    answer = answer_chat_question(run_id=run_id, context=context, history=history_for_prompt, question=body.question)

    with _lock:
        now = time.time()
        entry["chat_history"].append({"role": "user", "content": body.question, "timestamp": now})
        entry["chat_history"].append({"role": "assistant", "content": answer, "timestamp": now})

    return {"answer": answer}


class NoCacheStaticFiles(StaticFiles):
    """A local dev tool whose frontend gets rewritten constantly — browsers
    silently serving a stale cached app.js/index.html across those changes
    looks exactly like a broken feature (e.g. a UI list that stops updating)
    when the real cause is a never-refetched asset. Disable caching entirely
    rather than debug that class of bug repeatedly."""

    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


@app.get("/")
def serve_index(request: Request):
    """The one static path that needs server-side auth enforcement — every
    other static asset (styles.css, app.js, login.html/login.js) stays
    reachable unauthenticated via the mount below, but the app shell itself
    should redirect to the login page rather than flash stale/empty UI."""
    if _get_session_from_request(request) is None:
        return RedirectResponse("/login.html")
    return FileResponse("frontend/index.html")


# Serve the frontend last so /api/* wins routing.
app.mount("/", NoCacheStaticFiles(directory="frontend", html=True), name="frontend")
