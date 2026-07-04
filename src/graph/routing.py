"""Deterministic routing functions (CLAUDE.md: keep these out of inline
lambdas in the graph definition). Every loop-back edge checks its retry cap
from config/runtime.yaml before looping again, and always has a graceful
fallback (CLAUDE.md rule #3)."""

from __future__ import annotations

import yaml

from src.state import PipelineState


def _max_retries() -> int:
    with open("config/runtime.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["retry"]["max_retries"]


def _poll_max_attempts() -> int:
    with open("config/runtime.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["training"]["poll_max_attempts"]


def route_after_feature_engineering(state: PipelineState) -> str:
    if state.get("feature_plan_valid"):
        return "apply_feature_plan"
    if state.get("retry_count", {}).get("feature_engineering", 0) < _max_retries():
        return "feature_engineering"
    state.setdefault("errors", []).append(
        "feature_engineering: retry cap reached, falling back to report with no feature plan applied"
    )
    state["status"] = "failed"
    return "report"


def route_after_apply_feature_plan(state: PipelineState) -> str:
    if state.get("feature_plan_valid"):
        return "model_selection"
    if state.get("retry_count", {}).get("feature_engineering", 0) < _max_retries():
        return "feature_engineering"
    state.setdefault("errors", []).append(
        "apply_feature_plan: retry cap reached, falling back to report with no feature plan applied"
    )
    state["status"] = "failed"
    return "report"


def route_after_feature_engineering_prep(state: PipelineState) -> str:
    """API prep_graph variant: on success, stop here (END) rather than
    proceeding to apply_feature_plan — the API pauses the whole graph at this
    point so the user can review/approve the plan before anything is applied."""
    if state.get("feature_plan_valid"):
        return "done"
    if state.get("retry_count", {}).get("feature_engineering", 0) < _max_retries():
        return "feature_engineering"
    state.setdefault("errors", []).append(
        "feature_engineering: retry cap reached, falling back to report with no feature plan applied"
    )
    state["status"] = "failed"
    return "failed"


def route_after_apply_feature_plan_approved(state: PipelineState) -> str:
    """API train_graph variant: the plan reaching this node was already
    human-approved, so a failure here is a hard stop rather than a silent
    LLM-regenerated, unreviewed replan."""
    if state.get("feature_plan_valid"):
        return "model_selection"
    state.setdefault("errors", []).append(
        "apply_feature_plan: the approved feature plan failed to apply — start a new run and adjust your approval"
    )
    state["status"] = "failed"
    return "report"


def route_after_poll_training(state: PipelineState) -> str:
    results = state.get("training_results", [])
    all_terminal = bool(results) and all(r["status"] in ("succeeded", "failed") for r in results)
    if all_terminal:
        return "evaluate"
    if state.get("retry_count", {}).get("poll_training", 0) < _poll_max_attempts():
        return "poll_training"
    state.setdefault("errors", []).append("poll_training: attempt cap reached with jobs still pending")
    return "evaluate"
