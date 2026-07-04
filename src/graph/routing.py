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


def route_after_poll_training(state: PipelineState) -> str:
    results = state.get("training_results", [])
    all_terminal = bool(results) and all(r["status"] in ("succeeded", "failed") for r in results)
    if all_terminal:
        return "evaluate"
    if state.get("retry_count", {}).get("poll_training", 0) < _poll_max_attempts():
        return "poll_training"
    state.setdefault("errors", []).append("poll_training: attempt cap reached with jobs still pending")
    return "evaluate"
