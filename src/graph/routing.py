"""Deterministic routing functions (CLAUDE.md: keep these out of inline
lambdas in the graph definition). Every loop-back edge checks its retry cap
from config/runtime.yaml before looping again, and always has a graceful
fallback (CLAUDE.md rule #3)."""

from __future__ import annotations

import math

import yaml

from src.state import PipelineState


def _max_retries() -> int:
    with open("config/runtime.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["retry"]["max_retries"]


def _training_config() -> dict:
    with open("config/runtime.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["training"]


# Extra per-candidate wall-clock beyond the Optuna tuning budget itself
# (cross-validation folds + the final fit) that poll_max_attempts' adaptive
# sizing needs to account for.
_CV_FIT_OVERHEAD_SECONDS = 60


def poll_max_attempts(state: PipelineState) -> int:
    """How many poll_training loop-backs to allow before giving up on the
    jobs dispatched for THIS run, rather than a flat constant.

    config/runtime.yaml's poll_max_attempts is a floor (and, for small runs,
    the actual value) — but a flat cap doesn't scale with how much work was
    actually dispatched. A run with more candidates than max_concurrent_jobs
    trains in sequential batches, and each candidate's own budget is the
    Optuna tuning budget plus CV/fit overhead; sizing the cap off that keeps
    poll_training_node from abandoning jobs that are still legitimately
    running in the ThreadPoolExecutor (src/training/dispatch.py) — which
    otherwise finalizes the run as "completed" while the backend keeps
    training, unobserved, in the background.
    """
    cfg = _training_config()
    concurrency = max(int(cfg["max_concurrent_jobs"]), 1)
    n_candidates = len(state.get("training_run_ids", []))
    batches = math.ceil(n_candidates / concurrency) if n_candidates else 1
    per_candidate_seconds = cfg["hyperparam_search_budget_seconds"] + _CV_FIT_OVERHEAD_SECONDS
    needed_attempts = math.ceil(batches * per_candidate_seconds / cfg["poll_interval_seconds"])
    return max(int(cfg["poll_max_attempts"]), needed_attempts)


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
    if state.get("retry_count", {}).get("poll_training", 0) < poll_max_attempts(state):
        return "poll_training"

    # Cap genuinely exceeded: the ThreadPoolExecutor jobs behind any
    # still-pending/running candidate keep executing after this point (they
    # can't be cancelled once started), but nothing polls them again — mark
    # them explicitly instead of letting evaluate/report silently treat them
    # as absent, so the run's own results say what actually happened.
    timed_out = [r["candidate_name"] for r in results if r["status"] not in ("succeeded", "failed")]
    for r in results:
        if r["status"] not in ("succeeded", "failed"):
            r["status"] = "timed_out"
            r["error"] = (
                "poll_training: attempt cap reached while this candidate was still training; its "
                "background job may still be running, but this run stopped tracking it."
            )
    if timed_out:
        state.setdefault("errors", []).append(
            f"poll_training: attempt cap reached, {len(timed_out)} candidate(s) marked timed out: "
            f"{', '.join(timed_out)}"
        )
    return "evaluate"
