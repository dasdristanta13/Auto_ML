"""Adaptive poll_training cap.

CLAUDE.md rule #3 requires every loop-back edge to have a retry cap, but a
flat cap (config/runtime.yaml's poll_max_attempts) doesn't scale with the
work actually dispatched. When a run has more candidates (or a generous
tuning budget) than the flat cap anticipated, poll_training_node/
route_after_poll_training gave up while the ThreadPoolExecutor jobs for
still-training candidates kept running unobserved: evaluate/report finalized
the run as "completed" while the backend kept training in the background,
with no trace of it in training_results.

Fix: size the cap off the number of dispatched candidates and the training
config (src.graph.routing.poll_max_attempts), and when the cap is genuinely
exceeded, explicitly mark any still-pending/running candidate as
"timed_out" (rather than silently dropping it) so evaluate/report/the
frontend see the truth instead of an ambiguous gap.
"""

from __future__ import annotations

import pytest

from src.graph import nodes, routing

_CFG = {
    "max_concurrent_jobs": 2,
    "poll_interval_seconds": 2,
    "hyperparam_search_budget_seconds": 100,
    "poll_max_attempts": 10,
}


@pytest.fixture(autouse=True)
def _training_config(monkeypatch):
    monkeypatch.setattr(routing, "_training_config", lambda: dict(_CFG))


def _state(n_candidates: int, **overrides) -> dict:
    state = {"training_run_ids": [f"r{i}" for i in range(n_candidates)], "training_results": []}
    state.update(overrides)
    return state


def test_poll_max_attempts_scales_with_candidate_count():
    one_batch = routing.poll_max_attempts(_state(2))  # <= max_concurrent_jobs: 1 batch
    two_batches = routing.poll_max_attempts(_state(4))  # 2 batches
    assert two_batches > one_batch


def test_poll_max_attempts_never_below_configured_floor():
    assert routing.poll_max_attempts(_state(0)) >= _CFG["poll_max_attempts"]


def test_route_after_poll_training_keeps_polling_past_old_flat_default_for_many_candidates():
    # 4 candidates / 2 concurrent -> 2 batches -> needed attempts well above
    # the flat default of 10, which would previously have given up here.
    state = _state(4, retry_count={"poll_training": 40})
    state["training_results"] = [
        {"run_id": rid, "status": "running", "candidate_name": rid} for rid in state["training_run_ids"]
    ]
    assert routing.route_after_poll_training(state) == "poll_training"


def test_route_after_poll_training_marks_timed_out_candidates_past_adaptive_cap():
    state = _state(2)  # 1 batch
    cap = routing.poll_max_attempts(state)
    state["retry_count"] = {"poll_training": cap}
    state["training_results"] = [
        {"run_id": "r0", "status": "succeeded", "candidate_name": "r0", "metrics": {"f1": 0.9}, "error": None},
        {"run_id": "r1", "status": "running", "candidate_name": "r1", "error": None},
    ]

    assert routing.route_after_poll_training(state) == "evaluate"

    r0, r1 = state["training_results"]
    assert r0["status"] == "succeeded"  # untouched
    assert r1["status"] == "timed_out"
    assert r1["error"]
    assert any("timed out" in e.lower() for e in state["errors"])


def test_route_after_poll_training_returns_evaluate_when_all_terminal():
    state = _state(1)
    state["training_results"] = [{"run_id": "r0", "status": "succeeded", "candidate_name": "r0"}]
    assert routing.route_after_poll_training(state) == "evaluate"


class _FakePollTool:
    def __init__(self, results_by_id: dict[str, dict]) -> None:
        self._results = results_by_id

    def invoke(self, args: dict) -> dict:
        return dict(self._results[args["run_id"]])


def test_poll_training_node_increments_retry_count_past_old_flat_default(monkeypatch):
    monkeypatch.setattr(nodes.time, "sleep", lambda seconds: None)
    fake_results = {rid: {"run_id": rid, "status": "running"} for rid in ("r0", "r1", "r2", "r3")}
    monkeypatch.setattr(nodes, "poll_training_job", _FakePollTool(fake_results))

    # old flat cap (10) would have frozen retry_count at attempt 15 already;
    # the adaptive cap for 4 candidates / 2 concurrent (2 batches) allows it
    # to keep incrementing.
    state = {"training_run_ids": list(fake_results), "retry_count": {"poll_training": 15}}
    result_state = nodes.poll_training_node(state)
    assert result_state["retry_count"]["poll_training"] == 16
