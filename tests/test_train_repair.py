"""Tests for per-candidate training auto-repair: routing.route_after_poll_training's
repair branch, and train_candidate_repair_node actually diagnosing a real
training failure and successfully retrying it with corrected hyperparameters.

No mocked training here — a candidate with an invalid hyperparameter
(n_estimators=-5) is dispatched for real via train_model/poll_training_job so
the failure and the repair's retry are both genuine, not simulated."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yaml

from src.agents.train_candidate_repair_node import train_candidate_repair_node
from src.graph.routing import route_after_poll_training
from src.llm.client import LLMClient
from src.training.dispatch import poll_training_job, train_model


def _max_retries() -> int:
    with open("config/runtime.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["retry"]["max_retries"]


def _poll_until_terminal(run_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = poll_training_job.invoke({"run_id": run_id})
        if result["status"] in ("succeeded", "failed"):
            return result
        time.sleep(0.1)
    raise TimeoutError(f"run {run_id} never reached a terminal state")


def test_route_after_poll_training_repairs_failed_candidate_under_cap():
    state = {
        "training_results": [{"status": "failed", "candidate_name": "A"}],
        "candidate_repair_count": {},
    }
    assert route_after_poll_training(state) == "repair_training_candidates"


def test_route_after_poll_training_falls_to_evaluate_once_cap_reached():
    state = {
        "training_results": [{"status": "failed", "candidate_name": "A"}],
        "candidate_repair_count": {"A": _max_retries()},
    }
    assert route_after_poll_training(state) == "evaluate"


def test_train_candidate_repair_node_fixes_and_retries_a_real_failure(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "target": rng.choice([0, 1], n),
        }
    )
    dataset_path = tmp_path / "data.csv"
    df.to_csv(dataset_path, index=False)

    # n_estimators=-5 is invalid and raises inside RandomForestClassifier.fit(),
    # which _run_job catches and records as a genuine "failed" registry entry.
    failed_run_id = train_model.invoke(
        {
            "candidate_name": "bad_rf",
            "library": "sklearn",
            "estimator": "RandomForestClassifier",
            "hyperparams": {"n_estimators": -5},
            "dataset_path": str(dataset_path),
            "target_column": "target",
            "task_type": "classification",
            "cv_enabled": False,
        }
    )
    failed_result = _poll_until_terminal(failed_run_id)
    assert failed_result["status"] == "failed"

    def _fake_generate(self, run_id, node, system_prompt, user_prompt, json_schema=None, retries=1):
        assert node == "train_candidate_repair"
        return {
            "name": "bad_rf",
            "library": "sklearn",
            "estimator": "RandomForestClassifier",
            "hyperparams": {"n_estimators": 10},
            "rationale": "n_estimators must be a positive integer",
        }

    monkeypatch.setattr(LLMClient, "generate", _fake_generate)

    state = {
        "run_id": "repair-test",
        "task_spec": {"target_column": "target", "task_type": "classification"},
        "cv_enabled": False,
        "cv_folds": 5,
        "resampling_plan": {"enabled": False, "method": "none"},
        "transformed_dataset_path": str(dataset_path),
        "candidate_models": [
            {
                "name": "bad_rf",
                "library": "sklearn",
                "estimator": "RandomForestClassifier",
                "hyperparams": {"n_estimators": -5},
                "rationale": "",
            }
        ],
        "training_run_ids": [failed_run_id],
        "training_results": [failed_result],
        "candidate_repair_count": {},
        "training_repair_log": [],
    }

    result_state = train_candidate_repair_node(state)

    assert result_state["candidate_repair_count"]["bad_rf"] == 1
    assert len(result_state["training_repair_log"]) == 1
    assert result_state["training_repair_log"][0]["candidate_name"] == "bad_rf"
    new_run_id = result_state["training_run_ids"][0]
    assert new_run_id != failed_run_id
    assert result_state["candidate_models"][0]["hyperparams"] == {"n_estimators": 10}
    # identity fields must be forced back even though the fake LLM already
    # returned them correctly here — this asserts the retry actually used
    # the corrected hyperparams and ran to completion.
    retried_result = _poll_until_terminal(new_run_id)
    assert retried_result["status"] == "succeeded"
