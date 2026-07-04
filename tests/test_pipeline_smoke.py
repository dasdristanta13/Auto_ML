"""End-to-end smoke test for the LangGraph pipeline with all LLM calls
mocked (deterministic canned responses) so it runs with no API keys and no
network access — proves the graph wiring, retry/routing, sandboxing, and
async training dispatch all work together locally."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.graph.build_graph import build_graph
from src.llm.client import LLMClient
from src.state import new_state


def _fake_generate(self, run_id, node, system_prompt, user_prompt, json_schema=None, retries=1):
    if node == "understand_usecase":
        return {
            "task_type": "classification",
            "target_column": "churned",
            "metric": "f1",
            "constraints": [],
            "is_ambiguous": False,
            "ambiguity_reason": None,
        }
    if node == "feature_engineering":
        return {
            "steps": [
                {
                    "op": "impute",
                    "columns": ["tenure_months", "monthly_spend", "support_tickets"],
                    "params": {"strategy": "mean"},
                    "rationale": "fill missing values",
                }
            ],
            "plan_rationale": "basic imputation only for this smoke test",
        }
    if node == "model_selection":
        return {
            "candidates": [
                {
                    "name": "baseline_rf",
                    "library": "sklearn",
                    "estimator": "RandomForestClassifier",
                    "hyperparams": {"n_estimators": 10, "max_depth": 3, "random_state": 0},
                    "rationale": "fast baseline for a small imbalanced dataset",
                }
            ]
        }
    if node == "report":
        return "This is a test report."
    raise ValueError(f"unexpected node in fake_generate: {node}")


def test_full_pipeline_completes_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMClient, "generate", _fake_generate)
    # the CLI's feature_approval_checkpoint_node otherwise blocks on stdin input()
    monkeypatch.setenv("AUTOML_AUTO_APPROVE_FEATURES", "1")

    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "tenure_months": rng.normal(12, 5, n),
            "monthly_spend": rng.normal(60, 15, n),
            "support_tickets": rng.poisson(1, n),
            "churned": rng.choice([0, 1], n, p=[0.85, 0.15]),
        }
    )
    dataset_path = tmp_path / "churn.csv"
    df.to_csv(dataset_path, index=False)

    state = new_state(run_id="smoke-test", dataset_path=str(dataset_path), use_case_description="predict which customers will churn")

    graph = build_graph()
    final_state = graph.invoke(state, config={"recursion_limit": 100})

    assert final_state["status"] == "completed"
    assert final_state["feature_plan_valid"] is True
    # model_selection_node fills in every applicable classification family
    # alongside the LLM's single proposed candidate (CLAUDE.md rule #2:
    # completeness is a deterministic floor, not left to the LLM's judgment).
    candidate_names = {c["name"] for c in final_state["candidate_models"]}
    assert candidate_names == {
        "baseline_rf",
        "Logistic Regression",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM",
    }
    assert len(final_state["training_results"]) == 5
    assert all(r["status"] == "succeeded" for r in final_state["training_results"])
    assert "f1" in final_state["best_model"]["metrics"]
    assert final_state["report"]["narrative"] == "This is a test report."
