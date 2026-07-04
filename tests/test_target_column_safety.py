"""Regression tests for a real recurring incident: a near-unique identifier
column (e.g. "customer_id") getting confirmed as a classification target.

n_unique ~= row_count means classification effectively needs one class per
row. Some estimators fail fast (liblinear refuses multiclass outright) but
others (e.g. GradientBoostingClassifier, which fits one tree per class per
boosting round) don't fail — they just become catastrophically expensive and
look hung. Two layers guard against this: understand_usecase_node forces the
human checkpoint open when the model confidently picks an identifier-like
target, and apply_feature_plan_node refuses to dispatch training at all if
that target still slips through (e.g. a human clicks past the warning)."""

from __future__ import annotations

import pandas as pd

from src.agents.understand_usecase_node import understand_usecase_node
from src.graph.nodes import apply_feature_plan_node
from src.llm.client import LLMClient
from src.profiling.profile import profile_dataset


def _churn_df(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": range(n),
            "tenure_months": range(n),
            "churned": [0, 1] * (n // 2),
        }
    )


def test_understand_usecase_node_forces_ambiguous_when_llm_confidently_picks_an_identifier(monkeypatch):
    df = _churn_df()
    profile = profile_dataset(df)

    def _fake_generate(self, run_id, node, system_prompt, user_prompt, json_schema=None, retries=1):
        assert node == "understand_usecase"
        return {
            "task_type": "classification",
            "target_column": "customer_id",
            "metric": "f1",
            "constraints": [],
            "is_ambiguous": False,
            "ambiguity_reason": None,
        }

    monkeypatch.setattr(LLMClient, "generate", _fake_generate)

    state = {"run_id": "test", "use_case_description": "predict customer churn", "profile": profile}
    result = understand_usecase_node(state)

    assert result["task_spec"]["is_ambiguous"] is True
    assert "customer_id" in result["task_spec"]["ambiguity_reason"]
    assert result["needs_human_confirmation"] is True


def test_understand_usecase_node_leaves_a_real_target_column_alone(monkeypatch):
    df = _churn_df()
    profile = profile_dataset(df)

    def _fake_generate(self, run_id, node, system_prompt, user_prompt, json_schema=None, retries=1):
        return {
            "task_type": "classification",
            "target_column": "churned",
            "metric": "f1",
            "constraints": [],
            "is_ambiguous": False,
            "ambiguity_reason": None,
        }

    monkeypatch.setattr(LLMClient, "generate", _fake_generate)

    state = {"run_id": "test", "use_case_description": "predict customer churn", "profile": profile}
    result = understand_usecase_node(state)

    assert result["task_spec"]["is_ambiguous"] is False
    assert result["task_spec"]["target_column"] == "churned"


def test_apply_feature_plan_node_refuses_to_train_against_an_identifier_target(tmp_path):
    df = _churn_df()
    profile = profile_dataset(df)
    dataset_path = tmp_path / "data.csv"
    df.to_csv(dataset_path, index=False)

    state = {
        "run_id": "test",
        "dataset_path": str(dataset_path),
        "profile": profile,
        "task_spec": {"target_column": "customer_id", "task_type": "classification"},
        "feature_plan": {"steps": []},
        "retry_count": {},
    }

    result = apply_feature_plan_node(state)

    assert result["feature_plan_valid"] is False
    assert any("looks like an identifier" in e for e in result["errors"])
    assert "transformed_dataset_path" not in result


def test_apply_feature_plan_node_allows_a_real_classification_target(tmp_path):
    df = _churn_df()
    profile = profile_dataset(df)
    dataset_path = tmp_path / "data.csv"
    df.to_csv(dataset_path, index=False)

    state = {
        "run_id": "test",
        "dataset_path": str(dataset_path),
        "profile": profile,
        "task_spec": {"target_column": "churned", "task_type": "classification"},
        "feature_plan": {"steps": []},
        "retry_count": {},
    }

    result = apply_feature_plan_node(state)

    assert result["feature_plan_valid"] is True
    assert result["transformed_dataset_path"]
