"""Regression test: feature_engineering_node must reject a plan that
references a column not present in the dataset, instead of letting it through
to human approval and crashing apply_feature_plan later.

This reproduces a real incident: the LLM proposed a placeholder "encode" step
for a column called "dummy_target_column" that didn't exist anywhere in the
dataset ("included for extensibility if future columns appear"). The plan
validated as structurally correct (right JSON shape) and was approved by a
human, then apply_feature_plan failed with a pandas KeyError-shaped message,
skipping model_selection/dispatch_training/poll_training/evaluate entirely
and jumping straight to a failed report."""

from __future__ import annotations

import pandas as pd

from src.agents.feature_engineering_node import _known_columns, _validate_plan, feature_engineering_node
from src.llm.client import LLMClient
from src.profiling.profile import profile_dataset


def _hallucinated_plan_response() -> dict:
    return {
        "steps": [
            {
                "op": "drop",
                "columns": ["customer_id"],
                "params": {},
                "rationale": "identifier column",
            },
            {
                "op": "encode",
                "columns": ["dummy_target_column"],
                "params": {"method": "onehot"},
                "rationale": "placeholder for future categorical columns",
            },
        ],
        "plan_rationale": "drop identifier, placeholder encode",
    }


def test_validate_plan_rejects_step_referencing_a_nonexistent_column():
    known_columns = {"customer_id", "tenure_months", "monthly_spend", "churned"}
    plan, errors = _validate_plan(_hallucinated_plan_response(), known_columns, target_column="churned")

    assert plan is None
    assert any("dummy_target_column" in e for e in errors)


def test_validate_plan_rejects_step_that_touches_the_target_column():
    """Reproduces a second real incident: the confirm step's target_column was
    mistakenly set to an identifier-like column ("customer_id" instead of
    "churned"). feature_engineering_node correctly flagged customer_id as an
    identifier and dropped it — but since it was (mis)configured as the
    target, this silently destroyed the label. apply_feature_plan_node caught
    the lost target downstream, but only after a human had already approved
    the plan, surfacing a vague, unhelpful failure instead of a clear one."""
    known_columns = {"customer_id", "tenure_months", "monthly_spend", "churned"}
    raw = {
        "steps": [
            {"op": "drop", "columns": ["customer_id"], "params": {}, "rationale": "looks like an identifier"},
        ],
        "plan_rationale": "drop identifier",
    }
    plan, errors = _validate_plan(raw, known_columns, target_column="customer_id")

    assert plan is None
    assert any("target column" in e and "customer_id" in e for e in errors)


def test_known_columns_covers_both_narrow_and_wide_profile_shapes():
    narrow_profile = {"columns": {"a": {}, "b": {}}}
    assert _known_columns(narrow_profile) == {"a", "b"}

    wide_profile = {
        "columns": {"cat_col": {}},
        "numeric_summary": {"numeric_clusters": [{"member_columns": ["n1", "n2"]}, {"member_columns": ["n3"]}]},
    }
    assert _known_columns(wide_profile) == {"cat_col", "n1", "n2", "n3"}


def test_feature_engineering_node_retries_instead_of_approving_a_hallucinated_column(monkeypatch):
    df = pd.DataFrame(
        {
            "customer_id": range(50),
            "tenure_months": range(50),
            "monthly_spend": range(50),
            "churned": [0, 1] * 25,
        }
    )
    profile = profile_dataset(df)

    def _fake_generate(self, run_id, node, system_prompt, user_prompt, json_schema=None, retries=1):
        assert node == "feature_engineering"
        return _hallucinated_plan_response()

    monkeypatch.setattr(LLMClient, "generate", _fake_generate)

    state = {
        "run_id": "test",
        "task_spec": {"target_column": "churned", "task_type": "classification"},
        "profile": profile,
        "leakage_flags": [],
        "eda_report": {"suggested_steps": []},
        "retry_count": {},
    }

    result = feature_engineering_node(state)

    assert result["feature_plan_valid"] is False
    assert "dummy_target_column" in result["feature_plan_feedback"]
    assert result["retry_count"]["feature_engineering"] == 1


def test_feature_engineering_node_retries_instead_of_approving_a_plan_that_drops_the_target(monkeypatch):
    df = pd.DataFrame(
        {
            "customer_id": range(50),
            "tenure_months": range(50),
            "monthly_spend": range(50),
            "churned": [0, 1] * 25,
        }
    )
    profile = profile_dataset(df)

    def _fake_generate(self, run_id, node, system_prompt, user_prompt, json_schema=None, retries=1):
        assert node == "feature_engineering"
        return {
            "steps": [
                {"op": "drop", "columns": ["customer_id"], "params": {}, "rationale": "looks like an identifier"},
            ],
            "plan_rationale": "drop identifier",
        }

    monkeypatch.setattr(LLMClient, "generate", _fake_generate)

    # target_column mistakenly set to the identifier column instead of "churned"
    state = {
        "run_id": "test",
        "task_spec": {"target_column": "customer_id", "task_type": "classification"},
        "profile": profile,
        "leakage_flags": [],
        "eda_report": {"suggested_steps": []},
        "retry_count": {},
    }

    result = feature_engineering_node(state)

    assert result["feature_plan_valid"] is False
    assert "target column" in result["feature_plan_feedback"]
    assert result["retry_count"]["feature_engineering"] == 1
