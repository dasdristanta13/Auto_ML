"""explainability_node (src/agents/explainability_node.py) computes SHAP for
the winning model only, then narrates it — degrading gracefully when there's
no winning model yet, SHAP couldn't explain it, or the LLM call fails."""

from __future__ import annotations

import src.agents.explainability_node as explainability_node_module
from src.agents.explainability_node import explainability_node
from src.llm.client import LLMClient
from src.state import new_state


def test_explainability_node_narrates_computed_shap(monkeypatch, tmp_path):
    monkeypatch.setattr(
        explainability_node_module,
        "compute_explainability",
        lambda model_path, transformed_dataset_path: {
            "method": "tree",
            "feature_impact": [{"feature": "age", "mean_abs_shap": 0.4}],
            "narrative": None,
            "note": None,
        },
    )
    captured = {}

    def _fake_generate(self, run_id, node, system_prompt, user_prompt, json_schema=None, retries=1):
        captured.update(node=node, system_prompt=system_prompt)
        return "Age drives most predictions."

    monkeypatch.setattr(LLMClient, "generate", _fake_generate)

    state = new_state(run_id="expl-node-1", dataset_path="unused.csv", use_case_description="test")
    state["transformed_dataset_path"] = "unused.csv"
    state["best_model"] = {"model_path": str(tmp_path / "model.joblib"), "candidate_name": "rf"}

    result = explainability_node(state)

    assert result["best_model"]["explainability"]["method"] == "tree"
    assert result["best_model"]["explainability"]["narrative"] == "Age drives most predictions."
    assert captured["node"] == "explainability"
    assert "age" in captured["system_prompt"]


def test_explainability_node_skips_narrative_when_shap_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        explainability_node_module,
        "compute_explainability",
        lambda model_path, transformed_dataset_path: {
            "method": "unavailable", "feature_impact": [], "narrative": None,
            "note": "SHAP explanation unavailable for this model: boom",
        },
    )

    def _fail_generate(self, *a, **kw):
        raise AssertionError("LLM should not be called when SHAP is unavailable")

    monkeypatch.setattr(LLMClient, "generate", _fail_generate)

    state = new_state(run_id="expl-node-2", dataset_path="unused.csv", use_case_description="test")
    state["best_model"] = {"model_path": str(tmp_path / "model.joblib")}

    result = explainability_node(state)

    assert result["best_model"]["explainability"]["method"] == "unavailable"
    assert result["best_model"]["explainability"]["narrative"] is None


def test_explainability_node_tolerates_llm_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        explainability_node_module,
        "compute_explainability",
        lambda model_path, transformed_dataset_path: {
            "method": "linear", "feature_impact": [{"feature": "age", "mean_abs_shap": 0.1}],
            "narrative": None, "note": None,
        },
    )

    def _fake_generate(self, *a, **kw):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(LLMClient, "generate", _fake_generate)

    state = new_state(run_id="expl-node-3", dataset_path="unused.csv", use_case_description="test")
    state["best_model"] = {"model_path": str(tmp_path / "model.joblib")}

    result = explainability_node(state)

    assert result["best_model"]["explainability"]["method"] == "linear"
    assert result["best_model"]["explainability"]["narrative"] is None
    assert any("explainability" in e for e in result["errors"])


def test_explainability_node_noop_when_no_model_path():
    state = new_state(run_id="expl-node-4", dataset_path="unused.csv", use_case_description="test")
    state["best_model"] = {}

    result = explainability_node(state)

    assert result["best_model"] == {}
