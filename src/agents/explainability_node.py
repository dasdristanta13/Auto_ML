"""LLM-backed node: computes aggregate SHAP feature impact for the winning
model and narrates its top drivers in plain language. Runs once, only for
best_model (not every candidate) — SHAP + an LLM call for every discarded
candidate would multiply cost for data that's thrown away."""

from __future__ import annotations

from src.agents.prompt_utils import render_prompt
from src.llm.client import get_llm_client
from src.state import PipelineState
from src.training.dispatch import compute_explainability


def explainability_node(state: PipelineState) -> PipelineState:
    best_model = state.get("best_model") or {}
    model_path = best_model.get("model_path")
    if not model_path:
        return state

    result = compute_explainability(model_path, state.get("transformed_dataset_path", ""))

    if result["method"] != "unavailable":
        client = get_llm_client()
        system_prompt = render_prompt("explainability.md", FEATURE_IMPACT_JSON=result["feature_impact"])
        try:
            result["narrative"] = client.generate(
                run_id=state["run_id"],
                node="explainability",
                system_prompt=system_prompt,
                user_prompt="Write the explanation now.",
                json_schema=None,
            )
        except Exception as exc:  # noqa: BLE001 - narrative is enrichment, never fatal
            state.setdefault("errors", []).append(f"explainability: narrative unavailable: {exc}")

    best_model["explainability"] = result
    state["best_model"] = best_model
    return state
