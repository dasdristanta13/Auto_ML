"""LLM-backed node: computes aggregate SHAP feature impact (+ plots) for the
winning model, narrates its top drivers in plain language, and captions each
generated plot. Runs once, only for best_model (not every candidate) — SHAP +
LLM calls for every discarded candidate would multiply cost for data that's
thrown away."""

from __future__ import annotations

from src.agents.prompt_utils import render_prompt
from src.llm.client import get_llm_client
from src.state import PipelineState
from src.training.dispatch import compute_explainability

_CAPTIONS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary_plot_caption": {"type": "string"},
        "bar_plot_caption": {"type": "string"},
        "dependence_plot_captions": {"type": "object"},
        "key_insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tone": {"type": "string", "enum": ["driver", "risk", "minor"]},
                    "message": {"type": "string"},
                },
            },
        },
    },
}


def explainability_node(state: PipelineState) -> PipelineState:
    best_model = state.get("best_model") or {}
    model_path = best_model.get("model_path")
    if not model_path:
        return state

    result = compute_explainability(model_path, state.get("transformed_dataset_path", ""))
    result.setdefault("key_insights", [])

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

        dependence_plots = result.get("dependence_plots") or []
        has_plots = result.get("summary_plot") or result.get("bar_plot") or dependence_plots
        if has_plots:
            captions_prompt = render_prompt(
                "explainability_plot_captions.md",
                FEATURE_IMPACT_JSON=result["feature_impact"],
                DEPENDENCE_FEATURES_JSON=[p["feature"] for p in dependence_plots],
            )
            try:
                captions = client.generate(
                    run_id=state["run_id"],
                    node="explainability_captions",
                    system_prompt=captions_prompt,
                    user_prompt="Write the plot captions now.",
                    json_schema=_CAPTIONS_JSON_SCHEMA,
                )
                if result.get("summary_plot"):
                    result["summary_plot"]["caption"] = captions.get("summary_plot_caption")
                if result.get("bar_plot"):
                    result["bar_plot"]["caption"] = captions.get("bar_plot_caption")
                dependence_captions = captions.get("dependence_plot_captions") or {}
                for plot in dependence_plots:
                    plot["caption"] = dependence_captions.get(plot["feature"])
                result["key_insights"] = captions.get("key_insights") or []
            except Exception as exc:  # noqa: BLE001 - captions are enrichment, never fatal
                state.setdefault("errors", []).append(f"explainability: plot captions unavailable: {exc}")

    best_model["explainability"] = result
    state["best_model"] = best_model
    return state
