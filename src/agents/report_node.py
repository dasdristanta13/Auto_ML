"""LLM-backed node: generates the final plain-language report (PRD FR-23/FR-24).

Always renders even on a failed/capped-out run — a pipeline must terminate
in a clear user-facing explanation, never a silent hang (CLAUDE.md rule #3 /
PRD Reliability NFR).
"""

from __future__ import annotations

from src.agents.prompt_utils import render_prompt
from src.llm.client import get_llm_client
from src.state import PipelineState


def report_node(state: PipelineState) -> PipelineState:
    client = get_llm_client()
    system_prompt = render_prompt(
        "report.md",
        TASK_SPEC_JSON=state.get("task_spec", {}),
        FEATURE_PLAN_JSON=state.get("feature_plan", {}),
        LEAKAGE_FLAGS_JSON=state.get("leakage_flags", []),
        TRAINING_RESULTS_JSON=state.get("training_results", []),
        BEST_MODEL_JSON=state.get("best_model", {}),
    )
    narrative = client.generate(
        run_id=state["run_id"],
        node="report",
        system_prompt=system_prompt,
        user_prompt="Write the final report now.",
        json_schema=None,
    )

    state["report"] = {
        "narrative": narrative,
        "task_spec": state.get("task_spec", {}),
        "leakage_flags": state.get("leakage_flags", []),
        "training_results": state.get("training_results", []),
        "best_model": state.get("best_model", {}),
        "errors": state.get("errors", []),
    }
    if state.get("status") != "failed":
        state["status"] = "completed"
    return state
