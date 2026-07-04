"""LLM-backed node: parses the user's natural-language use case + the dataset
profile into a structured TaskSpec (PRD FR-8/FR-9)."""

from __future__ import annotations

from pydantic import ValidationError

from src.agents.prompt_utils import render_prompt
from src.llm.client import get_llm_client
from src.state import PipelineState, TaskSpec


def understand_usecase_node(state: PipelineState) -> PipelineState:
    client = get_llm_client()
    system_prompt = render_prompt(
        "understand_usecase.md",
        USE_CASE_DESCRIPTION=state["use_case_description"],
        PROFILE_JSON=state["profile"],
    )
    raw = client.generate(
        run_id=state["run_id"],
        node="understand_usecase",
        system_prompt=system_prompt,
        user_prompt="Return the task specification JSON now.",
        json_schema=TaskSpec.model_json_schema(),
    )

    try:
        task_spec = TaskSpec(**raw)
    except ValidationError as exc:
        state.setdefault("errors", []).append(f"understand_usecase: {exc}")
        task_spec = TaskSpec(is_ambiguous=True, ambiguity_reason=f"could not parse model output: {exc}")

    state["task_spec"] = task_spec.model_dump()
    # FR-9: never silently guess on an ambiguous task spec — route to a human checkpoint.
    state["needs_human_confirmation"] = task_spec.is_ambiguous
    return state
