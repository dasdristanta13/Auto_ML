"""LLM-backed node: parses the user's natural-language use case + the dataset
profile into a structured TaskSpec (PRD FR-8/FR-9)."""

from __future__ import annotations

from pydantic import ValidationError

from src.agents.prompt_utils import render_prompt
from src.llm.client import get_llm_client
from src.profiling.heuristics import looks_like_identifier
from src.state import PipelineState, TaskSpec


def _target_looks_like_an_identifier(profile: dict, target_column: str) -> bool:
    row_count = profile.get("row_count", 0)
    col = (profile.get("columns") or {}).get(target_column)
    if not col or not row_count:
        return False
    return looks_like_identifier(target_column, col.get("dtype", ""), col.get("n_unique", 0), row_count)


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

    # Defense in depth: the prompt already tells the model never to pick an
    # identifier column as target, but a cheap/small model can still do it
    # confidently (is_ambiguous=false) — this has caused real training
    # failures (e.g. classifying against a near-unique "customer_id"). Force
    # the existing human checkpoint open rather than trust the model's
    # confidence here, regardless of what it claims.
    if task_spec.target_column and not task_spec.is_ambiguous:
        profile = state.get("profile", {})
        if _target_looks_like_an_identifier(profile, task_spec.target_column):
            task_spec = task_spec.model_copy(
                update={
                    "is_ambiguous": True,
                    "ambiguity_reason": (
                        f"'{task_spec.target_column}' looks like an identifier column (near-unique per row), "
                        "not a predictive label — please double-check the target column below."
                    ),
                }
            )

    state["task_spec"] = task_spec.model_dump()
    # FR-9: never silently guess on an ambiguous task spec — route to a human checkpoint.
    state["needs_human_confirmation"] = task_spec.is_ambiguous
    return state
