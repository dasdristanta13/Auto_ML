"""LLM-backed node: proposes a shortlist of candidate models fitting the task
type and the dataset's actual characteristics (PRD FR-18)."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from src.agents.prompt_utils import render_prompt
from src.llm.client import get_llm_client
from src.state import CandidateModel, PipelineState


class _CandidateModelList(BaseModel):
    candidates: list[CandidateModel] = Field(default_factory=list)


def model_selection_node(state: PipelineState) -> PipelineState:
    client = get_llm_client()
    system_prompt = render_prompt(
        "model_selection.md",
        TASK_SPEC_JSON=state.get("task_spec", {}),
        PROFILE_JSON=state.get("profile", {}),
    )
    raw = client.generate(
        run_id=state["run_id"],
        node="model_selection",
        system_prompt=system_prompt,
        user_prompt="Return the candidate model list JSON now.",
        json_schema=_CandidateModelList.model_json_schema(),
    )

    try:
        parsed = _CandidateModelList(**raw)
        candidates = parsed.candidates
    except ValidationError as exc:
        state.setdefault("errors", []).append(f"model_selection: {exc}")
        candidates = []

    state["candidate_models"] = [c.model_dump() for c in candidates]
    return state
