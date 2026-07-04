"""LLM-backed node: emits a structured FeaturePlan (CLAUDE.md rule #2 —
structured plans preferred over free-form code). Any custom_code step is
statically AST-validated here; the actual sandboxed dry-run + full execution
happens in the deterministic apply_feature_plan node (src/graph/nodes.py).

Retry loop: on an invalid plan, increments state["retry_count"]["feature_engineering"]
and stores feedback for the next attempt. routing.py checks this against
config/runtime.yaml retry.max_retries (CLAUDE.md rule #3) before deciding
whether to loop back here again or fall back to the report node.
"""

from __future__ import annotations

from pydantic import ValidationError

from src.agents.prompt_utils import render_prompt
from src.llm.client import get_llm_client
from src.sandbox.validate import validate_code
from src.state import FeaturePlan, PipelineState


def _validate_plan(raw: dict) -> tuple[FeaturePlan | None, list[str]]:
    errors: list[str] = []
    try:
        plan = FeaturePlan(**raw)
    except ValidationError as exc:
        return None, [str(exc)]

    for step in plan.steps:
        if step.op == "custom_code":
            if not step.code:
                errors.append("custom_code step is missing `code`")
                continue
            result = validate_code(step.code)
            if not result.valid:
                errors.append(f"custom_code step for columns {step.columns} rejected: {'; '.join(result.errors)}")

    return (plan if not errors else None), errors


def feature_engineering_node(state: PipelineState) -> PipelineState:
    client = get_llm_client()
    system_prompt = render_prompt(
        "feature_engineering.md",
        TASK_SPEC_JSON=state.get("task_spec", {}),
        LEAKAGE_FLAGS_JSON=state.get("leakage_flags", []),
        PROFILE_JSON=state.get("profile", {}),
        PRIOR_ATTEMPT_FEEDBACK=(
            f"## Your previous attempt was rejected\n{state['feature_plan_feedback']}"
            if state.get("feature_plan_feedback")
            else ""
        ),
    )
    raw = client.generate(
        run_id=state["run_id"],
        node="feature_engineering",
        system_prompt=system_prompt,
        user_prompt="Return the feature engineering plan JSON now.",
        json_schema=FeaturePlan.model_json_schema(),
    )

    plan, errors = _validate_plan(raw)
    retry_count = dict(state.get("retry_count", {}))

    if plan is None:
        retry_count["feature_engineering"] = retry_count.get("feature_engineering", 0) + 1
        state["retry_count"] = retry_count
        state["feature_plan_valid"] = False
        state["feature_plan_feedback"] = "; ".join(errors)
        state.setdefault("errors", []).append(f"feature_engineering attempt rejected: {'; '.join(errors)}")
        state["feature_plan"] = raw
        return state

    state["feature_plan"] = plan.model_dump()
    state["feature_plan_valid"] = True
    state["feature_plan_feedback"] = ""
    return state
