"""LLM-backed node: diagnoses a failed training candidate's error and
proposes a corrected hyperparameter set for a single, capped retry
(CLAUDE.md rule #3 — every loop-back edge has a retry cap; here the cap is
tracked per candidate name rather than globally, since one candidate's bad
hyperparameters shouldn't block retrying the others).

Structured output only (CLAUDE.md rule #2): reuses CandidateModel, the same
schema model_selection_node already emits. This never touches src/sandbox/ —
that pipeline exists for LLM-generated executable code, not hyperparameter
JSON."""

from __future__ import annotations

import yaml
from pydantic import ValidationError

from src.agents.prompt_utils import render_prompt
from src.llm.client import get_llm_client
from src.state import CandidateModel, PipelineState
from src.training.dispatch import train_model


def _max_retries() -> int:
    with open("config/runtime.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["retry"]["max_retries"]


def train_candidate_repair_node(state: PipelineState) -> PipelineState:
    task_spec = state.get("task_spec", {})
    cv_enabled = state.get("cv_enabled", True)
    cv_folds = state.get("cv_folds", 5)
    resampling_plan = state.get("resampling_plan") or {"enabled": False, "method": "none"}
    max_retries = _max_retries()

    repair_count = dict(state.get("candidate_repair_count", {}))
    repair_log = list(state.get("training_repair_log", []))
    candidates_by_name = {c["name"]: dict(c) for c in state.get("candidate_models", [])}
    run_ids = list(state.get("training_run_ids", []))
    results = state.get("training_results", [])

    client = get_llm_client()

    for i, result in enumerate(results):
        if result.get("status") != "failed":
            continue
        name = result.get("candidate_name")
        attempts = repair_count.get(name, 0)
        if attempts >= max_retries:
            continue
        candidate = candidates_by_name.get(name)
        if not candidate:
            continue

        system_prompt = render_prompt(
            "train_candidate_repair.md",
            CANDIDATE_NAME=name,
            LIBRARY=candidate["library"],
            ESTIMATOR=candidate["estimator"],
            TASK_SPEC_JSON=task_spec,
            ORIGINAL_HYPERPARAMS_JSON=candidate.get("hyperparams", {}),
            ERROR_MESSAGE=result.get("error") or "unknown error",
        )
        raw = client.generate(
            run_id=state["run_id"],
            node="train_candidate_repair",
            system_prompt=system_prompt,
            user_prompt="Return the corrected candidate JSON now.",
            json_schema=CandidateModel.model_json_schema(),
        )

        try:
            repaired = CandidateModel(**raw)
        except ValidationError as exc:
            state.setdefault("errors", []).append(f"train_candidate_repair: {name}: {exc}")
            continue

        # Only the LLM's hyperparams/rationale are trusted — identity fields
        # are forced back to the original, vetted candidate (mirrors
        # feature_engineering_node forcing source="llm" after validation).
        repaired = repaired.model_copy(
            update={"name": name, "library": candidate["library"], "estimator": candidate["estimator"]}
        )

        new_run_id = train_model.invoke(
            {
                "candidate_name": repaired.name,
                "library": repaired.library,
                "estimator": repaired.estimator,
                "hyperparams": repaired.hyperparams,
                "dataset_path": state["transformed_dataset_path"],
                "target_column": task_spec["target_column"],
                "task_type": task_spec["task_type"],
                "cv_enabled": cv_enabled,
                "cv_folds": cv_folds,
                "resampling_enabled": resampling_plan.get("enabled", False),
                "resampling_method": resampling_plan.get("method", "none"),
            }
        )
        run_ids[i] = new_run_id
        candidates_by_name[name] = repaired.model_dump()
        repair_count[name] = attempts + 1
        repair_log.append(
            {
                "candidate_name": name,
                "attempt": attempts + 1,
                "original_error": result.get("error"),
                "new_hyperparams": repaired.hyperparams,
                "rationale": repaired.rationale,
            }
        )

    state["training_run_ids"] = run_ids
    state["candidate_models"] = list(candidates_by_name.values())
    state["candidate_repair_count"] = repair_count
    state["training_repair_log"] = repair_log
    return state
