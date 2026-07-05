"""LLM-backed node: proposes a shortlist of candidate models fitting the task
type and the dataset's actual characteristics (PRD FR-18)."""

from __future__ import annotations

import importlib.util
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from src.agents.prompt_utils import render_prompt
from src.llm.client import get_llm_client
from src.state import CandidateModel, PipelineState


class _CandidateModelList(BaseModel):
    candidates: list[CandidateModel] = Field(default_factory=list)


# Deterministic completeness floor (CLAUDE.md rule #2: structured output over
# free-form judgment calls). The LLM's shortlist is a rationale/hyperparameter
# source, not a gatekeeper — every model family applicable to the task type
# must be tried so "all possible models" isn't at the mercy of what the LLM
# happened to propose.
_CANONICAL_ESTIMATORS: dict[str, list[tuple[str, str, str, dict[str, Any]]]] = {
    "classification": [
        ("Logistic Regression", "sklearn", "LogisticRegression", {"max_iter": 1000}),
        ("Random Forest", "sklearn", "RandomForestClassifier", {"n_estimators": 200, "random_state": 0}),
        ("Gradient Boosting", "sklearn", "GradientBoostingClassifier", {"random_state": 0}),
        ("XGBoost", "xgboost", "XGBClassifier", {"n_estimators": 200, "eval_metric": "logloss", "random_state": 0}),
        ("LightGBM", "lightgbm", "LGBMClassifier", {"n_estimators": 200, "random_state": 0}),
    ],
    "regression": [
        ("Linear Regression", "sklearn", "LinearRegression", {}),
        ("Ridge Regression", "sklearn", "Ridge", {"alpha": 1.0}),
        ("Random Forest", "sklearn", "RandomForestRegressor", {"n_estimators": 200, "random_state": 0}),
        ("Gradient Boosting", "sklearn", "GradientBoostingRegressor", {"random_state": 0}),
        ("XGBoost", "xgboost", "XGBRegressor", {"n_estimators": 200, "random_state": 0}),
        ("LightGBM", "lightgbm", "LGBMRegressor", {"n_estimators": 200, "random_state": 0}),
    ],
}


_LIBRARY_IMPORT_NAMES = {"sklearn": "sklearn", "xgboost": "xgboost", "lightgbm": "lightgbm"}


def _library_available(library: str) -> bool:
    """xgboost/lightgbm are optional installs (requirements.txt) — the
    completeness floor must not inject a candidate that would fail on import
    and burn a training slot every run."""
    module = _LIBRARY_IMPORT_NAMES.get(library)
    return module is not None and importlib.util.find_spec(module) is not None


def _fill_missing_candidates(candidates: list[CandidateModel], task_type: Optional[str]) -> list[CandidateModel]:
    canonical = _CANONICAL_ESTIMATORS.get(task_type or "", [])
    present = {(c.library, c.estimator) for c in candidates}
    filled = list(candidates)
    for name, library, estimator, hyperparams in canonical:
        if (library, estimator) not in present and _library_available(library):
            filled.append(
                CandidateModel(
                    name=name,
                    library=library,
                    estimator=estimator,
                    hyperparams=dict(hyperparams),
                    rationale="Included automatically for full coverage of applicable model families.",
                )
            )
    return filled


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

    task_type = state.get("task_spec", {}).get("task_type")
    candidates = _fill_missing_candidates(candidates, task_type)

    state["candidate_models"] = [c.model_dump() for c in candidates]
    return state
