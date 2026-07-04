"""PipelineState — the single source of truth for data flowing between graph nodes.

Every node reads/writes this schema. Do not pass ad hoc dicts between nodes;
extend this file first if a new field is needed (see CLAUDE.md rule #1).
"""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

TaskType = Literal["classification", "regression", "forecasting", "clustering"]


class TaskSpec(BaseModel):
    """Structured interpretation of the user's natural-language use case."""

    task_type: Optional[TaskType] = None
    target_column: Optional[str] = None
    metric: Optional[str] = None
    constraints: list[str] = Field(default_factory=list)
    is_ambiguous: bool = True
    ambiguity_reason: Optional[str] = None


class FeatureStep(BaseModel):
    """One structured, schema-validated transformation step (rule #2: structured over free-form)."""

    op: Literal[
        "impute",
        "encode",
        "scale",
        "bin",
        "datetime_decompose",
        "drop",
        "custom_code",
    ]
    columns: list[str]
    params: dict[str, Any] = Field(default_factory=dict)
    # Only populated when op == "custom_code"; must pass src/sandbox/validate.py
    # before it is ever executed.
    code: Optional[str] = None
    rationale: str = ""


class FeaturePlan(BaseModel):
    steps: list[FeatureStep] = Field(default_factory=list)
    plan_rationale: str = ""


class CandidateModel(BaseModel):
    name: str
    library: Literal["sklearn", "xgboost", "lightgbm"]
    estimator: str
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class TrainingResult(BaseModel):
    run_id: str
    candidate_name: str
    status: Literal["pending", "running", "succeeded", "failed"]
    metrics: dict[str, float] = Field(default_factory=dict)
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    model_path: Optional[str] = None
    feature_importance: list[FeatureImportance] = Field(default_factory=list)


class PipelineState(TypedDict, total=False):
    run_id: str
    dataset_path: str
    use_case_description: str

    profile: dict[str, Any]  # PII-redacted statistical profile
    pii_report: dict[str, Any]
    leakage_flags: list[dict[str, Any]]

    task_spec: dict[str, Any]  # TaskSpec.model_dump()
    needs_human_confirmation: bool
    human_confirmed: bool

    feature_plan: dict[str, Any]  # FeaturePlan.model_dump()
    feature_plan_valid: bool
    feature_plan_feedback: str  # fed back into the prompt on retry
    transformed_dataset_path: str

    candidate_models: list[dict[str, Any]]
    training_run_ids: list[str]
    training_results: list[dict[str, Any]]
    best_model: dict[str, Any]

    report: dict[str, Any]

    retry_count: dict[str, int]  # keyed by node name, checked against config/runtime.yaml
    errors: list[str]
    status: Literal["running", "awaiting_human", "completed", "failed"]


def new_state(run_id: str, dataset_path: str, use_case_description: str) -> PipelineState:
    return PipelineState(
        run_id=run_id,
        dataset_path=dataset_path,
        use_case_description=use_case_description,
        retry_count={},
        errors=[],
        status="running",
        needs_human_confirmation=False,
        human_confirmed=False,
        feature_plan_valid=False,
        candidate_models=[],
        training_run_ids=[],
        training_results=[],
    )
