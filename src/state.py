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
    # provenance for the feature-approval UI: "llm" (feature_engineering_node's
    # own proposal) vs "eda" (deterministic completeness-floor fill-in from
    # src/profiling/eda.py, mirroring model_selection_node's candidate floor).
    source: Literal["llm", "eda"] = "llm"


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


class CVMetric(BaseModel):
    mean: float
    std: float


class TrainingResult(BaseModel):
    run_id: str
    candidate_name: str
    status: Literal["pending", "running", "succeeded", "failed"]
    metrics: dict[str, float] = Field(default_factory=dict)
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    model_path: Optional[str] = None
    feature_importance: list[FeatureImportance] = Field(default_factory=list)
    cv_folds: int = 0
    cv_metrics: dict[str, CVMetric] = Field(default_factory=dict)
    cv_note: Optional[str] = None
    resampling_applied: Optional[str] = None  # "smote" | "random_oversample" | "random_undersample" | None
    resampling_note: Optional[str] = None  # explains an auto-fallback (e.g. SMOTE -> random oversampling)


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

    eda_report: dict[str, Any]  # {"insights": [...], "suggested_steps": [...]} — src/profiling/eda.py
    resampling_suggestion: dict[str, Any]  # EDA's recommendation; resampling_plan below is the user's actual choice
    resampling_plan: dict[str, Any]  # {"enabled": bool, "method": "smote"|"random_oversample"|"random_undersample"|"none"}

    feature_plan: dict[str, Any]  # FeaturePlan.model_dump()
    feature_plan_valid: bool
    feature_plan_feedback: str  # fed back into the prompt on retry
    needs_feature_approval: bool
    feature_plan_approved: bool
    transformed_dataset_path: str

    candidate_models: list[dict[str, Any]]
    cv_enabled: bool  # user-configurable at the confirm checkpoint (default True)
    cv_folds: int  # user-requested fold count; auto-reduced per candidate if data can't support it
    training_run_ids: list[str]
    training_results: list[dict[str, Any]]
    candidate_repair_count: dict[str, int]  # keyed by candidate name, capped by config/runtime.yaml retry.max_retries
    training_repair_log: list[dict[str, Any]]  # audit trail: {candidate_name, attempt, original_error, new_hyperparams, rationale}
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
        needs_feature_approval=False,
        feature_plan_approved=False,
        resampling_plan={"enabled": False, "method": "none"},
        candidate_models=[],
        cv_enabled=True,
        cv_folds=5,
        training_run_ids=[],
        training_results=[],
        candidate_repair_count={},
        training_repair_log=[],
    )
