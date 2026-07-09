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
    # When set, training uses a chronological (leakage-safe) train/test split
    # ordered by this column instead of a random shuffle, and EDA/feature
    # engineering must leave the column intact for that split to happen.
    time_column: Optional[str] = None
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


class FeatureImpact(BaseModel):
    feature: str
    mean_abs_shap: float
    mean_signed_shap: float = 0.0


class ShapPlot(BaseModel):
    title: str
    feature: Optional[str] = None
    image_base64: str
    caption: Optional[str] = None


class KeyInsight(BaseModel):
    tone: Literal["driver", "risk", "minor"]
    message: str


class ExplainabilityResult(BaseModel):
    method: Literal["tree", "linear", "kernel", "unavailable"]
    feature_impact: list[FeatureImpact] = Field(default_factory=list)
    narrative: Optional[str] = None
    note: Optional[str] = None
    summary_plot: Optional[ShapPlot] = None
    bar_plot: Optional[ShapPlot] = None
    dependence_plots: list[ShapPlot] = Field(default_factory=list)
    fidelity_r2: Optional[float] = None
    background_sample_size: int = 0
    key_insights: list[KeyInsight] = Field(default_factory=list)


class CVMetric(BaseModel):
    mean: float
    std: float


class TuningTrial(BaseModel):
    """One completed Optuna trial. Scores are in the metric's natural units
    (rmse stays positive/lower-is-better); best_score is the best seen up to
    and including this trial, so the UI can draw a best-so-far line."""

    trial: int
    score: float
    best_score: float


class TuningInfo(BaseModel):
    """Live + final hyperparameter-tuning state for one training job.
    Updated in the job registry after every trial (the 2s poll loop carries
    it into training_results, so the UI renders progress while tuning runs).
    Trial 0 is always the LLM-proposed baseline hyperparams; best_params is
    the full merged param set the final model was fit with ({} when tuning
    was skipped/disabled and the baseline was used as-is)."""

    enabled: bool = False
    trials_total: int = 0
    trials_done: int = 0
    metric: Optional[str] = None
    lower_is_better: bool = False
    best_params: dict[str, Any] = Field(default_factory=dict)
    history: list[TuningTrial] = Field(default_factory=list)
    note: Optional[str] = None


class TrainingResult(BaseModel):
    run_id: str
    candidate_name: str
    # "timed_out": poll_training's attempt cap was reached while this
    # candidate was still pending/running (src/graph/routing.py) — its
    # background job may still be executing, but this run stopped tracking it.
    status: Literal["pending", "running", "succeeded", "failed", "timed_out"]
    metrics: dict[str, float] = Field(default_factory=dict)
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    model_path: Optional[str] = None
    feature_importance: list[FeatureImportance] = Field(default_factory=list)
    cv_folds: int = 0
    cv_metrics: dict[str, CVMetric] = Field(default_factory=dict)
    cv_note: Optional[str] = None
    tuning: TuningInfo = Field(default_factory=TuningInfo)
    resampling_applied: Optional[str] = None  # "smote" | "random_oversample" | "random_undersample" | None
    resampling_note: Optional[str] = None  # explains an auto-fallback (e.g. SMOTE -> random oversampling)
    explainability: Optional[ExplainabilityResult] = None


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
    # Statistical steps from the approved plan (mean/median impute, scale,
    # target encode) deferred out of apply_feature_plan_node and fit on the
    # training fold only, inside the training job's pipeline — fitting them
    # on the full dataset before the split leaks test-fold statistics (and,
    # for target encoding, each row's own label).
    training_preprocess_steps: list[dict[str, Any]]

    candidate_models: list[dict[str, Any]]
    cv_enabled: bool  # user-configurable at the confirm checkpoint (default True)
    cv_folds: int  # user-requested fold count; auto-reduced per candidate if data can't support it
    tuning_enabled: bool  # Optuna hyperparameter tuning per candidate (confirm checkpoint, default True)
    feature_selection_enabled: bool  # one-shot RFECV with a basic model, subset shared by all candidates
    feature_selection_result: dict[str, Any]  # select_features() output: what was eliminated and why
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
        needs_feature_approval=False,
        feature_plan_approved=False,
        resampling_plan={"enabled": False, "method": "none"},
        training_preprocess_steps=[],
        candidate_models=[],
        cv_enabled=True,
        cv_folds=5,
        tuning_enabled=True,
        feature_selection_enabled=False,
        training_run_ids=[],
        training_results=[],
    )
