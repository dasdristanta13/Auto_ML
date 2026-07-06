"""Deterministic (non-LLM) graph nodes: profiling, human checkpoint, feature
plan application, training dispatch/poll, and evaluation. LLM-backed nodes
live in src/agents/ (CLAUDE.md convention)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.preprocessing import OrdinalEncoder

from src.data_io import load_dataset
from src.profiling.eda import run_eda
from src.profiling.leakage import detect_target_leakage
from src.profiling.profile import profile_dataset
from src.sandbox.execute import SandboxExecutionError, SandboxTimeoutError, SandboxValidationError, dry_run, run_on_full_dataset
from src.state import PipelineState
from src.training.dispatch import poll_training_job, select_features, train_model

TRANSFORMED_DIR = Path("artifacts/transformed")


def _runtime_config() -> dict[str, Any]:
    with open("config/runtime.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def profile_node(state: PipelineState) -> PipelineState:
    if state.get("profile"):
        # experiment runs launched from an existing run reuse its profile —
        # same file, same deterministic output, no point re-reading it
        # (docs/superpowers/specs/2026-07-06-multi-experiment-design.md)
        return state
    df = load_dataset(state["dataset_path"])
    state["profile"] = profile_dataset(df)
    return state


def human_checkpoint_node(state: PipelineState) -> PipelineState:
    """PRD FR-9/FR-10: ambiguous task specs stop here for user confirmation
    rather than being silently guessed. In this local CLI reference
    implementation, confirmation is collected via stdin; a hosted deployment
    would surface this as a UI checkpoint instead."""
    if not state.get("needs_human_confirmation"):
        state["human_confirmed"] = True
        return state

    task_spec = dict(state.get("task_spec", {}))
    print("\n--- Human checkpoint: the platform's understanding is ambiguous ---")
    print(f"Reason: {task_spec.get('ambiguity_reason')}")
    print(f"Inferred so far: {task_spec}")

    target = input(f"Target column [{task_spec.get('target_column') or 'none inferred'}]: ").strip()
    task_type = input(f"Task type [{task_spec.get('task_type') or 'none inferred'}]: ").strip()
    metric = input(f"Success metric [{task_spec.get('metric') or 'none inferred'}]: ").strip()

    if target:
        task_spec["target_column"] = target
    if task_type:
        task_spec["task_type"] = task_type
    if metric:
        task_spec["metric"] = metric
    task_spec["is_ambiguous"] = False

    state["task_spec"] = task_spec
    state["human_confirmed"] = True
    state["needs_human_confirmation"] = False
    return state


def leakage_check_node(state: PipelineState) -> PipelineState:
    df = load_dataset(state["dataset_path"])
    target_column = state.get("task_spec", {}).get("target_column")
    state["leakage_flags"] = detect_target_leakage(df, target_column) if target_column else []
    return state


def eda_node(state: PipelineState) -> PipelineState:
    """Deterministic exploratory analysis (src/profiling/eda.py) — feeds the
    feature_engineering node's prompt and is shown to the user for approval
    before anything gets applied."""
    df = load_dataset(state["dataset_path"])
    result = run_eda(df, state.get("profile", {}), state.get("task_spec", {}), state.get("leakage_flags"))
    state["eda_report"] = {"insights": result["insights"], "suggested_steps": result["suggested_steps"]}
    state["resampling_suggestion"] = result["resampling_suggestion"]
    return state


def feature_approval_checkpoint_node(state: PipelineState) -> PipelineState:
    """CLI-only checkpoint (the API pauses between build_prep_graph and
    build_train_graph instead — see src/api/server.py). Shows the EDA-informed
    feature plan and any resampling suggestion, lets the user approve/reject
    individual steps and the resampling choice via stdin.

    Set AUTOML_AUTO_APPROVE_FEATURES=1 to accept every suggested step as-is
    and decline resampling without prompting — used by automated tests so
    this node never blocks on input() (mirrors AUTOML_MOCK_LLM's convention)."""
    plan = dict(state.get("feature_plan") or {})
    steps = plan.get("steps", [])
    resampling_suggestion = state.get("resampling_suggestion") or {"suggested": False, "method": "none"}

    if os.environ.get("AUTOML_AUTO_APPROVE_FEATURES") == "1":
        state["feature_plan_approved"] = True
        state["resampling_plan"] = {"enabled": False, "method": "none"}
        return state

    print("\n--- Feature engineering plan (review before it's applied) ---")
    approved_steps = []
    for i, step in enumerate(steps):
        origin = "data analysis" if step.get("source") == "eda" else "AI planner"
        print(f"[{i}] {step['op']} on {step['columns']} ({origin}): {step.get('rationale', '')}")
        answer = input("    Apply this step? [Y/n]: ").strip().lower()
        if answer not in ("n", "no"):
            approved_steps.append(step)
    plan["steps"] = approved_steps
    state["feature_plan"] = plan

    resampling_enabled, resampling_method = False, "none"
    if resampling_suggestion.get("suggested"):
        print(f"\nSuggested: {resampling_suggestion.get('reason')}")
        method = resampling_suggestion.get("method", "smote")
        answer = input(f"Apply {method} to balance classes during training? [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            resampling_enabled, resampling_method = True, method
    state["resampling_plan"] = {"enabled": resampling_enabled, "method": resampling_method}
    state["feature_plan_approved"] = True
    return state


def _apply_builtin_step(df: pd.DataFrame, op: str, columns: list[str], params: dict[str, Any]) -> pd.DataFrame:
    """Stateless/structural steps only. Statistical steps (mean/median
    impute, scale, target encode) never reach here — _is_training_time_step
    defers them to the training job so they are fit on the training fold
    only (see src/training/dispatch._build_preprocessor)."""
    df = df.copy()
    if op == "impute":
        strategy = params.get("strategy", "mean")
        for col in columns:
            if strategy == "most_frequent":
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else df[col])
            elif strategy == "constant":
                df[col] = df[col].fillna(params.get("fill_value", 0))
    elif op == "encode":
        method = params.get("method", "onehot")
        if method == "onehot":
            # dtype="int8": pandas >= 2.0 defaults dummy columns to bool, which
            # sklearn's SimpleImputer rejects outright — and the bool dtype
            # survives the CSV round-trip into the training job.
            df = pd.get_dummies(df, columns=columns, dummy_na=False, dtype="int8")
        elif method == "ordinal":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df[columns] = encoder.fit_transform(df[columns].astype(str))
    elif op == "bin":
        n_bins = params.get("n_bins", 5)
        for col in columns:
            df[col] = pd.cut(df[col], bins=n_bins, labels=False)
    elif op == "datetime_decompose":
        for col in columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = parsed.dt.year
            df[f"{col}_month"] = parsed.dt.month
            df[f"{col}_day"] = parsed.dt.day
            df[f"{col}_dayofweek"] = parsed.dt.dayofweek
            df = df.drop(columns=[col])
    elif op == "drop":
        df = df.drop(columns=[c for c in columns if c in df.columns])
    return df


def _is_training_time_step(step: dict[str, Any]) -> bool:
    """Statistical steps whose fitted parameters (means, scale factors,
    category-target means) must come from the training fold only — applying
    them to the full dataset before the split leaks test-fold statistics,
    and full-dataset target encoding leaks each row's own label. These are
    deferred into the training job's pipeline (src/training/dispatch.py).
    Stateless/structural ops (drop, datetime_decompose, onehot/ordinal
    encode, most_frequent/constant impute, bin, custom_code) still run here
    so their plan ordering relative to each other is preserved."""
    op, params = step.get("op"), step.get("params", {})
    if op == "scale":
        return True
    if op == "impute" and params.get("strategy", "mean") in ("mean", "median"):
        return True
    if op == "encode" and params.get("method") == "target":
        return True
    return False


def apply_feature_plan_node(state: PipelineState) -> PipelineState:
    """Applies the validated FeaturePlan's stateless/structural ops; any
    custom_code step is dry-run in the sandbox first, then run on the full
    dataset only after the dry-run succeeds (CLAUDE.md rule #6). Statistical
    steps are deferred to the training job (see _is_training_time_step).
    Output schema is checked afterward (PRD FR-17) before advancing."""
    df = load_dataset(state["dataset_path"])
    plan = state.get("feature_plan", {})
    retry_count = dict(state.get("retry_count", {}))
    deferred_steps: list[dict[str, Any]] = []

    try:
        for step in plan.get("steps", []):
            if _is_training_time_step(step):
                deferred_steps.append(step)
                continue
            op, columns, params = step["op"], step.get("columns", []), step.get("params", {})
            if op == "custom_code":
                code = step["code"]
                dry_run(code, df)  # validated + isolated dry-run on a sample first
                df = run_on_full_dataset(code, df)
            else:
                df = _apply_builtin_step(df, op, columns, params)
    except (SandboxValidationError, SandboxExecutionError, SandboxTimeoutError, KeyError, ValueError) as exc:
        retry_count["feature_engineering"] = retry_count.get("feature_engineering", 0) + 1
        state["retry_count"] = retry_count
        state["feature_plan_valid"] = False
        state["feature_plan_feedback"] = f"applying the plan failed: {exc}"
        state.setdefault("errors", []).append(f"apply_feature_plan: {exc}")
        return state

    target_column = state.get("task_spec", {}).get("target_column")
    if target_column and (target_column not in df.columns or df[target_column].isna().all()):
        retry_count["feature_engineering"] = retry_count.get("feature_engineering", 0) + 1
        state["retry_count"] = retry_count
        state["feature_plan_valid"] = False
        state["feature_plan_feedback"] = f"resulting dataset lost or fully nulled the target column '{target_column}'"
        return state

    TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRANSFORMED_DIR / f"{state['run_id']}.csv"
    df.to_csv(out_path, index=False)
    state["transformed_dataset_path"] = str(out_path)
    state["training_preprocess_steps"] = deferred_steps
    state["feature_plan_valid"] = True
    return state


def dispatch_training_node(state: PipelineState) -> PipelineState:
    task_spec = state.get("task_spec", {})
    cv_enabled = state.get("cv_enabled", True)
    cv_folds = state.get("cv_folds", 5)
    tuning_enabled = state.get("tuning_enabled", True)
    feature_selection_enabled = state.get("feature_selection_enabled", False)
    resampling_plan = state.get("resampling_plan") or {"enabled": False, "method": "none"}

    # feature elimination runs ONCE here with a basic linear model; the chosen
    # subset is shared by every candidate so all models train on the same
    # feature space (2026-07-06-eda-drops-and-rfe-design.md, revision).
    selection: dict[str, Any] = {}
    if feature_selection_enabled:
        selection = select_features(
            state["transformed_dataset_path"],
            task_spec["target_column"],
            task_spec["task_type"],
            task_spec.get("time_column"),
            state.get("training_preprocess_steps", []),
            task_spec.get("metric"),
        )
        state["feature_selection_result"] = selection

    run_ids = []
    for candidate in state.get("candidate_models", []):
        run_id = train_model.invoke(
            {
                "candidate_name": candidate["name"],
                "library": candidate["library"],
                "estimator": candidate["estimator"],
                "hyperparams": candidate.get("hyperparams", {}),
                "dataset_path": state["transformed_dataset_path"],
                "target_column": task_spec["target_column"],
                "task_type": task_spec["task_type"],
                "time_column": task_spec.get("time_column"),
                "preprocess_steps": state.get("training_preprocess_steps", []),
                "cv_enabled": cv_enabled,
                "cv_folds": cv_folds,
                "tuning_enabled": tuning_enabled,
                "tuning_metric": task_spec.get("metric"),
                "selected_features": selection.get("selected_features") or None,
                "feature_selection_note": selection.get("note"),
                "resampling_enabled": resampling_plan.get("enabled", False),
                "resampling_method": resampling_plan.get("method", "none"),
            }
        )
        run_ids.append(run_id)
    state["training_run_ids"] = run_ids
    return state


def poll_training_node(state: PipelineState) -> PipelineState:
    """Loop node: routing.py sends control back here (with backoff) until all
    jobs reach a terminal state or the poll attempt cap is hit — training
    never blocks an LLM call (CLAUDE.md rule #4)."""
    cfg = _runtime_config()["training"]
    retry_count = dict(state.get("retry_count", {}))
    attempt = retry_count.get("poll_training", 0)

    if attempt > 0:
        time.sleep(cfg["poll_interval_seconds"])

    results = [poll_training_job.invoke({"run_id": rid}) for rid in state.get("training_run_ids", [])]
    state["training_results"] = results

    all_terminal = all(r["status"] in ("succeeded", "failed") for r in results)
    if not all_terminal and attempt < cfg["poll_max_attempts"]:
        retry_count["poll_training"] = attempt + 1
        state["retry_count"] = retry_count
    return state


def evaluate_node(state: PipelineState) -> PipelineState:
    """Picks the best model per the task spec's metric (PRD FR-22). If no
    succeeded run actually reports that metric (the LLM may name one the
    evaluator never computes, e.g. "precision", or roc_auc on a multiclass
    target), fall back to the task type's default metric rather than
    declaring the whole run a failure — the fallback is recorded in errors
    so the report can surface it."""
    task_spec = state.get("task_spec", {})
    default_metric = "f1" if task_spec.get("task_type") == "classification" else "rmse"
    metric = task_spec.get("metric") or default_metric

    succeeded = [r for r in state.get("training_results", []) if r["status"] == "succeeded"]
    with_metric = [r for r in succeeded if metric in r.get("metrics", {})]
    if not with_metric and succeeded and metric != default_metric:
        state.setdefault("errors", []).append(
            f"evaluate: metric '{metric}' unavailable on trained candidates; fell back to '{default_metric}'"
        )
        metric = default_metric
        with_metric = [r for r in succeeded if metric in r.get("metrics", {})]

    if not with_metric:
        state["best_model"] = {}
        state.setdefault("errors", []).append("evaluate: no candidate produced a usable result for the target metric")
        return state

    lower_is_better = metric in ("rmse", "mae")
    best = min(with_metric, key=lambda r: r["metrics"][metric]) if lower_is_better else max(
        with_metric, key=lambda r: r["metrics"][metric]
    )
    state["best_model"] = best
    return state
