"""Deterministic (non-LLM) graph nodes: profiling, human checkpoint, feature
plan application, training dispatch/poll, and evaluation. LLM-backed nodes
live in src/agents/ (CLAUDE.md convention)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, RobustScaler, StandardScaler

from src.profiling.leakage import detect_target_leakage
from src.profiling.profile import profile_dataset
from src.sandbox.execute import SandboxExecutionError, SandboxTimeoutError, SandboxValidationError, dry_run, run_on_full_dataset
from src.state import PipelineState
from src.training.dispatch import poll_training_job, train_model

TRANSFORMED_DIR = Path("artifacts/transformed")


def _runtime_config() -> dict[str, Any]:
    with open("config/runtime.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def profile_node(state: PipelineState) -> PipelineState:
    df = pd.read_csv(state["dataset_path"])
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
    df = pd.read_csv(state["dataset_path"])
    target_column = state.get("task_spec", {}).get("target_column")
    state["leakage_flags"] = detect_target_leakage(df, target_column) if target_column else []
    return state


def _apply_builtin_step(df: pd.DataFrame, op: str, columns: list[str], params: dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    if op == "impute":
        strategy = params.get("strategy", "mean")
        for col in columns:
            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "most_frequent":
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else df[col])
            elif strategy == "constant":
                df[col] = df[col].fillna(params.get("fill_value", 0))
    elif op == "encode":
        method = params.get("method", "onehot")
        if method == "onehot":
            df = pd.get_dummies(df, columns=columns, dummy_na=False)
        elif method == "ordinal":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df[columns] = encoder.fit_transform(df[columns].astype(str))
        elif method == "target":
            target_col = params.get("target_column")
            for col in columns:
                if target_col and target_col in df.columns:
                    means = df.groupby(col)[target_col].transform("mean")
                    df[col] = means
    elif op == "scale":
        method = params.get("method", "standard")
        scaler = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}[method]()
        df[columns] = scaler.fit_transform(df[columns])
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


def apply_feature_plan_node(state: PipelineState) -> PipelineState:
    """Applies the validated FeaturePlan. Structured ops run inline; any
    custom_code step is dry-run in the sandbox first, then run on the full
    dataset only after the dry-run succeeds (CLAUDE.md rule #6). Output
    schema is checked afterward (PRD FR-17) before advancing."""
    df = pd.read_csv(state["dataset_path"])
    plan = state.get("feature_plan", {})
    retry_count = dict(state.get("retry_count", {}))

    try:
        for step in plan.get("steps", []):
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
    state["feature_plan_valid"] = True
    return state


def dispatch_training_node(state: PipelineState) -> PipelineState:
    task_spec = state.get("task_spec", {})
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
    """Picks the best model per the task spec's metric (PRD FR-22)."""
    task_spec = state.get("task_spec", {})
    metric = task_spec.get("metric") or ("f1" if task_spec.get("task_type") == "classification" else "rmse")
    lower_is_better = metric in ("rmse", "mae")

    succeeded = [r for r in state.get("training_results", []) if r["status"] == "succeeded" and metric in r.get("metrics", {})]
    if not succeeded:
        state["best_model"] = {}
        state.setdefault("errors", []).append("evaluate: no candidate produced a usable result for the target metric")
        return state

    best = min(succeeded, key=lambda r: r["metrics"][metric]) if lower_is_better else max(
        succeeded, key=lambda r: r["metrics"][metric]
    )
    state["best_model"] = best
    return state
