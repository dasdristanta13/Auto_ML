"""Deterministic auto-insight generation.

Every insight here is computed directly from data already sitting in
PipelineState (profile, leakage_flags, task_spec, training_results,
best_model) — no extra LLM call. This keeps insights instant, free, and
fully explainable (CLAUDE.md: no unexplained automated decisions), and
lets them appear progressively as each pipeline stage completes rather than
waiting on a single "insights" step.

Each insight is {id, category, tone, message}. tone is one of
"info" | "success" | "warning" | "danger" — the frontend maps these to the
same status colors used everywhere else (never color-alone; always paired
with an icon + label there).
"""

from __future__ import annotations

from typing import Any

from src.profiling.heuristics import IMBALANCE_THRESHOLD, looks_like_identifier, minority_ratio

_HIGH_NULL_RATE = 0.30
_MIN_ROWS_FOR_CARDINALITY_CHECK = 20

_MAX_SUGGESTED_QUESTIONS = 4
_FALLBACK_SUGGESTED_QUESTIONS = [
    "Is there a risk of target leakage in this model?",
    "What are the caveats I should know about?",
    "Why was this model chosen over the alternatives?",
]


def _insight(insight_id: str, category: str, tone: str, message: str) -> dict[str, Any]:
    return {"id": insight_id, "category": category, "tone": tone, "message": message}


def profile_insights(profile: dict[str, Any], task_spec: dict[str, Any]) -> list[dict[str, Any]]:
    insights: list[dict[str, Any]] = []
    row_count = profile.get("row_count", 0)
    columns = profile.get("columns", {})

    pii_count = profile.get("pii_report", {}).get("pii_columns_detected", 0)
    if pii_count:
        insights.append(
            _insight(
                "pii_redacted", "privacy", "info",
                f"{pii_count} column(s) were flagged as PII and redacted before reaching any AI-facing step.",
            )
        )

    if profile.get("is_wide_dataset"):
        insights.append(
            _insight(
                "wide_dataset", "scale", "info",
                f"This dataset is wide ({profile.get('column_count', 0)} columns) — profiled via correlated "
                "column clustering rather than exhaustive per-column stats to keep it out of the LLM's context.",
            )
        )

    worst_null_col, worst_null_rate = None, 0.0
    identifier_col = None
    for name, info in columns.items():
        null_rate = info.get("null_rate", 0.0) or 0.0
        if null_rate > worst_null_rate:
            worst_null_col, worst_null_rate = name, null_rate

        n_unique = info.get("n_unique", 0)
        if (
            not info.get("is_pii")
            and name != task_spec.get("target_column")
            and row_count > _MIN_ROWS_FOR_CARDINALITY_CHECK
            and looks_like_identifier(name, info.get("dtype", ""), n_unique, row_count)
        ):
            identifier_col = name

    if worst_null_col and worst_null_rate > _HIGH_NULL_RATE:
        insights.append(
            _insight(
                "high_null_rate", "data_quality", "warning",
                f"'{worst_null_col}' is missing in {worst_null_rate:.0%} of rows — how it's imputed will "
                "meaningfully affect the model.",
            )
        )

    if identifier_col:
        insights.append(
            _insight(
                "likely_identifier", "data_quality", "warning",
                f"'{identifier_col}' has a near-unique value in almost every row — it looks like an identifier "
                "column. Consider excluding it from training; identifiers can look predictive during training "
                "while carrying no real signal at prediction time.",
            )
        )

    strong_pairs = profile.get("strong_correlations", [])
    if strong_pairs:
        top = max(strong_pairs, key=lambda p: abs(p["correlation"]))
        insights.append(
            _insight(
                "strong_correlation", "correlation", "info",
                f"'{top['a']}' and '{top['b']}' are strongly correlated (r={top['correlation']:.2f}) — "
                "consider whether both add independent signal.",
            )
        )

    if task_spec.get("task_type") == "classification":
        target = task_spec.get("target_column")
        target_info = columns.get(target) if target else None
        positive_rate = minority_ratio(target_info)

        if positive_rate is not None and positive_rate < IMBALANCE_THRESHOLD:
            insights.append(
                _insight(
                    "class_imbalance", "imbalance", "warning",
                    f"Your target is imbalanced (minority class ~{positive_rate:.0%}) — accuracy alone will "
                    "look deceptively good. F1, precision/recall, or a cost-weighted metric usually matter more here.",
                )
            )

    return insights


def _leakage_insights(leakage_flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not leakage_flags:
        return []
    worst = max(leakage_flags, key=lambda f: {"high": 2, "medium": 1}.get(f.get("severity"), 0))
    tone = "danger" if worst.get("severity") == "high" else "warning"
    return [
        _insight(
            "leakage_flag", "leakage", tone,
            f"Possible leakage in '{worst['column']}': {worst['reason']} Verify this column would actually "
            "be known at prediction time before trusting these results.",
        )
    ]


def _model_insights(task_spec: dict[str, Any], training_results: list[dict[str, Any]], best_model: dict[str, Any]) -> list[dict[str, Any]]:
    insights: list[dict[str, Any]] = []
    metric = task_spec.get("metric")
    lower_is_better = metric in ("rmse", "mae")
    succeeded = [r for r in training_results if r.get("status") == "succeeded" and metric in (r.get("metrics") or {})]

    if metric and len(succeeded) >= 2:
        ranked = sorted(succeeded, key=lambda r: r["metrics"][metric], reverse=not lower_is_better)
        best, runner_up = ranked[0], ranked[1]
        delta = abs(best["metrics"][metric] - runner_up["metrics"][metric])
        relative = delta / abs(runner_up["metrics"][metric]) if runner_up["metrics"][metric] else 0
        if relative < 0.02:
            insights.append(
                _insight(
                    "close_race", "model_performance", "info",
                    f"'{best['candidate_name']}' and '{runner_up['candidate_name']}' perform almost identically "
                    f"on {metric} — the simpler or faster model may be the more practical choice.",
                )
            )
        else:
            insights.append(
                _insight(
                    "clear_winner", "model_performance", "success",
                    f"'{best['candidate_name']}' leads by {delta:.3f} {metric} over the next-best candidate "
                    f"('{runner_up['candidate_name']}').",
                )
            )

    if metric and best_model.get("cv_folds") and metric in (best_model.get("cv_metrics") or {}):
        cv = best_model["cv_metrics"][metric]
        relative_std = cv["std"] / abs(cv["mean"]) if cv["mean"] else 0
        folds = best_model["cv_folds"]
        if relative_std > 0.15:
            insights.append(
                _insight(
                    "cv_unstable", "model_performance", "warning",
                    f"Cross-validation scores for the best model varied by ±{cv['std']:.3f} across {folds} "
                    f"folds — treat the headline {metric} as an estimate, not a guarantee.",
                )
            )
        else:
            insights.append(
                _insight(
                    "cv_stable", "model_performance", "success",
                    f"The best model's {metric} was stable across {folds}-fold cross-validation (±{cv['std']:.3f}).",
                )
            )

    importance = best_model.get("feature_importance") or []
    if importance:
        top = importance[0]
        if top["importance"] > 0.5:
            insights.append(
                _insight(
                    "importance_concentrated", "feature_importance", "warning",
                    f"'{top['feature']}' alone accounts for {top['importance']:.0%} of the model's decisions — "
                    "if it's an identifier or otherwise not truly predictive, this often signals leakage rather "
                    "than genuine signal.",
                )
            )
        else:
            insights.append(
                _insight(
                    "top_driver", "feature_importance", "info",
                    f"'{top['feature']}' is the strongest driver of predictions ({top['importance']:.0%}).",
                )
            )

    return insights


def generate_insights(state: dict[str, Any], stages_done: list[str]) -> list[dict[str, Any]]:
    profile = state.get("profile") or {}
    if not profile:
        return []

    task_spec = state.get("task_spec") or {}
    insights = profile_insights(profile, task_spec)
    insights += _leakage_insights(state.get("leakage_flags") or [])
    insights += _model_insights(task_spec, state.get("training_results") or [], state.get("best_model") or {})

    flagged_something = any(i["category"] in ("data_quality", "imbalance", "leakage") for i in insights)
    if "leakage_check" in stages_done and not flagged_something:
        insights.append(
            _insight(
                "no_major_concerns", "data_quality", "success",
                "No major data-quality or leakage concerns were detected during profiling.",
            )
        )
    return insights


def suggested_questions(
    insights: list[dict[str, Any]], task_spec: dict[str, Any], best_model: dict[str, Any]
) -> list[str]:
    """Deterministic (no LLM call) prompts for the chat panel's suggestion
    chips, derived from what's actually notable in THIS run so they're
    relevant rather than generic. Capped at 4, deduplicated; falls back to a
    fixed generic set when nothing stands out."""
    questions: list[str] = []
    categories = {i.get("category") for i in insights}

    if "imbalance" in categories:
        questions.append("Why is my data imbalanced, and what was done about it?")
    if "leakage" in categories:
        questions.append("Is there a risk of target leakage in this model?")

    importance = best_model.get("feature_importance") or []
    if importance:
        questions.append(f"Why is '{importance[0]['feature']}' the strongest driver?")

    metric = task_spec.get("metric")
    if best_model.get("candidate_name") and metric:
        questions.append(f"How can I improve {metric}?")

    if not questions:
        questions = list(_FALLBACK_SUGGESTED_QUESTIONS)

    seen: set[str] = set()
    deduped: list[str] = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped[:_MAX_SUGGESTED_QUESTIONS]
