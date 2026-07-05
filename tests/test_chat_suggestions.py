"""suggested_questions() derives chat-panel suggestion chips deterministically
(no LLM call) from a run's own insights/results — see
docs/superpowers/specs/2026-07-05-ai-assistant-chat-design.md."""

from __future__ import annotations

from src.insights.auto_insights import suggested_questions


def test_imbalance_insight_yields_imbalance_question():
    insights = [{"id": "class_imbalance", "category": "imbalance", "tone": "warning", "message": "..."}]
    qs = suggested_questions(insights, {"metric": "f1"}, {"candidate_name": "rf"})
    assert "Why is my data imbalanced, and what was done about it?" in qs


def test_leakage_insight_yields_leakage_question():
    insights = [{"id": "leakage_flag", "category": "leakage", "tone": "danger", "message": "..."}]
    qs = suggested_questions(insights, {}, {})
    assert "Is there a risk of target leakage in this model?" in qs


def test_top_feature_importance_yields_feature_question():
    best_model = {"feature_importance": [{"feature": "tenure_months", "importance": 0.4}]}
    qs = suggested_questions([], {}, best_model)
    assert "Why is 'tenure_months' the strongest driver?" in qs


def test_metric_present_yields_improve_metric_question():
    best_model = {"candidate_name": "rf"}
    qs = suggested_questions([], {"metric": "f1"}, best_model)
    assert "How can I improve f1?" in qs


def test_no_notable_signal_falls_back_to_generic_set():
    qs = suggested_questions([], {}, {})
    assert qs
    assert len(qs) <= 4


def test_never_exceeds_four_and_has_no_duplicates():
    insights = [{"category": "imbalance"}, {"category": "leakage"}]
    best_model = {
        "feature_importance": [{"feature": "x", "importance": 0.9}],
        "candidate_name": "rf",
    }
    qs = suggested_questions(insights, {"metric": "f1"}, best_model)
    assert len(qs) <= 4
    assert len(qs) == len(set(qs))
