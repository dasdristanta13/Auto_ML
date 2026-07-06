"""The canonical-candidate completeness floor (model_selection_node) must
never inject a candidate whose library isn't importable in this environment —
xgboost/lightgbm are optional dependencies (requirements.txt), and a canonical
candidate that fails on `import xgboost` burns a training slot on every run."""

from __future__ import annotations

from src.agents import model_selection_node as msn
from src.agents.model_selection_node import _fill_missing_candidates
from src.agents.prompt_utils import render_prompt


def test_fill_missing_candidates_skips_unavailable_libraries(monkeypatch):
    monkeypatch.setattr(msn, "_library_available", lambda library: library == "sklearn")

    filled = _fill_missing_candidates([], "classification")

    assert filled, "sklearn families must still be filled in"
    assert all(c.library == "sklearn" for c in filled)


def test_fill_missing_candidates_includes_optional_libraries_when_available(monkeypatch):
    monkeypatch.setattr(msn, "_library_available", lambda library: True)

    filled = _fill_missing_candidates([], "classification")

    assert {c.library for c in filled} == {"sklearn", "xgboost", "lightgbm"}


def test_model_selection_prompt_renders_prior_attempt_feedback_when_present():
    rendered = render_prompt(
        "model_selection.md",
        TASK_SPEC_JSON={"task_type": "classification"},
        PROFILE_JSON={},
        PRIOR_ATTEMPT_FEEDBACK="## Your previous attempt was rejected\nunknown estimator 'Foo'",
    )
    assert "Your previous attempt was rejected" in rendered
    assert "unknown estimator 'Foo'" in rendered


def test_model_selection_prompt_omits_feedback_section_when_absent():
    rendered = render_prompt(
        "model_selection.md",
        TASK_SPEC_JSON={"task_type": "classification"},
        PROFILE_JSON={},
        PRIOR_ATTEMPT_FEEDBACK="",
    )
    assert "Your previous attempt was rejected" not in rendered


def test_model_selection_prompt_warns_about_max_features_auto():
    rendered = render_prompt(
        "model_selection.md", TASK_SPEC_JSON={}, PROFILE_JSON={}, PRIOR_ATTEMPT_FEEDBACK=""
    )
    assert "max_features" in rendered
    assert "sqrt" in rendered
