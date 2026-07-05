"""The canonical-candidate completeness floor (model_selection_node) must
never inject a candidate whose library isn't importable in this environment —
xgboost/lightgbm are optional dependencies (requirements.txt), and a canonical
candidate that fails on `import xgboost` burns a training slot on every run."""

from __future__ import annotations

from src.agents import model_selection_node as msn
from src.agents.model_selection_node import _fill_missing_candidates


def test_fill_missing_candidates_skips_unavailable_libraries(monkeypatch):
    monkeypatch.setattr(msn, "_library_available", lambda library: library == "sklearn")

    filled = _fill_missing_candidates([], "classification")

    assert filled, "sklearn families must still be filled in"
    assert all(c.library == "sklearn" for c in filled)


def test_fill_missing_candidates_includes_optional_libraries_when_available(monkeypatch):
    monkeypatch.setattr(msn, "_library_available", lambda library: True)

    filled = _fill_missing_candidates([], "classification")

    assert {c.library for c in filled} == {"sklearn", "xgboost", "lightgbm"}
