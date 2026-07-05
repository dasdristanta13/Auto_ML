"""Defense-in-depth for CandidateModel.hyperparams (src/state.py), which the
LLM controls as an untyped dict[str, Any] with no schema/enum validating
individual names or values. _sanitize_hyperparams closes two gaps:
unknown/hallucinated parameter names, and known deprecated values (sklearn
removed max_features="auto" for tree ensembles in 1.3+)."""

from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from src.training.dispatch import _build_estimator, _sanitize_hyperparams, known_estimators


def test_sanitize_drops_unknown_hyperparameter_key():
    result = _sanitize_hyperparams(LogisticRegression, {"max_iter": 500, "learning_rate": 0.1})
    assert result == {"max_iter": 500}


def test_sanitize_translates_max_features_auto_for_classifier():
    result = _sanitize_hyperparams(RandomForestClassifier, {"max_features": "auto", "n_estimators": 100})
    assert result == {"max_features": "sqrt", "n_estimators": 100}


def test_sanitize_translates_max_features_auto_for_regressor():
    result = _sanitize_hyperparams(RandomForestRegressor, {"max_features": "auto"})
    assert result == {"max_features": None}


def test_sanitize_passes_through_valid_values_unchanged():
    result = _sanitize_hyperparams(RandomForestClassifier, {"max_features": "sqrt", "n_estimators": 200})
    assert result == {"max_features": "sqrt", "n_estimators": 200}


def test_sanitize_skips_key_filtering_for_kwargs_accepting_init():
    class _AcceptsAnything:
        def __init__(self, **kwargs):
            pass

    result = _sanitize_hyperparams(_AcceptsAnything, {"anything_at_all": 1})
    assert result == {"anything_at_all": 1}


def test_known_estimators_returns_sklearn_names():
    names = known_estimators("sklearn")
    assert "LogisticRegression" in names
    assert "RandomForestClassifier" in names


def test_known_estimators_returns_empty_set_for_unknown_library():
    assert known_estimators("not-a-real-library") == set()


def test_build_estimator_still_raises_on_unknown_estimator_name():
    with pytest.raises(ValueError, match="unknown estimator"):
        _build_estimator("sklearn", "NotARealEstimator", {})


def test_build_estimator_sanitizes_before_construction():
    estimator = _build_estimator("sklearn", "RandomForestClassifier", {"max_features": "auto", "n_estimators": 50})
    assert estimator.max_features == "sqrt"
    assert estimator.n_estimators == 50
