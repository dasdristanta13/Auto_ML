"""LLMs frequently propose `max_features="auto"` for sklearn tree ensembles
(RandomForest*/GradientBoosting*) because it was the historical default prior
to sklearn 1.1 and was removed entirely in sklearn 1.3+. Since the LLM's
hyperparams are structured JSON, not validated against the installed sklearn
version (CLAUDE.md rule #2 covers structure, not value compatibility),
_build_estimator must sanitize this specific known-deprecated alias before
construction so a plausible, common LLM suggestion doesn't fail every run.

"auto" meant sqrt(n_features) for classifiers and n_features (i.e. no
restriction / None) for regressors in the old sklearn API — the mapping
below preserves that historical semantic rather than just guessing "sqrt"."""

from __future__ import annotations

import pytest

from src.training.dispatch import _build_estimator


@pytest.mark.parametrize(
    "estimator_name,expect_max_features",
    [
        ("RandomForestClassifier", "sqrt"),
        ("GradientBoostingClassifier", "sqrt"),
        ("RandomForestRegressor", None),
        ("GradientBoostingRegressor", None),
    ],
)
def test_max_features_auto_is_sanitized_per_estimator_kind(estimator_name, expect_max_features):
    estimator = _build_estimator("sklearn", estimator_name, {"max_features": "auto", "n_estimators": 10})

    assert estimator.max_features == expect_max_features


def test_other_max_features_values_pass_through_unchanged():
    estimator = _build_estimator("sklearn", "RandomForestClassifier", {"max_features": "log2"})

    assert estimator.max_features == "log2"


def test_estimators_without_max_features_are_unaffected():
    estimator = _build_estimator("sklearn", "LogisticRegression", {"max_iter": 500})

    assert estimator.max_iter == 500
