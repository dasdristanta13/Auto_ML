# Training config generation hardening

Date: 2026-07-06
Status: approved

## Context

The model-selection stage of the pipeline (`src/agents/model_selection_node.py`)
asks an LLM to emit a structured JSON shortlist of candidate models
(`CandidateModel`: `library`, `estimator`, `hyperparams`, `rationale`), which
`src/training/dispatch.py` turns into real scikit-learn/XGBoost/LightGBM
objects and trains asynchronously. Two prior incidents on a sibling branch
(`feature/ai-assistant-chat`) — a deprecated `max_features="auto"` value
crashing Random Forest construction, and a high-cardinality target crashing
XGBoost's label validation — showed that nothing currently stands between an
LLM-authored hyperparameter dict and the raw constructor call. Investigation
confirmed:

- `CandidateModel.hyperparams` is an untyped `dict[str, Any]`; `estimator` is
  a free string with no enum. Neither is validated beyond JSON shape.
- `model_selection_node` has no retry loop on schema-validation failure — a
  `ValidationError` is swallowed and candidates silently fall back to the
  deterministic completeness floor only, discarding the LLM's contribution.
  This is inconsistent with `feature_engineering_node`, which already retries
  with feedback on invalid output.
- The prompt (`model_selection.md`) gives no guidance on deprecated/invalid
  parameter values, no few-shot example, and no explicit constraint that
  hyperparameter names must be real constructor arguments.
- This branch (`feature/improvement_v1`) does not include the two prior
  fixes at all (confirmed: `_sanitize_hyperparams` does not exist here) —
  they only exist on the `feature/ai-assistant-chat` lineage.

## Goal

Make LLM-generated training configuration reliably produce a constructible,
trainable estimator, via three coordinated layers rather than prompt wording
alone:

1. A richer prompt with concrete constraints, deprecated-value guidance, and
   a retry-feedback slot.
2. A validation + bounded retry loop in `model_selection_node`, mirroring
   `feature_engineering_node`'s existing pattern, so schema/vocabulary errors
   give the LLM a genuine second attempt instead of being silently discarded.
3. A deterministic hyperparameter sanitizer in `dispatch.py`, using
   signature introspection (generalizes past one-off value fixes like
   `max_features`), as the last line of defense before construction.

## Out of scope

- The high-cardinality classification-target guardrail (a target-validation
  concern at the confirm checkpoint, unrelated to training-config
  generation; already solved on the sibling branch, not reintroduced here).
- A free-form code-gen fallback for training config — `CandidateModel` stays
  structured-JSON-only; no `custom_code`-style escape hatch is added.
- Widening the canonical model universe or adding new estimator families.
- Per-parameter numeric range validation (e.g. rejecting `n_estimators=-5`)
  — signature introspection catches invalid *names*, not invalid *values*,
  except for the specific deprecated-value translation table below.

## Design

### 1. Prompt (`src/agents/prompts/model_selection.md`)

Add, without removing any existing instruction:

- An explicit constraint: hyperparameter names must be real constructor
  arguments of the named estimator class — never invent a parameter name.
- A concrete deprecated-value example: sklearn removed `max_features="auto"`
  for tree ensembles (`RandomForest*`, `GradientBoosting*`) in 1.3+; use
  `"sqrt"`, `"log2"`, or `None` instead.
- A `{{PRIOR_ATTEMPT_FEEDBACK}}` token, rendered as a `## Your previous
  attempt was rejected` section (same phrasing/shape as
  `feature_engineering.md`) when `state["candidate_models_feedback"]` is set,
  empty string otherwise.
- One short worked example: a classification candidate for a mid-size,
  moderately imbalanced dataset, showing a well-formed `hyperparams` dict
  and a data-grounded `rationale`.

### 2. State (`src/state.py`)

Add to `PipelineState`, mirroring `feature_plan_valid`/`feature_plan_feedback`:

```python
candidate_models_valid: bool
candidate_models_feedback: str  # fed back into the prompt on retry
```

`new_state()` initializes `candidate_models_valid=False` (candidates aren't
trustworthy until the node validates them at least once, same convention as
`feature_plan_valid`).

### 3. Estimator registry + sanitizer (`src/training/dispatch.py`)

Extract the per-library name→class mapping already inside `_build_estimator`
into a named function, and add a lightweight lookup usable without a full
training dispatch:

```python
def _estimator_registry(library: str) -> dict[str, type]:
    """Single source of truth for valid estimator names per library — used
    by both training dispatch and model_selection_node's pre-dispatch
    validation. Lazily imports each library so an uninstalled optional
    dependency (xgboost/lightgbm) only fails when actually requested."""
    if library == "sklearn":
        import sklearn.ensemble as ens
        import sklearn.linear_model as lm
        return {
            "LogisticRegression": lm.LogisticRegression,
            "LinearRegression": lm.LinearRegression,
            "Ridge": lm.Ridge,
            "RandomForestClassifier": ens.RandomForestClassifier,
            "RandomForestRegressor": ens.RandomForestRegressor,
            "GradientBoostingClassifier": ens.GradientBoostingClassifier,
            "GradientBoostingRegressor": ens.GradientBoostingRegressor,
        }
    if library == "xgboost":
        import xgboost as xgb
        return {"XGBClassifier": xgb.XGBClassifier, "XGBRegressor": xgb.XGBRegressor}
    if library == "lightgbm":
        import lightgbm as lgb
        return {"LGBMClassifier": lgb.LGBMClassifier, "LGBMRegressor": lgb.LGBMRegressor}
    raise ValueError(f"unknown library '{library}'")


def known_estimators(library: str) -> set[str]:
    """Best-effort: returns an empty set if the library isn't installed in
    this environment, rather than raising. model_selection_node uses this to
    validate LLM-proposed estimator names before a candidate ever reaches
    dispatch — only called for libraries _library_available() already
    confirmed importable, so ImportError here is a defensive fallback, not
    the expected path."""
    try:
        return set(_estimator_registry(library))
    except ImportError:
        return set()
```

`_build_estimator` becomes:

```python
def _build_estimator(library: str, estimator: str, hyperparams: dict[str, Any]):
    registry = _estimator_registry(library)
    if estimator not in registry:
        raise ValueError(f"unknown estimator '{estimator}' for library '{library}'")
    estimator_cls = registry[estimator]
    hyperparams = _sanitize_hyperparams(estimator_cls, hyperparams)
    return estimator_cls(**hyperparams)
```

New sanitizer, called from the single choke point above so it covers the
baseline fit, every Optuna trial (`_tune_pipeline`'s `make_pipeline`), and
the final fit:

```python
import inspect

_DEPRECATED_HYPERPARAM_VALUES: dict[str, dict[Any, dict[str, Any]]] = {
    # param_name -> {llm_supplied_value: {"classifier": replacement, "regressor": replacement}}
    "max_features": {"auto": {"classifier": "sqrt", "regressor": None}},
}


def _sanitize_hyperparams(estimator_cls: type, hyperparams: dict[str, Any]) -> dict[str, Any]:
    """Defense-in-depth for CandidateModel.hyperparams (src/state.py), an
    untyped dict[str, Any] the LLM controls with no schema/enum validating
    individual names or values. Two independent protections:
    1. Drop any key estimator_cls.__init__ doesn't accept, via signature
       introspection, instead of a typo'd/hallucinated name crashing
       construction with TypeError. Skipped when __init__ declares
       **kwargs (some XGBoost/LightGBM versions accept arbitrary extra
       keys), since nothing can be validated against an open signature.
    2. Translate known deprecated/renamed values (e.g. sklearn's
       max_features="auto", removed in 1.3) to their modern equivalent.
    """
    sanitized = dict(hyperparams)
    params = inspect.signature(estimator_cls.__init__).parameters
    accepts_arbitrary_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if not accepts_arbitrary_kwargs:
        valid_names = set(params) - {"self"}
        sanitized = {k: v for k, v in sanitized.items() if k in valid_names}

    kind = "classifier" if estimator_cls.__name__.endswith("Classifier") else "regressor"
    for param, value_map in _DEPRECATED_HYPERPARAM_VALUES.items():
        if param in sanitized and sanitized[param] in value_map:
            sanitized[param] = value_map[sanitized[param]][kind]

    return sanitized
```

### 4. Node validation + retry (`src/agents/model_selection_node.py`)

```python
def _validate_candidates(raw: dict) -> tuple[list[CandidateModel] | None, list[str]]:
    errors: list[str] = []
    try:
        parsed = _CandidateModelList(**raw)
    except ValidationError as exc:
        return None, [str(exc)]

    for candidate in parsed.candidates:
        if not _library_available(candidate.library):
            continue  # environment gap, not an LLM error — filtered later, never retried
        known = dispatch.known_estimators(candidate.library)
        if candidate.estimator not in known:
            errors.append(
                f"'{candidate.estimator}' is not a recognized {candidate.library} estimator "
                f"(valid options: {sorted(known)})"
            )

    return (parsed.candidates if not errors else None), errors
```

`model_selection_node` gains the same retry shape as `feature_engineering_node`:

```python
def model_selection_node(state: PipelineState) -> PipelineState:
    client = get_llm_client()
    system_prompt = render_prompt(
        "model_selection.md",
        TASK_SPEC_JSON=state.get("task_spec", {}),
        PROFILE_JSON=state.get("profile", {}),
        PRIOR_ATTEMPT_FEEDBACK=(
            f"## Your previous attempt was rejected\n{state['candidate_models_feedback']}"
            if state.get("candidate_models_feedback")
            else ""
        ),
    )
    raw = client.generate(
        run_id=state["run_id"], node="model_selection", system_prompt=system_prompt,
        user_prompt="Return the candidate model list JSON now.",
        json_schema=_CandidateModelList.model_json_schema(),
    )

    candidates, errors = _validate_candidates(raw)
    retry_count = dict(state.get("retry_count", {}))

    if candidates is None:
        retry_count["model_selection"] = retry_count.get("model_selection", 0) + 1
        state["retry_count"] = retry_count
        state["candidate_models_valid"] = False
        state["candidate_models_feedback"] = "; ".join(errors)
        state.setdefault("errors", []).append(f"model_selection attempt rejected: {'; '.join(errors)}")
        return state

    candidates = [c for c in candidates if _library_available(c.library)]
    task_type = state.get("task_spec", {}).get("task_type")
    candidates = _fill_missing_candidates(candidates, task_type)

    state["candidate_models"] = [c.model_dump() for c in candidates]
    state["candidate_models_valid"] = True
    state["candidate_models_feedback"] = ""
    return state
```

(`import` of `src.training.dispatch as dispatch` added at module top; no
circular import — `dispatch.py` does not import from `src/agents/`.)

### 5. Routing (`src/graph/routing.py`)

```python
def route_after_model_selection(state: PipelineState) -> str:
    if state.get("candidate_models_valid"):
        return "dispatch_training"
    if state.get("retry_count", {}).get("model_selection", 0) < _max_retries():
        return "model_selection"
    state.setdefault("errors", []).append(
        "model_selection: retry cap reached, falling back to report with no candidates trained"
    )
    state["status"] = "failed"
    return "report"
```

### 6. Graph wiring (`src/graph/build_graph.py`)

In both `build_graph()` and `build_train_graph()`, replace:

```python
graph.add_edge("model_selection", "dispatch_training")
```

with:

```python
graph.add_conditional_edges(
    "model_selection",
    routing.route_after_model_selection,
    {"dispatch_training": "dispatch_training", "model_selection": "model_selection", "report": "report"},
)
```

## Data flow summary

```
model_selection_node: LLM JSON -> _validate_candidates
  valid    -> floor-fill -> candidate_models_valid=True  -> dispatch_training
  invalid, retries left  -> candidate_models_feedback set -> loop back to model_selection
  invalid, retries exhausted -> status=failed -> report

dispatch_training -> train_model -> _run_job -> _build_estimator
  -> _estimator_registry(library)[estimator] (raises only if candidate somehow
     still names an unknown estimator, e.g. a not-installed-library candidate
     that slipped through)
  -> _sanitize_hyperparams(estimator_cls, hyperparams) -> estimator_cls(**hyperparams)
```

## Testing

- `tests/test_build_estimator.py` (new): `_sanitize_hyperparams` drops an
  unknown key for a given estimator; `max_features="auto"` translates to
  `"sqrt"` for `RandomForestClassifier` and `None` for
  `RandomForestRegressor`; a non-deprecated value passes through unchanged;
  an estimator whose `__init__` accepts `**kwargs` is not key-filtered.
  `known_estimators("sklearn")` returns the expected non-empty name set.
- `tests/test_model_selection.py` (extended): `_validate_candidates` rejects
  an unrecognized estimator name for an installed library; does not reject
  (silently passes through untouched) a candidate for a library that is not
  installed; `model_selection_node` retries with feedback on a rejected
  first attempt and succeeds on a corrected second attempt (LLM client
  mocked to return bad-then-good JSON); retry cap exhaustion sets
  `status="failed"` and leaves `candidate_models_valid=False`.
- `tests/test_routing.py` (new): `route_after_model_selection` returns
  `dispatch_training` when valid, `model_selection` when invalid and under
  cap, `report` (with `status="failed"`) when the cap is reached.
- Existing `tests/test_model_selection.py` completeness-floor tests and
  `tests/test_tuning.py` continue to pass unmodified — `_build_estimator`'s
  external behavior (same registry, same `ValueError` on unknown estimator)
  is preserved; only construction now runs hyperparams through sanitization.

## Non-negotiables carried over

- No raw data enters any prompt — this change only touches structured
  `task_spec`/`profile` JSON already rendered into `model_selection.md`.
- Structured output over free-form code (CLAUDE.md rule #2) — `CandidateModel`
  stays JSON-only; no code-gen fallback is introduced for training config.
- Retry caps are mandatory (CLAUDE.md rule #3) — the new loop is bounded by
  the existing `config/runtime.yaml` `retry.max_retries`, with a graceful
  fallback to `report`, exactly like `feature_engineering_node`'s loop.
- Training remains async (CLAUDE.md rule #4) — no change to `train_model`'s
  dispatch contract; sanitization happens inside the existing construction
  path, not a new inline step.
