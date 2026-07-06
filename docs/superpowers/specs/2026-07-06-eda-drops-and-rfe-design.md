# EDA-Driven Drop Suggestions + Recursive Feature Elimination — Design

Date: 2026-07-06
Status: approved

## Problem

1. `leakage_check_node` detects features that are near-perfect proxies for the
   target (numeric correlation >= 0.95, categorical purity >= 0.98), but the
   flags are display-only — nothing suggests actually dropping those columns,
   so users ship leaky models unless they act manually.
2. There is no feature-selection mechanism: every surviving column reaches
   every candidate model. On wide/noisy data, recursive feature elimination
   (RFE) with the candidate's own importances typically yields a better,
   smaller model.

## Decisions (from brainstorming)

- High-severity leakage flags become suggested `drop` steps at the existing
  feature-approval checkpoint (approve/reject each — never silently applied,
  per CLAUDE.md). Medium-severity name-hint flags stay display-only.
- RFE is a per-run toggle at the confirm checkpoint (like CV/tuning),
  default **off** (it multiplies training cost). EDA adds an insight
  recommending it when the dataset is feature-heavy.
- RFECV (not fixed-n RFE): picks the feature count that maximizes CV score on
  the task's metric, using each candidate's own coef_/feature_importances_.

## Architecture

### A. Leakage flags → drop suggestions (`src/profiling/eda.py`)

- `run_eda(df, profile, task_spec, leakage_flags=None)` gains the flags
  parameter; `eda_node` passes `state["leakage_flags"]`.
- For each **high**-severity flag whose column isn't the target/time column:
  emit `_step("drop", [col], {}, rationale)` where the rationale names the
  leakage reason and warns the model won't generalize if trained on it.
- Columns already drop-suggested this way are skipped by the per-column
  suggestion loop (no duplicate impute/encode/drop suggestions).

### B. RFE toggle end to end

- `PipelineState`: `feature_selection_enabled: bool` (default False in
  `new_state`), mirroring `cv_enabled`/`tuning_enabled`.
- `ConfirmRequest` (src/api/server.py): `feature_selection_enabled: bool =
  False`; the confirm endpoint copies it into state.
- `dispatch_training_node` passes it to `train_model`; `train_model` and
  `_run_job` gain `feature_selection_enabled: bool = False`.
- `_make_pipeline` inserts `RFECV` between preprocessing and the model when
  enabled: `RFECV(estimator=<fresh instance of the candidate>, step=0.2,
  cv=_tuning_splitter(...), scoring=<task metric via _tuning_scoring>,
  min_features_to_select=1)`. Present in every fit path (holdout, k-fold CV,
  tuning trials) so scores stay comparable and leakage-safe.
- Auto-skip with a note when the feature matrix is too narrow (< 5 columns
  entering the pipeline) — same "adjust and explain" pattern as CV/SMOTE.
- Job registry gains `feature_selection`: `{enabled, n_features_selected,
  selected_features, note}`; `poll_training_job` passes it through, and
  feature importance is computed over the RFE-selected feature names.

### C. EDA insight for wide data (`src/profiling/eda.py`)

When the dataset has >= 15 non-target feature columns, append an insight
recommending enabling feature selection (RFE) at the confirm step.

### D. Frontend

- Confirm form: a "Feature selection (RFE)" checkbox beside the CV/tuning
  toggles, submitted as `feature_selection_enabled`.
- Results: when a trained model carries `feature_selection.enabled`, show
  "RFE kept N of M features" (tooltip lists the selected features).

## Error handling

- RFECV estimator lacking coef_/feature_importances_: not reachable — every
  registry estimator exposes one; guarded anyway by the tuning-style
  try/except in `_run_job` (falls back with a note rather than failing).
- Too few features → skip + note (never a crash).

## Testing

- `tests/test_eda_leakage_drops.py`: high-severity flag becomes a drop step
  with rationale; medium name-hint flag does not; target column never
  drop-suggested; no duplicate steps for the same column.
- `tests/test_feature_selection.py`: `_run_job` with
  `feature_selection_enabled=True` on a synthetic dataset with informative +
  noise features succeeds, reports `n_features_selected <= total`, and
  selected_features is a subset of the preprocessor's output names; narrow
  dataset (< 5 features) skips with a note; disabled runs report
  `enabled: False`.

## Out of scope

- Automatic (non-toggled) RFE, permutation importance, SHAP-based selection.
- Auto-applying drops without the approval checkpoint.
