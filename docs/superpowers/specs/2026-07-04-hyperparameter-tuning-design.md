# Hyperparameter tuning with live iteration progress

Date: 2026-07-04
Status: approved (user confirmed Optuna + live progress + final trend chart)

## Goal

Each candidate model gets Bayesian hyperparameter tuning (Optuna/TPE) inside
its async training job, with per-trial progress visible live in the UI and a
score-vs-trial trend chart for the winning model after the run.

## Decisions

- **Optuna (TPE sampler)** per user choice; new required dependency.
- Tuning runs **inside the existing training job thread** (no new services);
  per-candidate in-memory study, quiet logging.
- Objective: mean 3-fold CV score of the *full pipeline* (fold-safe
  preprocessor + estimator) on the **training fold only** — the holdout is
  never seen during tuning. Stratified for classification, TimeSeriesSplit
  when a time_column is set; folds auto-reduce with the same
  "explain, never silently omit" pattern as k-fold CV.
- Tuning metric derives from the task spec metric (f1 → f1_weighted,
  rmse → neg_root_mean_squared_error, etc.). History stores scores in
  natural units plus a lower_is_better flag.
- The **LLM-proposed hyperparams are evaluated first as trial 0** (baseline);
  the final model uses the best of baseline + all trials, so tuning can never
  do worse than the previous behavior.
- Budget: `training.tuning_trials` (new, default 15) and the previously dead
  `training.hyperparam_search_budget_seconds` (120) as the Optuna timeout —
  whichever hits first.
- Estimators with nothing to tune (LinearRegression) skip tuning with an
  explanatory note.
- **User control**: `tuning_enabled` toggle at the confirm checkpoint
  (state + ConfirmRequest + checkbox), default on.

## Data flow

1. `apply_feature_plan` → `dispatch_training_node` passes `tuning_enabled`
   and the task metric to `train_model`.
2. `_run_job` builds the pipeline, runs `_tune_pipeline`; an Optuna callback
   updates `_registry[run_id]["tuning"]` after every trial:
   `{enabled, trials_total, trials_done, metric, lower_is_better,
     best_params, history: [{trial, score, best_score}], note}`.
3. `poll_training_job` snapshots the registry; `poll_training_node` refreshes
   `state["training_results"]` every 2s; `GET /api/runs/{id}` therefore
   exposes live tuning progress with no new endpoint.
4. Final fit + holdout metrics + k-fold CV use the tuned params; the winning
   params flow into the exported training script.

## State schema (rule #1)

- `PipelineState.tuning_enabled: bool` (default True).
- `TrainingResult.tuning: TuningInfo` (new Pydantic models `TuningInfo`,
  `TuningTrial` in src/state.py).

## Frontend

- Confirm card: "Hyperparameter tuning" checkbox beside the CV controls.
- During training: per-candidate progress bar (trials done/total + best score
  so far) inside the existing train-progress panel, updating on the 2s poll.
- Results view: "Tuning trend" chart for the best model — per-trial score
  dots plus a best-so-far line, vanilla JS/SVG, theme-aware.

## Testing (TDD)

- Search-space sampling produces non-empty valid params for every canonical
  estimator; LinearRegression yields empty → skip note.
- Integration: tuned job succeeds, history length ≤ trials_total, best_score
  monotone (in the maximize orientation), best_params applied to final model,
  disabled flag short-circuits.
- Full fixture suite after graph/state changes (CLAUDE.md).
