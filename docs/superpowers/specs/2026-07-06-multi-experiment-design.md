# Multi-Experiment on the Same Dataset — Design

Date: 2026-07-06
Status: approved

## Problem

Each run bundles one CSV upload with one pipeline execution (`data/uploads/{run_id}.csv`).
Running a second experiment on the same data — a different use-case description,
target, or training configuration — requires re-uploading the file. Uploaded
datasets should be reusable: from any existing run the user can launch a new
experiment on the same file, and each experiment appears as its own run.

## Decisions (from brainstorming)

- **Feature shape**: re-run without re-upload. A new experiment is a full,
  first-class run that references the source run's dataset file.
- **Reuse depth**: the new experiment supplies a new use-case description; the
  already-computed dataset profile is reused (re-profiling is skipped), the
  task spec is re-inferred by the LLM from the new description, and the run
  pauses at the standard confirm checkpoint like any fresh upload.
- **UI entry**: a "New experiment" button on the run detail view, opening a
  small form with a single textarea for the new description.

## Architecture

### API (`src/api/server.py`)

`POST /api/runs/{run_id}/experiments`, body `{"description": str}`, guarded by
`require_session` like every other runs endpoint.

Behavior:
1. 404 if `run_id` is unknown.
2. 409 with "dataset no longer available, please re-upload" if the source
   run's `dataset_path` no longer exists on disk.
3. Creates a new run whose state is `new_state(new_run_id, same dataset_path,
   new description)`. The dataset file is **not** copied.
4. Copies the source state's `profile` into the new state when present (if the
   source is still mid-intake and has no profile yet, the new run simply
   profiles the file itself — graceful fallback, not an error).
5. Stores the run entry exactly like `create_run` does, plus:
   - `filename`: inherited from the source run (display name in the list).
   - `source_run_id`: lineage marker.
6. Starts the existing `_run_intake` background thread. Response mirrors
   `create_run`: `{"run_id", "status": "profiling"}`.

`GET /api/runs` (list) and `GET /api/runs/{id}` (detail) additionally return
`source_run_id` (None for ordinary uploads).

### Pipeline (`src/graph/nodes.py`)

`profile_node` returns early when `state.get("profile")` is already populated —
a deterministic shortcut, no graph rewiring. Everything downstream (understand
usecase, confirm checkpoint, leakage, features, training, report) is unchanged.

### Frontend (`frontend/`)

- Run detail view: a "New experiment" button; clicking reveals an inline form
  (textarea for the new use-case description + submit). On success, navigate to
  the newly created run.
- Runs list: rows with `source_run_id` show a small "re-run of <id>" hint.

## Error handling

- Unknown source run → 404.
- Dataset file deleted from disk → 409, actionable message.
- Empty/whitespace description → 400 (same expectation as the upload form).
- Source run without a profile yet → new run re-profiles (fallback, silent).

## Testing (`tests/test_api_experiments.py`)

Against the mock-LLM test app (existing conftest conventions):
- Creating an experiment from a finished run returns a distinct `run_id`,
  shares the same `dataset_path`, sets `source_run_id`, and skips re-profiling
  (profile object identical to the source's).
- `profile_node` unit check: pre-seeded profile short-circuits the CSV read.
- 404 for unknown source run; 409 when the dataset file was removed; 400 for a
  blank description.
- Runs list includes `source_run_id`.

## Out of scope (YAGNI, discussed and deferred)

- First-class dataset registry (dataset_id, dedup, storage-layout changes).
- Auto-fan-out of N variants per upload.
- Cross-experiment comparison view.
