# Agentic AutoML Platform

Upload a dataset, describe your goal in plain language, and a **LangGraph-orchestrated
agent pipeline** profiles the data, confirms the task with you, engineers features,
trains and tunes candidate models asynchronously, and returns the best model with a
plain-language report you can question in a built-in chat panel.

**Non-negotiable rule: raw data never enters an LLM context window.** Every LLM-backed
node sees only redacted statistical profiles, schemas, small capped samples, or tool
results — never a raw DataFrame or CSV content. All data manipulation happens in
deterministic Python or a sandboxed execution layer. See [CLAUDE.md](CLAUDE.md),
[PRD.md](PRD.md), and [DESIGN.md](DESIGN.md) for the full product/architecture spec this
codebase implements.

## Table of contents

- [Architecture at a glance](#architecture-at-a-glance)
- [The LangGraph pipeline](#the-langgraph-pipeline)
  - [PipelineState — the contract between nodes](#pipelinestate--the-contract-between-nodes)
  - [Full pipeline graph (CLI)](#full-pipeline-graph-cli)
  - [How the web API splits the same graph in three](#how-the-web-api-splits-the-same-graph-in-three)
  - [Run status state machine (API)](#run-status-state-machine-api)
  - [Node-by-node reference](#node-by-node-reference)
- [Cross-cutting subsystems](#cross-cutting-subsystems)
- [The AI Assistant chat panel](#the-ai-assistant-chat-panel)
- [Frontend](#frontend)
- [API reference](#api-reference)
- [Repository structure](#repository-structure)
- [Configuration](#configuration)
- [Quickstart (local)](#quickstart-local)
- [Testing](#testing)
- [Local stand-ins vs. production](#local-stand-ins-vs-production)
- [Known limitations / open questions](#known-limitations--open-questions)

## Architecture at a glance

```mermaid
flowchart LR
    subgraph Client
        FE["frontend/ (vanilla JS, no build step)\nhome view · run dashboard · AI Assistant panel"]
    end

    subgraph API["FastAPI — src/api/server.py"]
        RUNS["in-memory run registry\n(_runs dict + lock)"]
    end

    subgraph Pipeline["LangGraph pipelines — src/graph/"]
        INTAKE["build_intake_graph()"]
        PREP["build_prep_graph()"]
        TRAIN["build_train_graph()"]
    end

    subgraph Support["Supporting subsystems"]
        LLM["src/llm/client.py\nprovider-agnostic (Anthropic/OpenAI/Gemini)\n+ mock mode + JSONL tracing"]
        SANDBOX["src/sandbox/\nAST whitelist + isolated dry-run"]
        TRAINDISPATCH["src/training/dispatch.py\nThreadPoolExecutor job registry\nCV · Optuna tuning · resampling"]
        PROFILING["src/profiling/ + src/pii/\nprofiling · PII redaction · leakage · EDA"]
    end

    FE <-->|"REST + polling"| API
    API --> RUNS
    API -->|"POST /runs"| INTAKE
    API -->|"POST /confirm"| PREP
    API -->|"POST /approve-features"| TRAIN
    Pipeline --> LLM
    Pipeline --> SANDBOX
    Pipeline --> TRAINDISPATCH
    Pipeline --> PROFILING
```

The frontend never talks to LangGraph directly — it polls `GET /api/runs/{id}` every
1.5s and renders whatever `PipelineState` the run has accumulated so far (stage
timeline, live training progress, results table, insights, chat). Each of the three
human-checkpoint boundaries (upload → confirm → approve-features → train) is a
**separate HTTP round trip**, not a LangGraph `interrupt` — see
["How the web API splits the same graph in three"](#how-the-web-api-splits-the-same-graph-in-three).

## The LangGraph pipeline

### PipelineState — the contract between nodes

Every node reads and writes a single typed dict, `PipelineState` (`src/state.py`). Its
key regions:

| Region | Fields | Populated by |
|---|---|---|
| Input | `run_id`, `dataset_path`, `use_case_description` | `new_state()` |
| Profiling | `profile`, `leakage_flags`, `eda_report`, `resampling_suggestion` | `profile_node`, `leakage_check_node`, `eda_node` |
| Task spec | `task_spec` (a `TaskSpec`: target column, task type, metric, time column), `needs_human_confirmation`, `human_confirmed` | `understand_usecase_node`, the `/confirm` endpoint |
| Feature plan | `feature_plan` (a `FeaturePlan` of `FeatureStep`s), `feature_plan_valid`, `feature_plan_feedback`, `resampling_plan`, `transformed_dataset_path`, `training_preprocess_steps` | `feature_engineering_node`, `apply_feature_plan_node`, the `/approve-features` endpoint |
| Modeling | `candidate_models`, `cv_enabled`/`cv_folds`, `tuning_enabled`, `training_run_ids`, `training_results`, `best_model` | `model_selection_node`, `dispatch_training_node`, `poll_training_node`, `evaluate_node` |
| Output | `report`, `errors`, `retry_count`, `status` | `report_node`, routing functions |

Structured Pydantic models (`TaskSpec`, `FeatureStep`/`FeaturePlan`, `CandidateModel`,
`TrainingResult`, `TuningInfo`) back every field that crosses a node boundary — LLM
output is always validated against one of these schemas before it's trusted (CLAUDE.md
rule #2: structured output over free-form code).

### Full pipeline graph (CLI)

`build_graph()` (`src/graph/build_graph.py`) is the complete pipeline `run_local.py`
invokes in one shot, with human checkpoints as **stdin prompts**:

```mermaid
flowchart TD
    START([start]) --> PROFILE

    PROFILE["profile\n(deterministic)"] --> UNDERSTAND
    UNDERSTAND["understand_usecase\n(LLM)"] --> HUMAN1
    HUMAN1{{"human_checkpoint\n(stdin, only if ambiguous)"}} --> LEAKAGE
    LEAKAGE["leakage_check\n(deterministic)"] --> EDA
    EDA["eda\n(deterministic)"] --> FEATENG

    FEATENG["feature_engineering\n(LLM)"]
    FEATENG -- "plan valid" --> HUMAN2
    FEATENG -- "invalid, retries left" --> FEATENG
    FEATENG -- "invalid, retries exhausted" --> REPORT

    HUMAN2{{"feature_approval_checkpoint\n(stdin)"}} --> APPLY
    APPLY["apply_feature_plan\n(deterministic + sandbox)"]
    APPLY -- "applied cleanly" --> MODELSEL
    APPLY -- "failed, retries left" --> FEATENG
    APPLY -- "failed, retries exhausted" --> REPORT

    MODELSEL["model_selection\n(LLM)"] --> DISPATCH
    DISPATCH["dispatch_training\n(async job dispatch)"] --> POLL
    POLL["poll_training\n(poll loop, 2s backoff)"]
    POLL -- "jobs pending, attempts left" --> POLL
    POLL -- "all terminal / attempts exhausted" --> EVAL

    EVAL["evaluate\n(deterministic)"] --> REPORT
    REPORT["report\n(LLM)"] --> END([end])

    classDef llm fill:#efe9fd,stroke:#7c3aed,color:#191627;
    classDef det fill:#dcf5e5,stroke:#15803d,color:#191627;
    classDef checkpoint fill:#fdf0dc,stroke:#b45309,color:#191627;
    class UNDERSTAND,FEATENG,MODELSEL,REPORT llm;
    class PROFILE,LEAKAGE,EDA,APPLY,DISPATCH,POLL,EVAL det;
    class HUMAN1,HUMAN2 checkpoint;
```

Every loop-back edge is capped: `route_after_feature_engineering` and
`route_after_apply_feature_plan` (`src/graph/routing.py`) check
`retry_count["feature_engineering"] < config/runtime.yaml:retry.max_retries` (default
3) before looping, and fall back to `report` (with `status="failed"` and an explanatory
entry in `errors`) if the cap is hit — the pipeline never silently hangs (CLAUDE.md
rule #3). The `poll_training` loop is capped separately by
`training.poll_max_attempts` (default 150 × 2s ≈ 5 minutes).

### How the web API splits the same graph in three

`run_server.py`'s FastAPI backend does **not** run `build_graph()` end-to-end — a
LangGraph run can't pause mid-invocation for a browser round trip. Instead
`src/graph/build_graph.py` exposes three smaller compiled graphs, and
`src/api/server.py` starts the next one only after the corresponding endpoint is
called, in a background thread:

```mermaid
flowchart LR
    U1["POST /api/runs\n(upload + description)"] --> G1
    subgraph G1["build_intake_graph()"]
        direction TB
        P1[profile] --> U["understand_usecase"]
    end
    G1 -->|"status: awaiting_confirmation"| U2

    U2["POST /confirm\n(target/task/metric + cv/tuning config)"] --> G2
    subgraph G2["build_prep_graph()"]
        direction TB
        L2["leakage_check"] --> E2[eda] --> F2["feature_engineering\n(loops on invalid plan)"]
    end
    G2 -->|"status: awaiting_feature_approval"| U3

    U3["POST /approve-features\n(kept steps + resampling choice)"] --> G3
    subgraph G3["build_train_graph()"]
        direction TB
        A3["apply_feature_plan"] --> M3["model_selection"] --> D3["dispatch_training"] --> PO3["poll_training"] --> EV3[evaluate] --> R3[report]
    end
    G3 -->|"status: completed / failed"| DONE(["report + chat available"])
```

Two consequences worth knowing:

- **The API's human checkpoint is stricter than the CLI's.** `build_intake_graph()`
  always ends at `awaiting_confirmation`, whether or not `task_spec.is_ambiguous` —
  the browser always shows the confirm form (there is no silent-continue path), unlike
  the CLI's `human_checkpoint` node which skips the stdin prompt entirely when the task
  spec isn't ambiguous.
- **A failed plan can't silently regenerate after human approval.**
  `route_after_apply_feature_plan_approved` (used only by `build_train_graph()`) has no
  retry loop back to `feature_engineering` — if the *already-approved* plan fails to
  apply, the run fails outright rather than quietly replanning something the user never
  reviewed. The CLI/prep-graph variants (`route_after_apply_feature_plan`,
  `route_after_feature_engineering_prep`) do retry, because nothing has been shown to a
  human yet at that point.

### Run status state machine (API)

```mermaid
stateDiagram-v2
    [*] --> profiling: POST /api/runs
    profiling --> awaiting_confirmation: build_intake_graph finishes
    awaiting_confirmation --> running: POST /confirm
    running --> awaiting_feature_approval: build_prep_graph finishes
    running --> failed: retry cap hit / exception
    awaiting_feature_approval --> running: POST /approve-features
    running --> completed: build_train_graph finishes
    profiling --> cancelled: POST /cancel
    awaiting_confirmation --> cancelled: POST /cancel
    awaiting_feature_approval --> cancelled: POST /cancel
    running --> cancelled: POST /cancel (best-effort, takes effect between graph steps)
    completed --> [*]
    failed --> [*]
    cancelled --> [*]
```

### Node-by-node reference

| Node | Kind | File | What it does |
|---|---|---|---|
| `profile` | deterministic | `src/graph/nodes.py` | Reads the CSV, calls `profile_dataset()` (schema, null rates, cardinality, PII report, correlations/clusters, data-quality block). |
| `understand_usecase` | LLM | `src/agents/understand_usecase_node.py` | Parses the natural-language description + profile into a `TaskSpec`; sets `needs_human_confirmation` if ambiguous. |
| `human_checkpoint` | deterministic (CLI only) | `src/graph/nodes.py` | stdin prompt to correct the task spec, only if ambiguous. The API replaces this with the mandatory `/confirm` endpoint. |
| `leakage_check` | deterministic | `src/graph/nodes.py` → `src/profiling/leakage.py` | Best-effort heuristics for columns that leak the target (name hints, near-perfect correlation, categorical purity). |
| `eda` | deterministic | `src/graph/nodes.py` → `src/profiling/eda.py` | Rule-based feature-step suggestions (impute/encode/scale/drop/datetime-decompose) + a resampling suggestion for imbalanced targets — grounds the LLM's plan in this specific dataset. |
| `feature_engineering` | LLM | `src/agents/feature_engineering_node.py` | Emits a structured `FeaturePlan`; any `custom_code` step is AST-validated here (not yet executed). A deterministic completeness floor adds any EDA-flagged column the LLM's plan skipped. Retries on an invalid plan. |
| `feature_approval_checkpoint` | deterministic (CLI only) | `src/graph/nodes.py` | stdin approval of individual steps + the resampling suggestion. The API replaces this with `/approve-features`. |
| `apply_feature_plan` | deterministic + sandbox | `src/graph/nodes.py` | Applies stateless/structural steps (drop, one-hot/ordinal encode, datetime decompose, most-frequent/constant impute, `custom_code` after a sandboxed dry-run). Statistical steps (mean/median impute, scale, target-encode) are *deferred* into the training job so they fit on the training fold only. |
| `model_selection` | LLM | `src/agents/model_selection_node.py` | Proposes a shortlist with data-aware hyperparameters + rationale. A deterministic completeness floor (`_CANONICAL_ESTIMATORS`) still trains every applicable model family regardless of what the LLM proposed. |
| `dispatch_training` | deterministic | `src/graph/nodes.py` → `src/training/dispatch.py` | Fires off one async job per candidate via `train_model` (`@tool`), returns immediately with `run_id`s — never blocks on training (CLAUDE.md rule #4). |
| `poll_training` | deterministic (loop) | `src/graph/nodes.py` | Polls each job's status with backoff until all are terminal or the attempt cap is hit. |
| `evaluate` | deterministic | `src/graph/nodes.py` | Picks the best candidate by the task spec's metric (lower-is-better for `rmse`/`mae`). |
| `report` | LLM | `src/agents/report_node.py` | Writes the final plain-language narrative from the (already-computed, already-redacted) task spec/feature plan/results — always runs, even on a failed/capped-out run, so every run ends in a clear explanation. |

## Cross-cutting subsystems

**PII redaction (`src/pii/redact.py`)** runs before any profiling output is built
(CLAUDE.md rule #5): column-name hints (`email`, `ssn`, `phone`, …) and value-pattern
matching (regex against a sample) flag PII columns, which are then blanked to
`"[REDACTED]"` in the frame used for anything LLM-facing. The original, un-redacted
frame is only ever touched by deterministic training code, never by an agent.

**Deterministic profiling (`src/profiling/profile.py`)** produces the one artifact
that stands in for raw data everywhere downstream: row/column counts, per-column
dtype/null-rate/cardinality, numeric summaries, correlation pairs (or correlation
*clusters* for wide datasets > 50 columns, to keep the LLM's context bounded), a
**data-quality block** (`completeness`, `duplicate_row_rate`, `uniqueness`, `overall`
— powers the dashboard's Data Quality panel), and at most 5 redacted sample rows.

**Leakage detection (`src/profiling/leakage.py`)** is explicitly best-effort, never
a guarantee — every surfaced flag says so, per CLAUDE.md's open-questions note.

**Structured plans over free-form code.** Both `feature_engineering_node` and
`model_selection_node` emit schema-validated JSON (`FeaturePlan`, `CandidateModel`
list), not code — the one exception is a `custom_code` `FeatureStep`, which must pass
`src/sandbox/validate.py`'s AST whitelist (blocks `eval`/`exec`/`open`/`os`/`sys`/
`subprocess`/network modules/dunder access, requires a top-level `def transform(df)`)
and then a resource-capped, timeout-bounded dry-run on a small sample
(`src/sandbox/execute.py`, via a separate process) before it's ever allowed to touch
the full dataset.

**Async training (`src/training/dispatch.py`)** is the busiest module in the repo:

- A `ThreadPoolExecutor`-backed job registry stands in for Celery/Ray; `train_model`
  (a `@tool`) returns a `run_id` immediately, `poll_training_job` reports status.
- **Cross-validation**: `StratifiedKFold`/`KFold`/`TimeSeriesSplit` depending on task
  type and whether a `time_column` was set; fold count auto-reduces for small/rare
  classes and reports *why* rather than silently omitting CV.
- **Hyperparameter tuning**: Optuna (TPE sampler) searches per candidate, with the
  LLM-proposed hyperparams always scored first as trial 0 — the tuned model can never
  do worse than the untuned baseline. Live per-trial progress is written to the job
  registry so the UI's tuning-progress bars update mid-run.
- **Resampling**: SMOTE / random over-/under-sampling (imbalanced-learn), applied only
  inside the training fold via an `imblearn.pipeline.Pipeline` so synthetic rows never
  leak into a CV test fold or the holdout set. SMOTE auto-falls-back to random
  oversampling when the minority class is too small for its `k_neighbors`.
- **Hyperparameter sanitization**: LLM-proposed `max_features="auto"` for sklearn tree
  ensembles (a value removed in sklearn 1.3+, though still the historical default an
  LLM is likely to suggest) is remapped to its true historical equivalent —
  `"sqrt"` for classifiers, `None` for regressors — in `_build_estimator`, so a very
  plausible LLM suggestion doesn't fail every run.
- **Target-cardinality guardrail**: `POST /confirm` rejects a classification task
  whose target column has more unique values than half the row count
  (`src/profiling/heuristics.target_too_high_cardinality_for_classification`) — such a
  target makes a holdout split structurally unable to generalize (most test-set
  classes were never seen in training) and makes XGBoost's stricter contiguous-label
  validation fail outright.
- All statistical preprocessing (impute/scale/target-encode) lives *inside* the fitted
  sklearn `Pipeline`/`ImbPipeline`, fit on the training fold only — `cross_validate`
  re-fits it per fold, so no test-fold statistic (or, for target encoding, any label)
  ever leaks into training.

**Deterministic auto-insights (`src/insights/auto_insights.py`)** derive
plain-language, tone-coded observations (PII redacted, wide dataset, high null rate,
class imbalance, identifier-like column, model performance) directly from
`PipelineState` — no extra LLM call, so they appear instantly and are fully
explainable. The same module also derives the chat panel's suggested-question chips.

**Provider-agnostic LLM client (`src/llm/client.py`)** reads `config/models.yaml` to
pick a provider (Anthropic/OpenAI/Gemini) and model *per node* — switching a node's
model is a one-line YAML edit, never a code change. `AUTOML_MOCK_LLM=1` swaps every
provider call for a deterministic canned response (still trace-logged), so the entire
pipeline and web UI run with no API keys or network. Every call — prompt, response,
provider/model, error — is appended to `logs/traces/{run_id}.jsonl`
(`src/llm/tracing.py`), a local stand-in for LangSmith, and exposed via
`GET /api/runs/{id}/trace`.

## The AI Assistant chat panel

Once a run reaches `completed`/`failed`, its dashboard's right-rail "AI Assistant"
panel lets you ask questions about *that run's* results. It is deliberately **not** a
LangGraph node — it's an on-demand call (`src/agents/chat_node.py`,
`POST /api/runs/{id}/chat`) invoked directly by the API, gated to only run once the
report exists (409 otherwise). It receives exactly the same already-redacted,
already-computed subset of `PipelineState` the report/frontend already show (profile
summary, EDA insights, leakage flags, feature-plan steps, training results, best
model, report narrative) — never the raw dataset, and no tool access, so the
"raw data never enters an LLM context window" rule holds by construction. Suggested
questions are deterministic (derived from the run's own insights via
`suggested_questions()`), and conversation history is capped to the last 3 exchanges
per call so prompt size stays bounded.

## Frontend

`frontend/` is a single-page, no-build-step vanilla JS/CSS app (`index.html`, `app.js`,
`styles.css`) served directly by FastAPI, polling `GET /api/runs/{id}` every 1.5s.

- **Home view** — a hero/intake form (the working "new experiment" flow), honest trust
  chips (no coding required, runs locally, raw data never reaches the LLM, full audit
  trace), a stats strip and Recent Projects list once runs exist, a live
  Pipeline-in-Progress mini-rail for the active run, and a 5-card "how it works" grid.
- **Run dashboard** — a two-column layout: stage-tracker + live training progress +
  confirm/feature-approval checkpoints + dataset summary + class distribution +
  data-quality overview + model comparison table + tuning-trend chart + feature
  importance in the main column; the AI Assistant panel + recent activity in the right
  rail. Every panel hides itself when its backing data isn't available yet — nothing
  is fabricated.
- **Report / Test / Ask AI** — a tabbed card with the final narrative + model/script
  download + raw LLM trace disclosure, a "test the model" form built from the saved
  pipeline's actual input schema, and the chat panel described above.
- Light/dark theming is CSS-custom-property driven; SVG donuts and the tuning chart
  re-tint on toggle.

## API reference

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/runs` | Multipart upload (`file`, `description`) → `{run_id}`; starts `build_intake_graph()`. |
| GET | `/api/runs` | List runs — `{run_id, filename, status, created_at, description, best_score, metric}`. |
| GET | `/api/runs/{id}` | Full run summary: status, stage timeline, task spec, profile (incl. quality/top_values), leakage flags, feature plan, training results, insights, report, chat history, suggested questions, errors. |
| POST | `/api/runs/{id}/confirm` | `{target_column, task_type, metric, time_column?, cv_enabled, cv_folds, tuning_enabled}` — the mandatory human checkpoint; rejects (400) high-cardinality classification targets; starts `build_prep_graph()`. |
| POST | `/api/runs/{id}/approve-features` | `{approved_step_indices, resampling_enabled, resampling_method}` — starts `build_train_graph()`. |
| POST | `/api/runs/{id}/cancel` | Best-effort cancellation; takes effect between graph steps, not mid-node. |
| GET | `/api/runs/{id}/model` | Download the best model (`.joblib`). |
| GET | `/api/runs/{id}/script` | Download a standalone, dependency-light Python script reproducing the winning run. |
| GET | `/api/runs/{id}/model/schema` | Raw input columns/types the saved pipeline expects (feeds the "test the model" form). |
| POST | `/api/runs/{id}/predict` | `{values}` → prediction (+ class probabilities if available) from the saved model. |
| GET | `/api/runs/{id}/trace` | Full LLM audit trace (every prompt/response/provider/model for this run). |
| POST | `/api/runs/{id}/chat` | `{question}` → `{answer}` — the AI Assistant; 409 until the run is `completed`/`failed`. |

## Repository structure

```
/src
  /agents          # LLM-backed nodes (one file per node) + prompts/*.md + chat_node.py
  /graph            # StateGraph definitions (build_graph.py), routing.py, nodes.py (deterministic)
  /profiling        # profile.py, eda.py, leakage.py, heuristics.py (shared cardinality/imbalance rules)
  /insights         # auto_insights.py — deterministic insights + suggested chat questions
  /tools            # typed @tool functions exposed to LLM agents (row-capped)
  /sandbox          # validate.py (AST whitelist), execute.py (isolated dry-run)
  /training         # dispatch.py — async job registry, CV, Optuna tuning, resampling
  /pii              # redact.py — PII detection/redaction
  /api              # server.py — FastAPI backend
  /llm              # client.py (provider-agnostic + mock mode), tracing.py
  /export           # script_export.py — standalone training-script generator
  state.py          # PipelineState schema (source of truth)
/frontend           # static single-page UI (no build step): index.html, app.js, styles.css
/config
  models.yaml       # provider+model per node
  runtime.yaml       # retry caps, sandbox limits, training/tuning/CV defaults, LLM budgets
/tests
  /fixtures         # synthetic datasets + generate_fixtures.py
run_server.py       # web entrypoint (API + frontend)
run_local.py        # CLI entrypoint (stdin human checkpoints, full build_graph())
```

## Configuration

**`config/models.yaml`** — a `default` block plus per-node overrides (`understand_usecase`,
`feature_engineering`, `model_selection`, `report`, `chat`); each entry sets
`provider`/`model`/`temperature`/`max_tokens`. API keys are read from environment
variables (`.env`, see `.env.example`) — never hardcoded.

**`config/runtime.yaml`** — every cap the pipeline enforces in code, not by
convention: `retry.max_retries` (loop-back cap), `sandbox.*` (timeout, sample rows,
memory), `tools.max_sample_rows` (row-level tool output cap), `training.*`
(concurrent job limit, poll interval/attempts, Optuna trial/budget defaults, default CV
folds), and `budgets.*` (max LLM calls/tokens per run).

## Quickstart (local)

```bash
uv venv .venv
uv pip install -r requirements.txt --python .venv
```

### Web UI (recommended)

```bash
# no API keys needed for a first test drive:
cp .env.example .env        # then set AUTOML_MOCK_LLM=1 in .env
.venv/Scripts/python run_server.py
```

Open http://127.0.0.1:8000 — drop a CSV (generate samples first, below), describe the
prediction goal, confirm the task spec when prompted, review/approve the feature plan,
and watch the pipeline run through to a downloadable model + report + chat.

```bash
# generate sample datasets to play with:
.venv/Scripts/python -m tests.fixtures.generate_fixtures
```

### CLI

```bash
.venv/Scripts/python run_local.py --file tests/fixtures/imbalanced_classification.csv \
    --description "predict which customers will churn"
```

### Switching LLM providers

Per-node provider/model lives in [config/models.yaml](config/models.yaml) — each node
can independently use `anthropic`, `openai`, or `gemini`. Set the matching API key in
`.env`. With `AUTOML_MOCK_LLM=1`, all providers are bypassed with deterministic canned
responses (including for `chat`).

## Testing

```bash
.venv/Scripts/python -m pytest tests/
```

Notable fixtures/suites: highly imbalanced classification, high-cardinality
categoricals, time-series data (leakage-prone chronological split), wide datasets
(500+ columns), datasets with injected PII, ambiguous/missing target columns, malicious
sandboxed code (disallowed imports/infinite loops/wrong schema), and full API
integration tests (intake → confirm → approve-features → train → chat) run with
`AUTOML_MOCK_LLM=1` so no keys/network are needed in CI.

## Local stand-ins vs. production

| Concern | Local (this repo) | Production design |
|---|---|---|
| Job queue | `ThreadPoolExecutor` in-process | Celery/Ray + Redis |
| Sandbox | AST whitelist + subprocess w/ timeout | Docker/gVisor, no network, read-only mounts |
| Run store | in-memory dict | Postgres + model registry |
| Artifacts | `artifacts/` folder | S3-compatible object storage |
| Tracing | JSONL files in `logs/traces/` | LangSmith or equivalent |

## Known limitations / open questions

- **Leakage detection is heuristic, not guaranteed** — every surfaced flag says so
  explicitly; false negatives are expected (`src/profiling/leakage.py`).
- **Target/metric disambiguation always routes to a human checkpoint** rather than
  being silently auto-selected for ambiguous use cases (CLAUDE.md).
- **The exported training script** (`GET /.../script`) is a deterministic
  transcription of `apply_feature_plan`/`dispatch.py`'s logic with no shared code path
  by design (so it's dependency-free) — it can drift if those change without the
  export logic being updated in step.
