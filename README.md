# Agentic AutoML Platform

Upload a dataset, describe your goal in plain language, and a LangGraph-orchestrated
agent pipeline profiles the data, confirms the task with you, engineers features,
trains candidate models asynchronously, and returns the best model with a
plain-language report. Raw data never enters an LLM context window — agents see
only redacted statistical profiles (see [CLAUDE.md](CLAUDE.md) and [PRD.md](PRD.md)).

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

Open http://127.0.0.1:8000 — drop a CSV (generate samples first, below),
describe the prediction goal, confirm the task spec when prompted, and watch
the pipeline run through to a downloadable model + report.

```bash
# generate sample datasets to play with:
.venv/Scripts/python -m tests.fixtures.generate_fixtures
```

### CLI

```bash
.venv/Scripts/python run_local.py --file tests/fixtures/imbalanced_classification.csv \
    --description "predict which customers will churn"
```

### Tests

```bash
.venv/Scripts/python -m pytest tests/
```

## Switching LLM providers

Per-node provider/model lives in [config/models.yaml](config/models.yaml) —
each pipeline node (use-case understanding, feature engineering, model
selection, reporting) can independently use `anthropic`, `openai`, or `gemini`.
Set the matching API key in `.env`. With `AUTOML_MOCK_LLM=1`, all providers are
bypassed with deterministic canned responses.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/runs` | multipart upload (`file`, `description`) → `{run_id}`; starts profiling |
| GET | `/api/runs` | list runs |
| GET | `/api/runs/{id}` | status, progress stages, task spec, leakage flags, results, report |
| POST | `/api/runs/{id}/confirm` | `{target_column, task_type, metric}` — human checkpoint; starts training phase |
| GET | `/api/runs/{id}/model` | download the best model (`.joblib`) |
| GET | `/api/runs/{id}/trace` | full LLM audit trace for the run |

## Local stand-ins vs production

| Concern | Local (this repo) | Production design |
|---|---|---|
| Job queue | `ThreadPoolExecutor` in-process | Celery/Ray + Redis |
| Sandbox | AST whitelist + subprocess w/ timeout | Docker/gVisor, no network, read-only mounts |
| Run store | in-memory dict | Postgres + model registry |
| Artifacts | `artifacts/` folder | S3-compatible object storage |
| Tracing | JSONL files in `logs/traces/` | LangSmith or equivalent |
