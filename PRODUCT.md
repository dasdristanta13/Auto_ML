# PRODUCT.md — Frontend Product Spec

Scope: this document defines the product surface for the platform's web UI. It complements PRD.md (system-level product requirements) with frontend-specific screens, states, and interaction requirements.

## 1. Primary Personas & Jobs-to-be-Done

- **Business user / analyst**: uploads data, describes a use case in plain language, wants a model + explanation without writing code.
- **Data scientist**: wants to inspect/override automated decisions, export pipeline code, iterate faster than manual work.

Both personas need to **trust** the output before using it. The UI's core job is making an autonomous, multi-step agentic process legible and controllable at every checkpoint — not just showing a final result.

## 2. Information Architecture

```
/datasets                — list, upload, connect
/datasets/:id            — profile view (schema, stats, quality flags)
/pipelines/new           — use-case intake flow
/pipelines/:id           — live run view (graph progress, streaming logs)
/pipelines/:id/report    — final report (metrics, rationale, feature importance)
/models                  — registry, versions, comparisons
/models/:id              — model detail, download, export code, deploy (future)
/settings                — org, budgets, connectors, PII rules
```

## 3. Core Screens & Requirements

### 3.1 Dataset Upload & Profile View
- Upload via drag-drop file or connector (Postgres/Snowflake/BigQuery — v2).
- On upload: show parsing/profiling progress (this can take real time on large files — never a silent spinner with no feedback).
- Profile view shows: row/column counts, per-column type + null% + cardinality, flagged PII columns (clearly marked, explain that they'll be redacted from any AI-facing step), flagged potential target leakage, correlation highlights.
- Never render more than a capped sample of raw rows (≤20) in the UI — same constraint as the backend. Make this visible as a trust signal, not a limitation to hide: "Showing 12 of 340,000 rows — your full dataset is never sent to the AI model."

### 3.2 Use-Case Intake
- Single, prominent free-text input: "What do you want to predict or solve?"
- System responds with its structured interpretation: task type, target column, proposed metric — shown as **editable, not just displayed**. Every inferred field must have a visible affordance to correct it before the pipeline starts.
- If ambiguous (multiple plausible targets, unclear metric), block progression with a clarifying question — never silently guess and proceed.
- Show an estimated cost/time range before the user commits to running the pipeline.

### 3.3 Live Pipeline Run View
- This is the highest-trust-risk screen: an autonomous system is running unattended for minutes. The UI must never feel like a black box.
- Show the pipeline as a **sequence of named stages** (profiling → cleaning → feature engineering → model search → training → evaluation), each with a live status (pending / running / done / skipped / needs input).
- Stream key decisions as they happen in plain language ("Detected class imbalance (8% positive) — applying SMOTE resampling"), not raw logs by default. Raw logs available behind a "view details" toggle for technical users.
- Retry/iteration loops must be visible, not hidden: if a step fails and retries, show that explicitly with a capped counter ("Attempt 2 of 3"), never let it look stuck.
- Any human-in-loop checkpoint pauses the run and surfaces a clear, specific question with the decision options — not a generic "input needed" banner.
- Always show a way to cancel a running pipeline.

### 3.4 Report View
- Leads with the outcome: model performance against the chosen metric, in plain language before raw numbers ("This model correctly identifies 82% of customers who churn").
- Shows model comparison (all candidates tried, not just the winner) so the user can see the search wasn't arbitrary.
- Feature importance visualization, explained in terms of the user's actual columns, not generic ML terms.
- A dedicated, unmissable **caveats section**: leakage risks flagged but unresolved, data quality issues, metric tradeoffs made. This is not optional fine print — it's core to the report's credibility.
- Actions: download model, export pipeline as code, request iteration with a new constraint (reruns only the affected stages, not the whole pipeline).

### 3.5 Model Registry
- List/compare all trained models across runs, with metrics, dataset lineage, and creation date.
- Version history per model; ability to roll back or re-derive.

## 4. Cross-Cutting Requirements

### 4.1 Trust & Transparency (product-critical, not polish)
- Every automated decision anywhere in the product must be traceable to a one-line rationale, visible on hover/click, not buried in logs.
- The product must never present a guess as a fact. Distinguish clearly between "detected" (data-driven, high confidence) and "inferred" (LLM judgment call, needs confirmation).

### 4.2 Progress & Latency Honesty
- No indefinite spinners. Every long-running action shows what's happening and, where possible, an estimated remaining time or step count.
- Background/async jobs (training) must be safe to navigate away from — user returns later to a persisted state, not a lost run.

### 4.3 Error & Empty States
- Errors state what happened and what to do next, in the product's voice — never a raw stack trace to non-technical users (available behind a details toggle for technical users).
- Empty states (no datasets yet, no models yet) are calls to action, not dead ends.

### 4.4 Cost & Budget Visibility
- Any action with a real compute/token cost (starting a pipeline, re-running a stage) shows an estimate before commit and actual cost after completion.

### 4.5 Accessibility & Responsiveness
- Keyboard-navigable throughout, visible focus states, reduced-motion respected.
- Core flows (upload, intake, report) must work down to tablet width at minimum; live pipeline view should degrade gracefully on mobile (status summary rather than full graph).

## 5. Out of Scope (v1 Frontend)
- In-browser dataset editing/cleaning tools (beyond confirming/correcting AI proposals).
- Real-time collaborative editing of a pipeline by multiple users.
- Custom visual pipeline builder (drag-and-drop DAG editing) — v1 is guided/conversational, not a visual programming tool.