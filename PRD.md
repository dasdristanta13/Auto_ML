# Product Requirements Document: Agentic AutoML Platform

**Status:** Draft v1.0
**Owner:** [TBD]
**Last updated:** July 2026

---

## 1. Overview

### 1.1 Problem Statement
Building a machine learning model today requires a data scientist to manually profile data, engineer features, select algorithms, tune hyperparameters, and interpret results. This is slow, requires specialized expertise, and doesn't scale to the volume of "give me a model for X" requests that businesses generate.

### 1.2 Product Vision
A platform where a user uploads a dataset, describes their use case in plain language (e.g., "predict which customers will churn next month"), and an agentic system autonomously plans and executes the entire ML pipeline — profiling, cleaning, feature engineering, model selection, training, and evaluation — returning a trained model, performance report, and human-readable rationale for every decision made.

### 1.3 Core Constraint
Raw data must never be sent to the LLM in bulk. The LLM acts as a planner and orchestrator over statistical summaries and tool calls; all data-heavy operations happen in a deterministic execution layer. This constraint is both a cost/latency requirement and a data-privacy requirement (many prospective customers will have data governance policies preventing raw data from leaving their environment or hitting a third-party LLM API in full).

### 1.4 Goals
- Reduce time-to-first-model from days/weeks to under an hour for standard tabular use cases.
- Make ML accessible to users without data science expertise, without producing a black box — every decision must be explainable.
- Keep the system safe to run on arbitrary customer data without a human manually reviewing every generated pipeline before execution.

### 1.5 Non-Goals (v1)
- Deep learning / unstructured data (images, text, audio) — v1 scope is tabular data only.
- Fully autonomous production deployment without human sign-off (v1 stops at a validated model + report; deployment is a separate, explicitly-triggered step).
- Real-time/streaming training pipelines.
- Multi-dataset joins/relational schema reasoning (v1 assumes a single flat dataset per pipeline run).

---

## 2. Users & Use Cases

### 2.1 Target Users
- **Primary**: Analysts/PMs/domain experts who understand their business problem but lack ML engineering skills.
- **Secondary**: Data scientists who want to accelerate the boilerplate (profiling, baseline models) and focus on refinement.

### 2.2 Representative Use Cases
- Churn prediction from customer usage data
- Credit risk / default prediction
- Demand forecasting from historical sales data
- Lead scoring from CRM data
- Fraud/anomaly flagging from transaction logs

### 2.3 User Journey (Happy Path)
1. User uploads a dataset (CSV/Parquet/DB connection).
2. User describes the use case in natural language.
3. Platform profiles the data and confirms its understanding of the task (target column, task type, metric) — user confirms or corrects.
4. Platform runs the full pipeline autonomously, streaming progress (profiling → cleaning → feature engineering → model search → training → evaluation).
5. Platform returns: best model, performance metrics, comparison of candidate models tried, feature importance, a plain-language rationale of key decisions, and caveats/limitations.
6. User can download the model, export the pipeline code, or request iteration ("optimize for recall instead," "try excluding column X").

---

## 3. Functional Requirements

### 3.1 Data Ingestion
- FR-1: Support CSV, Parquet, and JSON file upload up to [define size limit, e.g., 5GB].
- FR-2: Support direct connection to common data warehouses (Postgres, Snowflake, BigQuery) — v2 candidate if not v1.
- FR-3: Validate file integrity and schema parseability on upload; reject/flag corrupt files with a clear error.

### 3.2 Data Profiling
- FR-4: Generate a statistical profile of the dataset (schema, dtypes, null rates, cardinality, distributions, correlations) without loading the full dataset into any LLM context.
- FR-5: Detect and flag PII columns automatically; redact before any downstream LLM-facing artifact is constructed.
- FR-6: Support profiling of wide datasets (500+ columns) via column clustering/summarization rather than exhaustive per-column output.
- FR-7: Detect target leakage candidates and surface them to the user/agent before training.

### 3.3 Use-Case Understanding
- FR-8: Parse natural-language use-case descriptions into a structured task specification: task type (classification/regression/forecasting/clustering), target column, success metric, and constraints (e.g., "must be interpretable," "optimize for recall").
- FR-9: If the task specification is ambiguous (e.g., multiple plausible target columns, no clear metric), trigger a human-in-loop confirmation step rather than guessing silently.
- FR-10: Allow the user to correct the platform's understanding at this checkpoint before any compute-heavy work begins.

### 3.4 Pipeline Planning & Execution (LangGraph Orchestration)
- FR-11: Orchestrate the pipeline as a directed graph with explicit, auditable state at every step.
- FR-12: Route dynamically based on detected data characteristics (e.g., class imbalance → resampling subgraph; time-series data → time-aware split subgraph).
- FR-13: Cap all retry/iteration loops (feature engineering validation, model iteration) at a configurable maximum; fail gracefully with a clear explanation if the cap is reached.
- FR-14: Emit transformation logic as structured, schema-validated plans wherever the operation is a known pattern; fall back to freeform LLM-generated code only for genuinely custom transforms.

### 3.5 Feature Engineering
- FR-15: Support standard transformations (imputation, encoding, scaling, binning, datetime decomposition) via structured plans.
- FR-16: Support custom transformation code generation for cases not covered by standard patterns, subject to sandbox validation (see 3.8).
- FR-17: Validate every transformation's output schema/dtypes against expectations before applying it to the full dataset.

### 3.6 Model Selection & Training
- FR-18: Propose a shortlist of candidate algorithms appropriate to the task type and data characteristics (e.g., not proposing linear regression alone for a highly nonlinear, high-cardinality categorical dataset).
- FR-19: Dispatch training as asynchronous jobs; never block an LLM call on training completion.
- FR-20: Support hyperparameter search within a bounded budget (time/compute caps configurable per plan tier).
- FR-21: Track all training runs (hyperparameters, metrics, duration, resource usage) in a model registry.

### 3.7 Evaluation & Reporting
- FR-22: Evaluate candidate models against the success metric defined in the task specification.
- FR-23: Generate a plain-language report explaining: what was done, why, what worked, what didn't, and any caveats (e.g., "leakage risk in column X was flagged but not fully resolved").
- FR-24: Provide feature importance / explainability output (e.g., SHAP values) for the selected model.
- FR-25: Allow the user to request iteration with a new constraint without restarting the entire pipeline from scratch.

### 3.8 Safety & Sandboxing
- FR-26: All LLM-generated code must be statically validated (import whitelist, no filesystem/network/subprocess access) before execution.
- FR-27: All LLM-generated code must be dry-run on a small data slice in an isolated sandbox before running against the full dataset.
- FR-28: Sandbox execution must be resource-capped (CPU, memory, wall-clock time) and have no network access.
- FR-29: Failed validation/dry-runs must feed structured error feedback back to the LLM for self-correction, subject to the retry cap in FR-13.

### 3.9 Observability & Auditability
- FR-30: Log every LLM call (prompt, tool calls, response) associated with a pipeline run, retrievable for debugging and compliance review.
- FR-31: Every automated decision (metric choice, algorithm choice, feature transform) must be traceable to a rationale in the final report — no unexplained decisions.

---

## 4. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Data privacy** | No raw data rows beyond a capped, redacted sample (≤20 rows) may enter any LLM prompt, under any code path. |
| **PII handling** | PII must be detected and redacted before profiling output is generated, not after. Configurable PII detection rules per customer/region (e.g., GDPR vs. non-regulated environments). |
| **Security** | LLM-generated code execution must be isolated (containerized/microVM), with no network egress and read-only data mounts. |
| **Cost control** | Token/compute budgets must be enforced per pipeline run, with hard stops and graceful degradation (partial report) rather than silent overruns. |
| **Latency** | Standard tabular dataset (<1M rows, <100 columns) should complete profiling-to-first-model within [define target, e.g., 30 minutes]. |
| **Reliability** | Retry loops must be capped; a failed pipeline must always terminate in a clear, user-facing explanation, never a silent hang or infinite loop. |
| **Scalability** | Profiling and training must support datasets larger than available memory via sampling or distributed compute (Spark/Dask), not by assuming everything fits in a pandas DataFrame. |
| **Explainability** | Every model output must be accompanied by feature importance and a natural-language rationale; no "black box, trust us" outputs. |
| **Auditability** | Full reasoning/tool-call traces must be retained and queryable per pipeline run for a defined retention period. |

---

## 5. System Architecture (Summary)

See accompanying architecture doc for full detail. Key components:
- **Profiling layer** (deterministic) — generates the statistical summary that replaces raw data in LLM context.
- **PII redaction layer** — scrubs sensitive data before any LLM-facing artifact exists.
- **LangGraph orchestrator** — stateful, conditional-routing pipeline connecting all agent nodes.
- **Tool layer** — typed, capped-output functions bridging LLM decisions to real data operations.
- **Sandbox executor** — isolated environment for validating/running LLM-generated code.
- **Async training infrastructure** — job queue (Celery/Ray) decoupling LLM orchestration from long-running compute.
- **Model registry** — versioned storage of trained models, metrics, and metadata.
- **Observability layer** — full trace logging of agent decisions for debugging/compliance.

---

## 6. Caveats, Risks & Open Questions

### 6.1 Technical Risks
- **Profile fidelity gap**: statistical summaries can obscure multimodal distributions, rare-but-important categories, or subtle feature interactions the LLM never sees. Mitigation: allow agents to request targeted drill-downs rather than relying solely on a static upfront profile.
- **Code hallucination**: LLM-generated transformation/feature code can be subtly incorrect even when it executes without error. Mitigation: schema/type validation post-execution, not just pre-execution syntax checks; prefer structured plans over free-form code wherever possible.
- **Leakage detection is heuristic, not guaranteed**: automated target leakage detection will have false negatives. This must be clearly communicated in every report, not presented as a guarantee.
- **Non-determinism**: LLM planning decisions may vary run-to-run even at low temperature. This affects reproducibility claims — decide and document what reproducibility guarantee (if any) the platform makes.
- **Cost/latency compounding**: multi-step agentic pipelines with retry loops can produce unpredictable cost per run. Requires hard budgets and clear user-facing cost estimates before execution, especially for larger datasets or expensive hyperparameter search.

### 6.2 Product/Business Risks
- **Trust**: users will not adopt an "automated model" without transparent rationale and the ability to inspect/override decisions. Explainability is not optional polish — it's core to adoption.
- **Metric ambiguity**: business use cases often don't map cleanly to a single metric (e.g., "reduce churn" could optimize for precision, recall, or a cost-weighted metric). Silent guessing here produces models that are technically correct but business-wrong. Requires a clarification step, not an assumption.
- **Scope creep on data types**: pressure to support text/image data early will be strong; this significantly increases complexity (different profiling, different sandboxing needs, different privacy risk surface) and should be explicitly deferred past v1.
- **Regulatory exposure**: depending on target customer verticals (finance, healthcare), model explainability and data handling may need to meet specific regulatory bars (e.g., adverse action explanations in credit use cases). This should be scoped per target market before GA.

### 6.3 Open Questions (need decisions before build)
- What is the maximum dataset size supported in v1, and what's the fallback UX when a dataset exceeds it (sampling? rejection? tiered pricing?)
- Which LLM model tier is used at each node (planning vs. code-gen vs. report generation) — cost/quality tradeoff needs explicit answers, not left to default.
- What reproducibility guarantee does the platform make for a given pipeline run (bit-for-bit identical model, or "statistically similar")?
- Does v1 support any human-editable intermediate step (e.g., user reviews/edits the feature engineering plan before training), or is it fully autonomous until the final report?
- What is the data retention/deletion policy for uploaded datasets and generated intermediate artifacts (profiles, samples, logs)?
- Is on-premise/VPC deployment a requirement for target enterprise customers, given the data-privacy constraint already baked into the architecture?

---

## 7. Success Metrics

- **Time-to-model**: median time from dataset upload to delivered model/report for standard tabular use cases.
- **Pipeline completion rate**: % of pipeline runs that complete successfully without hitting a retry cap / failure state.
- **User override rate**: % of automated decisions (target column, metric, algorithm choice) that users manually correct — a proxy for how well the understanding/planning stage performs.
- **Model quality vs. baseline**: performance of platform-produced models vs. a manually-built baseline on the same dataset (internal benchmark suite).
- **Explainability satisfaction**: user-rated clarity/trust in the generated rationale reports (survey-based, qualitative in v1).
- **Cost per successful pipeline run**: LLM token cost + compute cost, tracked to ensure unit economics hold as usage scales.

---

## 8. Rollout Plan (Suggested)

1. **Internal alpha**: run against a benchmark suite of synthetic + anonymized real datasets covering the fixture categories in the engineering test plan (imbalanced classification, high-cardinality categoricals, time-series, wide datasets, PII-heavy datasets, ambiguous targets).
2. **Closed beta**: small set of design-partner customers with clear use cases (e.g., churn prediction), tight feedback loop on explainability and override rate.
3. **GA scoping**: revisit non-goals (deployment automation, multi-dataset joins, unstructured data) based on beta feedback before committing to v2 scope.

---

## 9. Appendix: Related Documents
- Technical Architecture Doc (LangGraph node/edge design, state schema, tool layer, sandbox design)
- CLAUDE.md (engineering conventions and non-negotiable architectural rules for this repository)
