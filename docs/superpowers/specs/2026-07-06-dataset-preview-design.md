# Dataset Preview — "Data" Tab Design Spec

Date: 2026-07-06
Status: Approved for planning
Source spec: `DATASET_PREVIEW.md` (full 10-section module); this design implements a scoped subset — see "Scope" below.

## Scope

`DATASET_PREVIEW.md` describes a persistent, versioned "Datasets" registry (uploads independent of any experiment, version history, drift analysis, dataset-level AI assistant, related-experiments cross-linking). The current app is run-centric: a CSV is uploaded as part of starting one experiment; there is no dataset entity that outlives a run. Building the full 10-section module is a multi-week effort spanning several independent specs.

This spec covers **one slice**: the **Data tab** of a dataset detail page — interactive row preview, column summary, correlations, missing values, and outliers — wired to real (if size-limited) computation, reachable from a new "Datasets" section in the sidebar.

**Explicitly out of scope for this spec** (left as disabled/"soon" affordances, matching the sidebar's existing convention):
- Dataset Overview/Profile/Features/Relationships/Versions/Settings tabs' real content
- Dataset versioning / version comparison / drift analysis
- The AI Dataset Assistant panel
- Related Experiments tab
- Quick Actions grid (Launch AutoML / Generate Report / etc. as a dedicated widget)
- "Download dataset" / "Share dataset" as working features

These are structurally impossible without a persistent dataset registry (versioning, drift) or are independent features better scoped separately (AI assistant, quick actions). Buttons/tabs for them render but are non-interactive, consistent with the existing `nav-item disabled` pattern in `frontend/index.html`.

## Dataset identity (key architectural decision)

There is no new database table or persistent entity. A "dataset" **is** a top-level run — one whose `source_run_id` is empty (re-run experiments reuse their source's `dataset_path` and are not separately listed as datasets). The dataset detail page is keyed by that run's `run_id` and reads its already-computed `profile`/`profile_columns`, plus new endpoints that lazily read the underlying CSV.

This means: a dataset shows up in the Datasets list as soon as its run is created (during "profiling"), and its stats reflect whatever that run's `state["profile"]` currently holds. There is exactly one dataset per uploaded file for the lifetime of this app (no re-upload/versioning).

## Backend

### In-memory DataFrame cache

New module-level cache in `src/api/server.py`, alongside the existing `_runs` dict:

```python
_dataset_df_cache: dict[str, pd.DataFrame] = {}
_dataset_df_cache_lock = threading.Lock()

def _get_cached_df(run_id: str, dataset_path: str) -> pd.DataFrame:
    with _dataset_df_cache_lock:
        if run_id not in _dataset_df_cache:
            _dataset_df_cache[run_id] = load_dataset(dataset_path)
        return _dataset_df_cache[run_id]
```

Loaded lazily on first Data-tab API call for that run, not at upload time — avoids adding latency to the existing upload path. Runs are immutable once created (their CSV never changes), so no invalidation is needed.

**Known limitation (stated plainly, not hidden):** this loads the full CSV into memory per run, same as every existing profiling node (`load_dataset` is already called fresh in `nodes.py`, `dispatch.py`, `tools/data_tools.py`). It does **not** implement the source spec's "100M+ rows via server-side pagination without ever loading the full dataset" requirement — it matches this codebase's current local/single-user/in-memory architecture, not the eventual S3/Postgres-backed one described in `CLAUDE.md`. Pagination/sorting/filtering below are real (computed server-side, only a page's worth of rows serialized to JSON per request) but they operate on a fully-loaded DataFrame, so memory scales with dataset size regardless of page size.

### New profiling module: `src/profiling/preview.py`

Deterministic (non-LLM) functions, following the existing pattern in `src/profiling/profile.py` and `eda.py`:

- `paginate_rows(df, page, page_size, sort_by, sort_dir, search) -> dict` — slices, sorts (single column), and substring-filters (case-insensitive, across string-representable columns) a DataFrame; returns `{"rows": [...], "total_count": int, "duplicate_row_indices": [...]}`. `duplicate_row_indices` are indices within the returned page only (via `df.duplicated(keep=False)`), so the frontend can highlight them without a second round trip.
- `column_detail(df, column, target_column) -> dict` — histogram bins for numeric columns (`np.histogram`, 20 bins default), top/rare/random value samples for categorical columns, quartiles/skew/kurtosis, and correlation-with-target when a numeric target is confirmed.
- `correlation_matrix(df, method) -> dict` — full pairwise matrix over numeric columns for `pearson`/`spearman`/`kendall` (`df.corr(method=...)`); for `mutual_info`, pairwise `sklearn.feature_selection.mutual_info_regression` (each numeric column against every other, symmetric matrix, diagonal = 1.0). Capped at 50 numeric columns (matches `WIDE_DATASET_COLUMN_THRESHOLD` reasoning in `profile.py`) — above that, returns a `"truncated": true` flag and the top 50 highest-variance columns only, rather than an O(n²) blowup.
- `missing_value_matrix(df) -> dict` — per-column null count/rate, plus a correlation matrix (phi coefficient via `df.isna().corr()`) of *nullness* between columns that have any nulls, for the "which columns tend to be missing together" view.
- `detect_outliers(df, method, columns) -> dict` — per numeric column for `iqr`/`zscore` (reuses the same IQR logic already in `src/profiling/eda.py::_iqr_outlier_rate`, extracted to a shared helper rather than duplicated); multivariate `isolation_forest`/`lof` via `sklearn.ensemble.IsolationForest` / `sklearn.neighbors.LocalOutlierFactor` over all numeric columns (mean-imputed for the detector only, never mutating the returned data). Returns `{"method": ..., "outlier_count": int, "affected_columns": [...], "example_row_indices": [...]}`, capped at 20 example rows per the tool-output row cap convention (`CLAUDE.md` conventions section).
- `ml_readiness_score(profile, leakage_flags) -> float` — new composite heuristic: `0.4 * quality.completeness + 0.3 * quality.uniqueness + 0.2 * (1 - high_severity_leakage_ratio) + 0.1 * (1 - is_wide_dataset_penalty)`. Documented inline as a heuristic, not a guarantee (same caveat convention as `detect_target_leakage`).

Memory usage is added to `profile_dataset()` in `src/profiling/profile.py` as `profile["memory_bytes"] = int(df.memory_usage(deep=True).sum())` — computed once at profiling time, no new read needed.

Text vs. categorical split (for the KPI "Text Features" count): an object-dtype column counts as "text" when its non-null values' mean string length exceeds 30 characters, else "categorical" — computed in the KPI aggregation step in `preview.py`, not stored on every column (keeps `profile_dataset()` output unchanged for existing consumers).

### New API endpoints (`src/api/server.py`)

All require `Depends(require_session)`, matching every existing `/api/runs*` route. All 404 via the existing `_get_entry` helper.

- `GET /api/datasets` — list of datasets: filters `list_runs()`'s existing logic down to entries where `source_run_id` is empty, shaped for the Datasets list page (filename, row/column count, quality score, status, created_at).
- `GET /api/runs/{id}/preview?page=1&page_size=50&sort_by=&sort_dir=asc&search=` — paginated rows. `page_size` capped at 200 (400 error above that). Column values are **not** PII-redacted (the viewing user owns their own upload; `CLAUDE.md`'s "raw data never enters an LLM context window" rule is about the LLM boundary specifically, not the human-facing UI) but each row's cells matching a PII column are tagged so the UI can badge them, matching the existing `is_pii` badge convention used elsewhere in the dashboard.
- `GET /api/runs/{id}/columns/{name}` — column detail (histogram/samples/stats/target-correlation + any already-computed EDA recommendation for that column, when `eda_report`/`feature_plan` exist on the run; otherwise a `"analyzed": false` flag so the UI shows "run further into the pipeline to see ML insights" rather than fabricating data).
- `GET /api/runs/{id}/correlations?method=pearson` — full correlation matrix.
- `GET /api/runs/{id}/missing-values` — missing value matrix.
- `GET /api/runs/{id}/outliers?method=iqr` — outlier detection result.

Each of the last four validates `method`/`name` against known values (400 on unknown), following the existing `_VALID_RESAMPLING_METHODS` validation pattern.

## Frontend

### Navigation

- `frontend/index.html`: the disabled `Datasets <em>soon</em>` sidebar item becomes a real `nav-item` (`id="nav-datasets"`), following the same markup pattern as `nav-dashboard`/`nav-new`.
- New top-level view `#datasets-view` (sibling to `#intake-view`/`#run-view`), shown/hidden the same way (`showDatasetsView()` alongside the existing `showIntakeView()`/`showRunView()`).
- New view `#dataset-detail-view` for a single dataset, with a breadcrumb (`Datasets > <filename>`) and the 7-tab bar. Only the "Data" tab (`#tab-data-btn`) is enabled; the other six render with a `disabled`/muted style and a "soon" badge, matching `.nav-item.disabled`.

### Datasets list (`#datasets-view`)

Card grid or table (reusing `.card`/table styles already in `styles.css`) driven by `GET /api/datasets`: filename, rows, columns, quality score badge, status badge (reusing the existing `.status-badge` classes), created_at. Clicking a row opens `#dataset-detail-view` for that `run_id`.

### Data tab

1. **KPI row** — reuses `.stat-row`/`.stat-card` styling from the existing run dashboard. Cards: Total Rows, Total Columns, Missing Values (%), Duplicate Rows, Memory Usage (formatted via existing `formatDuration`-style helper, new `formatBytes()`), Numeric/Categorical/Datetime/Text feature counts, Target Column + mini distribution (reuses the existing donut renderer), Data Quality Score (reuses `.quality-ring`), ML Readiness Score (new ring, same SVG pattern).
2. **Interactive preview table** — new component, not a reuse of `.results-table` (different feature set: virtualized/paginated, resizable/reorderable columns). Sticky header + first column via CSS `position: sticky`. Column viz toggle and layout (widths/order/visibility) persisted to `localStorage` under a per-dataset key (`automl-preview-layout-${runId}`). Type-aware cell rendering per column `dtype` (from `profile_columns`): numeric (right-aligned, `--cat` heatmap tint scaled within that page's min/max), categorical (colored badge, reusing chip styles), boolean (green/red pill), datetime (human format via existing date handling), text (truncate + `title` tooltip expand). Missing cells get a distinct muted style; duplicate rows (from the API's `duplicate_row_indices`) get a left-border tint, consistent with existing `tr.best` styling in `.results-table`.
3. **Column Explorer panel** — a slide-over or inline panel (reusing `.card` styling) triggered by a per-column "Profile" icon button in the table header, fed by `GET /api/runs/{id}/columns/{name}`. Shows histogram/value chart (SVG, following the existing tuning-trend chart's hand-rolled SVG approach — no new charting library, keeping with "no build step"), stats, target correlation, and EDA-derived insights when available.
4. **Profiling sub-tabs** below the table: Column Summary (static table from existing `profile_columns`), Correlations (SVG heatmap + method `<select>` calling the new endpoint), Missing Values (matrix + heatmap), Outliers (method `<select>`, per-column counts, and a mini preview table scoped to `example_row_indices`, reusing the preview table renderer).

### Styling

Extends `styles.css` following its existing CSS-custom-property theme system (light/dark via `:root[data-theme]`), the same `--cat-*` donut palette, and the same card/chip/badge vocabulary already established — no new design language introduced.

## Testing

- Backend: unit tests for each new `preview.py` function (pagination bounds, sort/search correctness, histogram bin counts, correlation matrix symmetry, outlier method dispatch, `ml_readiness_score` bounds `[0, 1]`) — new `tests/test_preview.py`, following the existing `/tests` module-per-source-file convention.
- Fixture coverage: run the new endpoints against the existing `/tests/fixtures` datasets (high-cardinality categoricals, wide 500+ column data — exercises the 50-column correlation cap, PII datasets — exercises the PII badge-not-redact behavior) per `CLAUDE.md`'s testing expectations for changes touching profiling.
- Tool-output cap test: assert `/columns/{name}` and `/outliers` never return more than their documented row caps (20), matching the existing convention for tool functions.
- Manual UI verification via the `verify` skill once implemented: upload a dataset, open its Datasets entry, paginate/sort/search the preview table, switch correlation/outlier methods, confirm disabled tabs are inert.

## Open questions deferred to implementation

None blocking — the four analytics endpoints, KPI heuristics, and frontend component boundaries above are specified concretely enough to plan from directly.
