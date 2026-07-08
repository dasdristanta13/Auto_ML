/* Agentic AutoML frontend — vanilla JS, no build step.
   modern_ui: dashboard shell (sidebar / stat cards / panels) in light+dark
   themes. Every widget is fed by real API data; polls GET /api/runs/{id}. */

const SVG = (d, extra = "") =>
  `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ${extra}>${d}</svg>`;

const ICONS = {
  check: SVG('<path d="M20 6 9 17l-5-5"/>', 'stroke-width="2.5"'),
  warning: SVG('<path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>'),
  error: SVG('<circle cx="12" cy="12" r="9"/><path d="m9.5 9.5 5 5m0-5-5 5"/>'),
  db: SVG('<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>'),
  chat: SVG('<path d="M21 12a8 8 0 0 1-8 8H4l2.5-2.5A8 8 0 1 1 21 12Z"/>'),
  userCheck: SVG('<circle cx="9" cy="8" r="4"/><path d="M2 21c0-3.9 3.1-7 7-7 1.4 0 2.7.4 3.8 1.1"/><path d="m15 18 2 2 4-4"/>'),
  search: SVG('<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>'),
  sliders: SVG('<path d="M4 21v-7m0-4V3m8 18v-9m0-4V3m8 18v-5m0-4V3M1 14h6M9 8h6m2 8h6"/>'),
  grid: SVG('<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 3v18"/>'),
  layers: SVG('<path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/>'),
  send: SVG('<path d="m22 2-11 11"/><path d="M22 2 15 22l-4-9-9-4 20-7Z"/>'),
  cpu: SVG('<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m-3 6h3m14-6h3m-3 6h3"/>'),
  gauge: SVG('<path d="M12 15 8.5 8.5"/><path d="M3 12a9 9 0 1 1 18 0"/><path d="M3 12h2m14 0h2"/>'),
  file: SVG('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z"/><path d="M14 2v6h6"/>'),
  trophy: SVG('<path d="M8 21h8m-4-4v4M7 4h10v6a5 5 0 0 1-10 0V4Z"/><path d="M7 6H4a2 2 0 0 0 2 5m11-5h3a2 2 0 0 1-2 5"/>'),
  clock: SVG('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/>'),
  sparkle: '<svg class="icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2 14 9 21 12 14 15 12 22 10 15 3 12 10 9Z"/></svg>',
  shield: SVG('<path d="M12 22s8-3.6 8-10V5l-8-3-8 3v7c0 6.4 8 10 8 10Z"/>'),
  bulb: SVG('<path d="M9 18h6M10 22h4M12 2a6 6 0 0 0-4 10.5c.6.6 1 1.4 1 2.5h6c0-1.1.4-1.9 1-2.5A6 6 0 0 0 12 2Z"/>'),
  chevron: SVG('<path d="m6 9 6 6 6-6"/>'),
};

const STAGES = [
  { node: "profile", label: "Data profiling", icon: "db" },
  { node: "understand_usecase", label: "Understanding", icon: "chat" },
  { node: "confirm", label: "Your review", icon: "userCheck" },
  { node: "leakage_check", label: "Leakage check", icon: "search" },
  { node: "eda", label: "Data exploration", icon: "bulb" },
  { node: "feature_engineering", label: "Feature plan", icon: "sliders" },
  { node: "feature_approval", label: "Your review", icon: "userCheck" },
  { node: "apply_feature_plan", label: "Transform", icon: "grid" },
  { node: "model_selection", label: "Model search", icon: "layers" },
  { node: "dispatch_training", label: "Dispatch", icon: "send" },
  { node: "poll_training", label: "Training", icon: "cpu" },
  { node: "evaluate", label: "Evaluation", icon: "gauge" },
  { node: "report", label: "Report", icon: "file" },
];

const JOURNEY_GROUPS = [
  { label: "Data Received", nodes: ["profile"] },
  { label: "Data Inspection", nodes: ["leakage_check", "eda"] },
  { label: "Feature Engineering", nodes: ["feature_engineering", "apply_feature_plan"] },
  { label: "Model Search", nodes: ["model_selection", "dispatch_training", "poll_training"] },
  { label: "Evaluation", nodes: ["evaluate"] },
  { label: "Champion Selected", nodes: ["report"] },
];

/* Plain-language copy for the reasoning rail — describes what each
   deterministic/LLM pipeline stage actually does and why, independent of
   any single run's data (DESIGN.md §4.1: distinguish "detected" facts from
   product-authored explanation, never fabricate per-run specifics here). */
const STAGE_DETAILS = {
  profile: {
    what: "Scanning your file to build a statistical profile — types, null rates, cardinality, and PII flags.",
    why: "This profile is the only thing later steps see about your data; raw rows are never sent to the AI model.",
  },
  understand_usecase: {
    what: "Interpreting your goal into a structured task: target column, task type, and success metric.",
    why: "Getting this right up front means the rest of the pipeline optimizes for what you actually asked for.",
  },
  confirm: {
    what: "Waiting for you to confirm or correct the inferred task before any heavy compute runs.",
    why: "Nothing trains until you've reviewed the target column, task type, and metric.",
  },
  leakage_check: {
    what: "Checking whether any column could be leaking information from the target or the future.",
    why: "A leaking column can make a model look great in testing and fail in production.",
  },
  eda: {
    what: "Running exploratory analysis — correlations, distributions, class balance — to ground the feature plan in your data.",
    why: "Feature suggestions are based on statistics computed here, not on the AI's guess.",
  },
  feature_engineering: {
    what: "Drafting a feature plan: imputation, encoding, and scaling steps tailored to what the analysis found.",
    why: "Every step is a structured, reviewable plan — never free-form code run unsupervised.",
  },
  feature_approval: {
    what: "Waiting for your review of the suggested feature steps.",
    why: "You can uncheck anything before it's applied to your data.",
  },
  apply_feature_plan: {
    what: "Applying the approved feature steps to build the training-ready dataset.",
    why: "Only the steps you approved are applied — no silent extras.",
  },
  model_selection: {
    what: "Comparing candidate model families suited to your task type and data size.",
    why: "A broad search means the final pick isn't just the first thing that was tried.",
  },
  dispatch_training: {
    what: "Queuing training jobs for each candidate model, dispatched asynchronously.",
    why: "Training never blocks the agent; it runs in the background and is polled for progress.",
  },
  poll_training: {
    what: "Training and cross-validating each candidate, including hyperparameter search where enabled.",
    why: "Every candidate is evaluated the same way so the comparison is fair.",
  },
  evaluate: {
    what: "Scoring every trained candidate on held-out data and selecting the best by your chosen metric.",
    why: "The winner is picked by the metric you confirmed, not by which model happened to finish first.",
  },
  report: {
    what: "Writing the plain-language report: what the model does well, feature importance, and caveats.",
    why: "Caveats and limitations are always included — never softened into pure marketing language.",
  },
};

const METRICS = {
  classification: ["f1", "accuracy", "roc_auc"],
  regression: ["rmse", "mae", "r2"],
  forecasting: ["rmse", "mae", "r2"],
};

/* donut palette — set via CSS custom properties so it re-validates per theme */
const DONUT_KEYS = ["--cat-1", "--cat-2", "--cat-3", "--cat-4"];

const $ = (id) => document.getElementById(id);

/* Wraps every /api/runs* call: on a 401 (missing/expired session — see
   docs/superpowers/specs/2026-07-05-login-page-design.md), redirect to the
   login page instead of letting the caller's existing error handling show
   a confusing "failed to..." message for what's actually a logged-out
   session. Uses window.fetch explicitly so this definition is never itself
   rewritten by the blanket fetch()->authFetch() replacement above it. */
async function authFetch(url, options) {
  const res = await window.fetch(url, options);
  if (res.status === 401) {
    window.location.href = "/login.html";
    throw new Error("session expired — redirecting to login");
  }
  return res;
}

/* Fires immediately on script load; doesn't block the rest of this file's
   synchronous setup below, but in practice the very first authFetch() call
   (loadRecentRuns(), at the bottom of this file) will also redirect within
   the same tick if the session turns out to be missing — this is a fast
   UX path, not the security boundary (that's the server's 401s above). */
(async function guardSession() {
  try {
    const res = await window.fetch("/api/auth/session");
    const data = await res.json();
    if (!data.authenticated) window.location.href = "/login.html";
  } catch {
    window.location.href = "/login.html";
  }
})();

let pollTimer = null;
let currentRunId = null;
let currentRunStatus = null;
let selectedFile = null;
let lastRun = null;
let predictFormLoadedFor = null;
let explainabilityLoadedFor = null;
let currentDatasetRunId = null;
let previewState = { page: 1, pageSize: 50, sortBy: null, sortDir: "asc", search: "" };
let previewColumns = [];

/* ================= theme ================= */

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("automl-theme", theme);
  const isDark = theme === "dark";
  $("theme-label").textContent = isDark ? "Light mode" : "Dark mode";
  document.querySelector(".icon-moon").classList.toggle("hidden", isDark);
  document.querySelector(".icon-sun").classList.toggle("hidden", !isDark);
  $("theme-toggle").setAttribute("aria-pressed", String(isDark));
  if (lastRun) {
    renderDatasetSummary(lastRun); // re-tint donuts for the new surface
    renderClassDistribution(lastRun);
  }
}
$("theme-toggle").addEventListener("click", () => {
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});
applyTheme(localStorage.getItem("automl-theme") || "light");

/* ================= nav ================= */

$("nav-dashboard").addEventListener("click", () => {
  if (lastRun) showRunView();
  else showIntakeView();
});
$("nav-new").addEventListener("click", showIntakeView);
$("new-run-btn").addEventListener("click", showIntakeView);
$("nav-datasets").addEventListener("click", showDatasetsView);

function setActiveNav(id) {
  document.querySelectorAll(".nav-item.active").forEach((el) => el.classList.remove("active"));
  $(id).classList.add("active");
}

function showIntakeView() {
  stopPolling();
  currentRunId = null;
  $("run-view").classList.add("hidden");
  $("datasets-view").classList.add("hidden");
  $("dataset-detail-view").classList.add("hidden");
  $("intake-view").classList.remove("hidden");
  setActiveNav("nav-new");
  $("header-eyebrow").textContent = "Agentic AutoML";
  $("run-title").textContent = "Start an experiment";
  $("run-desc").textContent = "Upload data, describe your goal, get a model — explained.";
  $("header-tags").classList.add("hidden");
  $("header-tags").innerHTML = "";
  $("run-breadcrumb").classList.add("hidden");
  $("status-badge").classList.add("hidden");
  $("share-btn").classList.add("hidden");
  $("export-report-btn").classList.add("hidden");
  $("cancel-btn").classList.add("hidden");
  $("rerun-btn").classList.add("hidden");
  $("rerun-card").classList.add("hidden");
  $("reasoning-rail").classList.add("hidden");
  selectedFile = null;
  dropzone.classList.remove("has-file");
  $("dropzone-label").innerHTML = "<strong>Drop a CSV here</strong> or click to browse";
  $("estimate-row").innerHTML = "";
  $("description").value = "";
  updateSubmit();
  loadRecentRuns();
  history.replaceState(null, "", window.location.pathname);
}

function showRunView() {
  $("intake-view").classList.add("hidden");
  $("datasets-view").classList.add("hidden");
  $("dataset-detail-view").classList.add("hidden");
  $("run-view").classList.remove("hidden");
  setActiveNav("nav-dashboard");
}

function showDatasetsView() {
  stopPolling();
  currentRunId = null;
  $("intake-view").classList.add("hidden");
  $("run-view").classList.add("hidden");
  $("dataset-detail-view")?.classList.add("hidden");
  $("datasets-view").classList.remove("hidden");
  setActiveNav("nav-datasets");
  $("share-btn").classList.add("hidden");
  $("export-report-btn").classList.add("hidden");
  $("header-tags").classList.add("hidden");
  $("run-breadcrumb").classList.add("hidden");
  loadDatasetsList();
  history.replaceState(null, "", window.location.pathname);
}

async function loadDatasetsList() {
  const box = $("datasets-list");
  let datasets = [];
  try {
    datasets = await (await authFetch("/api/datasets")).json();
  } catch {
    box.innerHTML = `<p class="muted small">Could not load datasets.</p>`;
    return;
  }
  $("datasets-sub").textContent = `${datasets.length} dataset${datasets.length === 1 ? "" : "s"}`;
  if (!datasets.length) {
    box.innerHTML = `<p class="muted small">No datasets yet — start an experiment to upload one.</p>`;
    return;
  }
  box.innerHTML = datasets
    .map(
      (d) => `
    <button type="button" class="dataset-row" data-run-id="${d.run_id}">
      <span class="dataset-row-main">
        <span class="dataset-row-name">${escapeHtml(d.filename)}</span>
        <span class="muted small">${d.row_count != null ? Number(d.row_count).toLocaleString() + " rows" : "…"} · ${d.column_count ?? "?"} columns</span>
      </span>
      <span class="dataset-row-meta">
        ${d.quality_score != null ? `<span class="chip detected">${Math.round(d.quality_score * 100)}% quality</span>` : ""}
        <span class="status-badge ${d.status}">${d.status.replaceAll("_", " ")}</span>
        <span class="muted small">${relativeTime(d.created_at)}</span>
      </span>
    </button>`
    )
    .join("");
  box.querySelectorAll(".dataset-row").forEach((el) => {
    el.addEventListener("click", () => openDatasetDetail(el.dataset.runId));
  });
}

function showDatasetDetailView() {
  $("intake-view").classList.add("hidden");
  $("run-view").classList.add("hidden");
  $("datasets-view").classList.add("hidden");
  $("dataset-detail-view").classList.remove("hidden");
  setActiveNav("nav-datasets");
  $("share-btn").classList.add("hidden");
  $("export-report-btn").classList.add("hidden");
  $("header-tags").classList.add("hidden");
  $("run-breadcrumb").classList.add("hidden");
}

async function openDatasetDetail(runId) {
  currentDatasetRunId = runId;
  showDatasetDetailView();
  $("column-explorer").classList.add("hidden");
  let run;
  try {
    const res = await authFetch(`/api/runs/${runId}`);
    if (!res.ok) throw new Error("failed to load dataset");
    run = await res.json();
  } catch {
    $("dataset-breadcrumb-name").textContent = "Could not load dataset";
    return;
  }
  $("dataset-breadcrumb-name").textContent = run.filename;

  let summary = { feature_type_counts: {}, ml_readiness_score: 0 };
  try {
    summary = await (await authFetch(`/api/runs/${runId}/dataset-summary`)).json();
  } catch { /* KPI row degrades gracefully to "—" placeholders */ }
  renderDatasetKpis(run, summary);

  previewState = { page: 1, pageSize: 50, sortBy: null, sortDir: "asc", search: "" };
  $("preview-search").value = "";
  await loadPreviewTable(run);

  for (const tab of PROFILING_SUBTABS) $(`ptab-${tab}-panel`).dataset.loaded = "";
  renderColumnSummaryTab(run);
  switchProfilingSubtab("summary");
}

function renderDatasetKpis(run, summary) {
  const profile = run.profile_summary || {};
  const counts = summary.feature_type_counts || {};
  const cards = [
    { icon: "db", tint: "violet", label: "Total Rows", value: profile.row_count != null ? Number(profile.row_count).toLocaleString() : "—" },
    { icon: "grid", tint: "violet", label: "Total Columns", value: String(profile.column_count ?? "—") },
    { icon: "warning", tint: "amber", label: "Missing Values", value: profile.quality ? `${(100 - profile.quality.completeness * 100).toFixed(1)}%` : "—" },
    { icon: "layers", tint: "amber", label: "Duplicate Rows", value: profile.quality ? String(profile.quality.duplicate_row_count) : "—" },
    { icon: "file", tint: "violet", label: "Memory Usage", value: formatBytes(profile.memory_bytes) },
    { icon: "sliders", tint: "green", label: "Numeric Features", value: String(counts.numeric ?? "—") },
    { icon: "grid", tint: "green", label: "Categorical Features", value: String(counts.categorical ?? "—") },
    { icon: "clock", tint: "green", label: "Datetime Features", value: String(counts.datetime ?? "—") },
    { icon: "file", tint: "green", label: "Text Features", value: String(counts.text ?? "—") },
    { icon: "shield", tint: "violet", label: "Data Quality Score", value: profile.quality ? `${Math.round(profile.quality.overall * 100)}%` : "—" },
    { icon: "shield", tint: "violet", label: "ML Readiness Score", value: `${Math.round((summary.ml_readiness_score ?? 0) * 100)}%` },
  ];
  const targetCol = (run.task_spec || {}).target_column;
  if (targetCol) {
    cards.push({ icon: "trophy", tint: "violet", label: "Target Column", value: targetCol });
  }
  $("dataset-kpi-row").innerHTML = cards
    .map(
      (c) => `
      <div class="stat-card">
        <span class="stat-icon ${c.tint}">${ICONS[c.icon]}</span>
        <div class="stat-body">
          <div class="stat-label">${c.label}</div>
          <div class="stat-value">${escapeHtml(c.value)}</div>
        </div>
      </div>`
    )
    .join("");
}

/* ================= profiling sub-tabs ================= */

const PROFILING_SUBTABS = ["summary", "correlations", "missing", "outliers"];

function switchProfilingSubtab(name) {
  for (const tab of PROFILING_SUBTABS) {
    const isActive = tab === name;
    $(`ptab-${tab}-btn`).classList.toggle("active", isActive);
    $(`ptab-${tab}-btn`).setAttribute("aria-selected", String(isActive));
    $(`ptab-${tab}-panel`).classList.toggle("hidden", !isActive);
  }
  if (name === "correlations" && !$("ptab-correlations-panel").dataset.loaded) loadCorrelationsTab();
  if (name === "missing" && !$("ptab-missing-panel").dataset.loaded) loadMissingValuesTab();
  if (name === "outliers" && !$("ptab-outliers-panel").dataset.loaded) loadOutliersTab();
}
for (const tab of PROFILING_SUBTABS) {
  $(`ptab-${tab}-btn`).addEventListener("click", () => switchProfilingSubtab(tab));
}

function renderColumnSummaryTab(run) {
  const columns = run.profile_columns || [];
  let html = "<table class=\"results-table\"><tr><th>Column</th><th>Type</th><th>Missing %</th><th>Unique %</th><th>Cardinality</th></tr>";
  const rowCount = (run.profile_summary || {}).row_count || 1;
  for (const c of columns) {
    html += `<tr>
      <td>${escapeHtml(c.name)}</td>
      <td>${escapeHtml(c.dtype)}</td>
      <td class="num">${((c.null_rate || 0) * 100).toFixed(1)}%</td>
      <td class="num">${(((c.n_unique || 0) / rowCount) * 100).toFixed(1)}%</td>
      <td class="num">${c.n_unique ?? "—"}</td>
    </tr>`;
  }
  html += "</table>";
  $("ptab-summary-panel").innerHTML = html;
  $("ptab-summary-panel").dataset.loaded = "1";
}

/* ================= correlations sub-tab ================= */

function renderCorrelationHeatmap(container, result, options = {}) {
  const { columns, matrix } = result;
  const emptyMessage = options.emptyMessage || "Not enough numeric columns for a correlation matrix.";
  const ariaLabel = options.ariaLabel || "Correlation heatmap";
  if (!columns.length || !matrix.length) {
    container.innerHTML = `<p class="muted small">${escapeHtml(emptyMessage)}</p>`;
    return;
  }
  const cell = 34;
  let html = "";
  if (result.truncated) {
    html += `<p class="muted small">Showing the top ${columns.length} numeric columns by variance (dataset has more).</p>`;
  }
  html += `<div class="heatmap-scroll"><svg width="${cell * (columns.length + 1)}" height="${cell * (columns.length + 1)}" role="img" aria-label="${escapeHtml(ariaLabel)}">`;
  columns.forEach((name, i) => {
    html += `<text x="${cell * (i + 1) + cell / 2}" y="${cell * 0.9}" font-size="9" text-anchor="middle" transform="rotate(-45 ${cell * (i + 1) + cell / 2} ${cell * 0.9})">${escapeHtml(name)}</text>`;
    html += `<text x="${cell * 0.95}" y="${cell * (i + 1) + cell / 2 + 3}" font-size="9" text-anchor="end">${escapeHtml(name)}</text>`;
  });
  matrix.forEach((row, i) => {
    row.forEach((value, j) => {
      const intensity = Math.min(Math.abs(value), 1);
      const color = value >= 0 ? `rgba(124, 58, 237, ${intensity})` : `rgba(220, 38, 38, ${intensity})`;
      html += `<rect x="${cell * (j + 1)}" y="${cell * (i + 1)}" width="${cell}" height="${cell}" fill="${color}"><title>${escapeHtml(columns[i])} × ${escapeHtml(columns[j])}: ${value.toFixed(2)}</title></rect>`;
    });
  });
  html += "</svg></div>";
  container.innerHTML = html;
}

async function loadCorrelationsTab() {
  const panel = $("ptab-correlations-panel");
  panel.innerHTML = `
    <div class="chip-row">
      <label class="field" style="max-width:200px">
        <span class="visually-hidden">Correlation method</span>
        <select id="correlation-method-select">
          <option value="pearson">Pearson</option>
          <option value="spearman">Spearman</option>
          <option value="kendall">Kendall</option>
          <option value="mutual_info">Mutual Information</option>
        </select>
      </label>
    </div>
    <div id="correlation-heatmap-box"><p class="muted small">Loading…</p></div>`;
  panel.dataset.loaded = "1";

  const fetchAndRender = async () => {
    const method = $("correlation-method-select").value;
    let result;
    try {
      const res = await authFetch(`/api/runs/${currentDatasetRunId}/correlations?method=${method}`);
      if (!res.ok) throw new Error("failed to load correlations");
      result = await res.json();
    } catch {
      $("correlation-heatmap-box").innerHTML = `<p class="muted small">Could not load correlations.</p>`;
      return;
    }
    renderCorrelationHeatmap($("correlation-heatmap-box"), result);
  };
  $("correlation-method-select").addEventListener("change", fetchAndRender);
  await fetchAndRender();
}

/* ================= missing values sub-tab ================= */

async function loadMissingValuesTab() {
  const panel = $("ptab-missing-panel");
  panel.innerHTML = `<p class="muted small">Loading…</p>`;
  panel.dataset.loaded = "1";

  let result;
  try {
    const res = await authFetch(`/api/runs/${currentDatasetRunId}/missing-values`);
    if (!res.ok) throw new Error("failed to load missing-value analysis");
    result = await res.json();
  } catch {
    panel.innerHTML = `<p class="muted small">Could not load missing-value analysis.</p>`;
    return;
  }

  const rows = result.per_column.filter((r) => r.null_count > 0).sort((a, b) => b.null_rate - a.null_rate);
  let html = `<div class="quality-bars">${rows
    .map(
      (r) => `
    <div class="quality-row">
      <span class="quality-name">${escapeHtml(r.column)}</span>
      <span class="fi-track"><span class="fi-fill quality-fill" style="width:${(r.null_rate * 100).toFixed(1)}%;background:var(--accent-warning)"></span></span>
      <span class="quality-value mono">${(r.null_rate * 100).toFixed(1)}%</span>
    </div>`
    )
    .join("")}</div>`;
  if (!rows.length) html = `<p class="muted small">No missing values in this dataset.</p>`;

  html += `<h4 class="missing-corr-title">Which columns tend to be missing together</h4><div id="missing-corr-box"></div>`;
  panel.innerHTML = html;
  renderCorrelationHeatmap($("missing-corr-box"), { columns: result.missing_correlation.columns, matrix: result.missing_correlation.matrix }, {
    emptyMessage: "Fewer than two columns have missing values, so there's nothing to correlate.",
    ariaLabel: "Missing-value correlation heatmap",
  });
}

/* ================= outliers sub-tab ================= */

async function loadOutliersTab() {
  const panel = $("ptab-outliers-panel");
  panel.innerHTML = `
    <div class="chip-row">
      <label class="field" style="max-width:200px">
        <span class="visually-hidden">Outlier detection method</span>
        <select id="outlier-method-select">
          <option value="iqr">IQR</option>
          <option value="zscore">Z-score</option>
          <option value="isolation_forest">Isolation Forest</option>
          <option value="lof">Local Outlier Factor</option>
        </select>
      </label>
    </div>
    <div id="outlier-result-box"><p class="muted small">Loading…</p></div>`;
  panel.dataset.loaded = "1";

  const fetchAndRender = async () => {
    const method = $("outlier-method-select").value;
    const box = $("outlier-result-box");
    box.innerHTML = `<p class="muted small">Detecting…</p>`;
    let result;
    try {
      const res = await authFetch(`/api/runs/${currentDatasetRunId}/outliers?method=${method}`);
      if (!res.ok) throw new Error("failed to run outlier detection");
      result = await res.json();
    } catch {
      box.innerHTML = `<p class="muted small">Could not run outlier detection.</p>`;
      return;
    }
    box.innerHTML = `
      <div class="chips">
        <span class="chip flagged">${result.outlier_count} outlier row(s) detected</span>
        ${(result.affected_columns || []).map((c) => `<span class="chip detected">${escapeHtml(c)}</span>`).join("")}
      </div>
      ${result.example_row_indices && result.example_row_indices.length ? `<p class="muted small">Example row indices: ${result.example_row_indices.join(", ")}</p>` : ""}`;
  };
  $("outlier-method-select").addEventListener("change", fetchAndRender);
  await fetchAndRender();
}

/* ================= interactive preview table ================= */

function classifyPreviewType(dtype) {
  const d = String(dtype || "").toLowerCase();
  if (d.includes("bool")) return "boolean";
  if (d.includes("datetime") || d.includes("date")) return "datetime";
  if (d.includes("int") || d.includes("float")) return "numeric";
  return "categorical";
}

async function loadPreviewTable(run) {
  previewColumns = run.profile_columns || [];
  const layoutKey = `automl-preview-layout-${currentDatasetRunId}`;
  const savedLayout = JSON.parse(localStorage.getItem(layoutKey) || "{}");
  const hiddenColumns = new Set(savedLayout.hiddenColumns || []);

  $("preview-colvis").innerHTML = previewColumns
    .map(
      (c) => `<label class="colvis-item"><input type="checkbox" data-col="${escapeHtml(c.name)}" ${hiddenColumns.has(c.name) ? "" : "checked"}/> ${escapeHtml(c.name)}</label>`
    )
    .join("");
  $("preview-colvis").querySelectorAll("input[type=checkbox]").forEach((input) => {
    input.addEventListener("change", () => {
      const hidden = new Set(
        Array.from($("preview-colvis").querySelectorAll("input:not(:checked)")).map((el) => el.dataset.col)
      );
      localStorage.setItem(layoutKey, JSON.stringify({ hiddenColumns: [...hidden] }));
      fetchAndRenderPreviewPage();
    });
  });

  await fetchAndRenderPreviewPage();
}

async function fetchAndRenderPreviewPage() {
  const params = new URLSearchParams({
    page: String(previewState.page),
    page_size: String(previewState.pageSize),
    sort_dir: previewState.sortDir,
  });
  if (previewState.sortBy) params.set("sort_by", previewState.sortBy);
  if (previewState.search) params.set("search", previewState.search);

  let data;
  try {
    const res = await authFetch(`/api/runs/${currentDatasetRunId}/preview?${params}`);
    if (!res.ok) throw new Error("failed to load preview");
    data = await res.json();
  } catch {
    $("preview-table").innerHTML = `<tr><td>Could not load preview.</td></tr>`;
    return;
  }
  renderPreviewTable(data);
}

function renderPreviewTable(data) {
  const layoutKey = `automl-preview-layout-${currentDatasetRunId}`;
  const savedLayout = JSON.parse(localStorage.getItem(layoutKey) || "{}");
  const hiddenColumns = new Set(savedLayout.hiddenColumns || []);
  const visibleColumns = previewColumns.filter((c) => !hiddenColumns.has(c.name));
  const piiSet = new Set(data.pii_columns || []);
  const duplicateSet = new Set(data.duplicate_row_indices || []);

  const numericRanges = {};
  for (const col of visibleColumns) {
    if (classifyPreviewType(col.dtype) !== "numeric") continue;
    const values = data.rows.map((r) => r[col.name]).filter((v) => v != null);
    numericRanges[col.name] = values.length ? { min: Math.min(...values), max: Math.max(...values) } : null;
  }

  let html = "<tr>" + visibleColumns
    .map(
      (c) => `<th data-col="${escapeHtml(c.name)}" class="sortable">
        ${escapeHtml(c.name)} <span class="col-type-badge">${classifyPreviewType(c.dtype)}</span>
        ${piiSet.has(c.name) ? `<span class="chip flagged" title="PII column">PII</span>` : ""}
        <button type="button" class="col-profile-btn" data-col="${escapeHtml(c.name)}" title="Profile this column">${ICONS.search}</button>
      </th>`
    )
    .join("") + "</tr>";

  for (const row of data.rows) {
    const isDup = duplicateSet.has(row._row_index);
    html += `<tr class="${isDup ? "preview-row-duplicate" : ""}">`;
    for (const col of visibleColumns) {
      const value = row[col.name];
      const type = classifyPreviewType(col.dtype);
      if (value == null) {
        html += `<td class="preview-cell-missing">—</td>`;
      } else if (type === "numeric") {
        const range = numericRanges[col.name];
        const pct = range && range.max > range.min ? (value - range.min) / (range.max - range.min) : 0;
        html += `<td class="num preview-cell-numeric" style="background:linear-gradient(90deg, var(--accent-primary-soft) ${(pct * 100).toFixed(0)}%, transparent ${(pct * 100).toFixed(0)}%)">${escapeHtml(String(value))}</td>`;
      } else if (type === "boolean") {
        html += `<td><span class="chip ${value ? "detected" : "flagged"}">${escapeHtml(String(value))}</span></td>`;
      } else if (type === "categorical") {
        html += `<td><span class="chip detected">${escapeHtml(String(value))}</span></td>`;
      } else {
        html += `<td title="${escapeHtml(String(value))}">${escapeHtml(String(value))}</td>`;
      }
    }
    html += "</tr>";
  }
  $("preview-table").innerHTML = html;

  $("preview-table").querySelectorAll("th.sortable").forEach((th) => {
    th.addEventListener("click", (e) => {
      if (e.target.closest(".col-profile-btn")) return;
      const col = th.dataset.col;
      previewState.sortDir = previewState.sortBy === col && previewState.sortDir === "asc" ? "desc" : "asc";
      previewState.sortBy = col;
      fetchAndRenderPreviewPage();
    });
  });
  $("preview-table").querySelectorAll(".col-profile-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      openColumnExplorer(btn.dataset.col);
    });
  });

  const totalPages = Math.max(1, Math.ceil(data.total_count / previewState.pageSize));
  $("preview-pager").innerHTML = `
    <button type="button" class="btn ghost" id="preview-prev" ${previewState.page <= 1 ? "disabled" : ""}>Prev</button>
    <span class="muted small">Page ${data.page} of ${totalPages} · ${data.total_count.toLocaleString()} rows</span>
    <button type="button" class="btn ghost" id="preview-next" ${previewState.page >= totalPages ? "disabled" : ""}>Next</button>`;
  $("preview-prev").addEventListener("click", () => { previewState.page -= 1; fetchAndRenderPreviewPage(); });
  $("preview-next").addEventListener("click", () => { previewState.page += 1; fetchAndRenderPreviewPage(); });
}

async function openColumnExplorer(columnName) {
  $("column-explorer-name").textContent = columnName;
  $("column-explorer").classList.remove("hidden");
  $("column-explorer-body").innerHTML = `<p class="muted small">Loading…</p>`;

  let detail;
  try {
    const res = await authFetch(`/api/runs/${currentDatasetRunId}/columns/${encodeURIComponent(columnName)}`);
    if (!res.ok) throw new Error("failed to load column detail");
    detail = await res.json();
  } catch {
    $("column-explorer-body").innerHTML = `<p class="muted small">Could not load column details.</p>`;
    return;
  }

  let html = `<p class="muted small mono">${escapeHtml(detail.dtype)}</p>`;

  if (detail.is_numeric) {
    // A numeric column that is entirely NaN reports is_numeric: true but omits
    // histogram/stats altogether (see src/profiling/preview.py) — guard rather
    // than assume they're always present alongside is_numeric.
    if (detail.histogram && detail.stats) {
      const hist = detail.histogram;
      const maxCount = Math.max(...hist.counts, 1);
      html += `<div class="explorer-histogram">${hist.counts
        .map((c) => `<span class="explorer-bar" style="height:${((c / maxCount) * 100).toFixed(0)}%" title="${c}"></span>`)
        .join("")}</div>`;
      const s = detail.stats;
      html += `<div class="explorer-stats">
        <div>Mean <strong>${s.mean.toFixed(2)}</strong></div>
        <div>Median <strong>${s.median.toFixed(2)}</strong></div>
        <div>Std Dev <strong>${s.std.toFixed(2)}</strong></div>
        <div>Min <strong>${s.min.toFixed(2)}</strong></div>
        <div>Max <strong>${s.max.toFixed(2)}</strong></div>
        <div>P25 <strong>${s.p25.toFixed(2)}</strong></div>
        <div>P75 <strong>${s.p75.toFixed(2)}</strong></div>
        <div>Skew <strong>${s.skew.toFixed(2)}</strong></div>
      </div>`;
      if (detail.correlation_with_target != null) {
        html += `<p class="muted small">Correlation with target: <strong>${detail.correlation_with_target.toFixed(3)}</strong></p>`;
      }
    } else {
      html += `<p class="muted small">This column has no non-missing values, so no distribution can be shown.</p>`;
    }
  } else {
    html += `<div class="field"><span>Top values</span><ul class="callout-list">${Object.entries(detail.top_values || {})
      .map(([k, v]) => `<li>${escapeHtml(k)} <span class="muted small">(${v})</span></li>`)
      .join("")}</ul></div>`;
  }

  const insights = detail.ml_insights || { analyzed: false };
  if (insights.analyzed) {
    html += `<div class="field"><span>ML Insights</span><ul class="callout-list">${(insights.recommended_steps || [])
      .map((s) => `<li><span class="step-op">${escapeHtml(s.op)}</span><span class="step-rationale">${escapeHtml(s.rationale || "")}</span></li>`)
      .join("")}${(insights.leakage_flags || [])
      .map((f) => `<li>${ICONS.warning}<span>${escapeHtml(f.reason)}</span></li>`)
      .join("")}</ul></div>`;
  } else {
    html += `<p class="muted small">Run further into the pipeline to see ML insights for this column.</p>`;
  }

  $("column-explorer-body").innerHTML = html;
}

$("column-explorer-close").addEventListener("click", () => $("column-explorer").classList.add("hidden"));

let previewSearchDebounce = null;
$("preview-search").addEventListener("input", (e) => {
  clearTimeout(previewSearchDebounce);
  previewSearchDebounce = setTimeout(() => {
    previewState.search = e.target.value.trim();
    previewState.page = 1;
    fetchAndRenderPreviewPage();
  }, 300);
});

$("dataset-breadcrumb-back").addEventListener("click", showDatasetsView);

/* ================= recent runs (sidebar) ================= */

async function loadRecentRuns() {
  const box = $("nav-runs");
  let runs = [];
  try {
    runs = await (await authFetch("/api/runs")).json();
  } catch {
    box.innerHTML = `<span class="nav-runs-empty">No runs yet</span>`;
    renderHome([]);
    return;
  }
  if (!runs.length) {
    box.innerHTML = `<span class="nav-runs-empty">No runs yet</span>`;
  } else {
    box.innerHTML = "";
    for (const run of runs.slice(0, 6)) {
      const el = document.createElement("button");
      el.type = "button";
      el.className = "nav-run";
      el.title = `${run.filename} — ${run.description || ""}`;
      el.innerHTML = `<span class="dot ${run.status}"></span>${escapeHtml(run.filename)}`;
      el.onclick = () => openRun(run.run_id);
      box.appendChild(el);
    }
  }
  renderHome(runs);
}

/* ================= home view ================= */

const ACTIVE_STATUSES = ["profiling", "running", "awaiting_confirmation", "awaiting_feature_approval"];
/* cross-run "best" is only meaningful for higher-is-better metrics */
const HIGHER_IS_BETTER = new Set(["f1", "accuracy", "roc_auc", "r2"]);
let homePipelineRunId = null;

function renderHome(runs) {
  const active = runs.filter((r) => ACTIVE_STATUSES.includes(r.status));
  let bestRun = null;
  for (const r of runs) {
    if (r.status !== "completed" || r.best_score == null || !HIGHER_IS_BETTER.has(r.metric)) continue;
    if (!bestRun || r.best_score > bestRun.best_score) bestRun = r;
  }

  const statsBox = $("home-stats");
  statsBox.classList.toggle("hidden", !runs.length);
  if (runs.length) {
    statsBox.innerHTML = `
      <div class="home-stat"><div class="home-stat-value">${runs.length}</div><div class="home-stat-label">Total experiments</div></div>
      <div class="home-stat"><div class="home-stat-value">${bestRun ? bestRun.best_score.toFixed(3) : "—"}</div><div class="home-stat-label">${bestRun ? `Best ${escapeHtml(bestRun.metric)} score` : "Best score"}</div></div>
      <div class="home-stat"><div class="home-stat-value">${active.length}</div><div class="home-stat-label">Active run${active.length === 1 ? "" : "s"}</div></div>`;
  }

  $("home-band").classList.toggle("hidden", !runs.length);
  if (runs.length) {
    $("home-projects-list").innerHTML = runs
      .slice(0, 5)
      .map(
        (r) => `
      <button type="button" class="home-project" data-run-id="${r.run_id}">
        <span class="home-project-main">
          <span class="home-project-name">${escapeHtml(r.filename)}</span>
          <span class="home-project-desc">${escapeHtml(r.description || "")}</span>
        </span>
        <span class="home-project-meta">
          ${r.source_run_id ? `<span class="muted small" title="New experiment on the dataset from run ${escapeHtml(r.source_run_id)}">re-run</span>` : ""}
          ${r.best_score != null ? `<span class="home-project-score mono">${escapeHtml(r.metric)}: ${r.best_score.toFixed(3)}</span>` : ""}
          <span class="status-badge ${r.status}">${r.status.replaceAll("_", " ")}</span>
          <span class="home-project-time">${relativeTime(r.created_at)}</span>
        </span>
      </button>`
      )
      .join("");
    $("home-projects-list").querySelectorAll(".home-project").forEach((el) => {
      el.addEventListener("click", () => openRun(el.dataset.runId));
    });
  }
  renderHomePipeline(active[0] || null);
}

async function renderHomePipeline(activeRun) {
  const card = $("home-pipeline");
  if (!activeRun) {
    card.classList.add("hidden");
    homePipelineRunId = null;
    return;
  }
  homePipelineRunId = activeRun.run_id;
  let run;
  try {
    run = await (await authFetch(`/api/runs/${activeRun.run_id}`)).json();
  } catch {
    return;
  }
  if (homePipelineRunId !== activeRun.run_id) return; // superseded by a newer refresh
  card.classList.remove("hidden");
  $("home-pipeline-sub").textContent = run.filename;

  const done = new Set(run.stages_done || []);
  const durations = {};
  for (const rec of run.stage_timeline || []) durations[rec.node] = rec.duration_seconds;
  let activeAssigned = false;
  $("home-pipeline-rail").innerHTML = STAGES.map((stage) => {
    const stageDone = stage.node === "poll_training" ? done.has("evaluate") : done.has(stage.node);
    let cls = "pending";
    if (stageDone) cls = "done";
    else if (!activeAssigned) {
      cls = "active";
      activeAssigned = true;
    }
    const duration = durations[stage.node];
    return `<li class="mini-stage ${cls}">
      <span class="mini-stage-dot">${cls === "done" ? ICONS.check : ""}</span>
      <span class="mini-stage-label">${stage.label}</span>
      <span class="mini-stage-time mono">${cls === "done" && duration != null ? formatDuration(duration) : cls === "active" ? "running" : ""}</span>
    </li>`;
  }).join("");
}

/* ================= new run form ================= */

const dropzone = $("dropzone");
const fileInput = $("file-input");

dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) setFile(fileInput.files[0]); });

function setFile(file) {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    alert("Please choose a .csv file.");
    return;
  }
  selectedFile = file;
  dropzone.classList.add("has-file");
  $("dropzone-label").innerHTML = `<strong>${escapeHtml(file.name)}</strong> (${(file.size / 1024).toFixed(1)} KB)`;
  estimateDataset(file);
  updateSubmit();
}

/* honest client-side size estimate before upload commits (reads only the
   first 64KB, never the whole file) */
async function estimateDataset(file) {
  const row = $("estimate-row");
  row.innerHTML = `<span class="chip detected">Reading file…</span>`;
  try {
    const head = await file.slice(0, 65536).text();
    const lines = head.split(/\r\n|\n/).filter((l) => l.length > 0);
    const columnCount = (lines[0] || "").split(",").length;
    const bytesPerRow = lines.length > 1 ? head.length / lines.length : head.length;
    const estimatedRows = bytesPerRow > 0 ? Math.round(file.size / bytesPerRow) : lines.length;
    row.innerHTML = `
      <span class="chip detected">${ICONS.check} ~${estimatedRows.toLocaleString()} rows (estimated)</span>
      <span class="chip detected">${ICONS.check} ${columnCount} columns</span>
      <span class="chip detected">${ICONS.check} runtime: usually under a minute locally</span>`;
  } catch {
    row.innerHTML = "";
  }
}

$("description").addEventListener("input", updateSubmit);
function updateSubmit() {
  $("submit-btn").disabled = !(selectedFile && $("description").value.trim());
}

$("new-run-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData();
  form.append("file", selectedFile);
  form.append("description", $("description").value.trim());
  $("submit-btn").disabled = true;
  $("submit-btn").textContent = "Uploading…";
  try {
    const res = await authFetch("/api/runs", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const { run_id } = await res.json();
    openRun(run_id);
  } catch (err) {
    alert("Failed to start run: " + err.message);
  } finally {
    $("submit-btn").textContent = "Run pipeline";
    updateSubmit();
  }
});

/* ================= re-run dataset (multi-experiment) ================= */

$("rerun-btn").addEventListener("click", () => {
  $("rerun-card").classList.toggle("hidden");
  if (!$("rerun-card").classList.contains("hidden")) $("rerun-description").focus();
});

$("rerun-close-btn").addEventListener("click", () => {
  $("rerun-card").classList.add("hidden");
});

$("rerun-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const description = $("rerun-description").value.trim();
  if (!description || !currentRunId) return;
  const btn = $("rerun-submit-btn");
  btn.disabled = true;
  btn.textContent = "Starting…";
  try {
    const res = await authFetch(`/api/runs/${currentRunId}/experiments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ description }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const { run_id } = await res.json();
    $("rerun-card").classList.add("hidden");
    $("rerun-description").value = "";
    openRun(run_id);
  } catch (err) {
    alert("Failed to start experiment: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Start experiment";
  }
});

$("cancel-btn").addEventListener("click", async () => {
  if (!confirm("Cancel this run? Work completed so far is kept, but no further stages will run.")) return;
  try {
    await authFetch(`/api/runs/${currentRunId}/cancel`, { method: "POST" });
  } catch { /* poll() reflects the outcome regardless */ }
});

$("run-breadcrumb-datasets").addEventListener("click", () => {
  if (lastRun) openDatasetDetail(lastRun.source_run_id || lastRun.run_id);
});

/* ================= run view + polling ================= */

function openRun(runId) {
  currentRunId = runId;
  currentRunStatus = null;
  lastRun = null;
  showRunView();
  history.replaceState(null, "", `?run=${encodeURIComponent(runId)}`);
  $("rerun-card").classList.add("hidden");
  $("trace-details").classList.add("hidden");
  $("trace-body").textContent = "";
  predictFormLoadedFor = null;
  explainabilityLoadedFor = null;
  $("explainability-narrative").textContent = "";
  $("explainability-plots").innerHTML = "";
  $("explainability-empty").classList.add("hidden");
  $("predict-result").classList.add("hidden");
  chatPendingQuestion = null;
  $("chat-input").value = "";
  $("chat-error").classList.add("hidden");
  switchRunTab("overview");
  $("tab-test-panel").classList.add("hidden");
  stopPolling();
  poll();
  pollTimer = setInterval(poll, 1500);
  loadRecentRuns();
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

async function poll() {
  if (!currentRunId) return;
  let run;
  try {
    const res = await authFetch(`/api/runs/${currentRunId}`);
    if (!res.ok) {
      // A shared/bookmarked ?run= link can point at a run that no longer
      // exists; without lastRun there's nothing to keep showing, so fall
      // back to the intake view instead of a permanently blank dashboard.
      if (!lastRun) {
        stopPolling();
        showIntakeView();
      }
      return;
    }
    run = await res.json();
  } catch { return; }

  render(run);
  if (["completed", "failed", "cancelled"].includes(run.status)) {
    stopPolling();
    loadRecentRuns();
  }
}

function render(run) {
  lastRun = run;
  currentRunStatus = run.status;

  $("header-eyebrow").textContent = run.source_run_id
    ? `Run ${run.run_id} · re-run of ${run.source_run_id}`
    : `Run ${run.run_id}`;
  $("run-title").textContent = run.filename;
  $("run-desc").textContent = run.description || "";
  $("rerun-btn").classList.remove("hidden");
  $("share-btn").classList.remove("hidden");
  $("export-report-btn").classList.toggle("hidden", !run.report);
  renderHeaderTags(run);
  renderBreadcrumb(run);

  const badge = $("status-badge");
  badge.classList.remove("hidden");
  badge.className = `status-badge ${run.status}`;
  $("status-badge-text").textContent = run.status.replaceAll("_", " ");

  $("cancel-btn").classList.toggle(
    "hidden",
    !["profiling", "running", "awaiting_confirmation", "awaiting_feature_approval"].includes(run.status)
  );

  renderChampionBanner(run);
  renderJourneyCondensed(run);
  renderLeaderboardCondensed(run);
  renderModelRationale(run);
  renderPipelineActions(run);
  renderStatCards(run);
  renderStageTracker(run);
  renderReasoningRail(run);
  renderLogsTab(run);
  renderTrainProgress(run);
  renderConfirm(run);
  renderLeakage(run);
  renderFeatureApproval(run);
  renderDatasetSummary(run);
  renderClassDistribution(run);
  renderQuality(run);
  renderInsights(run);
  renderResults(run);
  renderExperimentsTab(run);
  renderFeatureImportance(run);
  renderActivity(run);
  renderReport(run);
  renderChat(run);
  renderCaveats(run);
  renderErrors(run);
}

/* ================= champion banner ================= */

function renderChampionBanner(run) {
  const best = run.best_model || {};
  const card = $("champion-banner");
  if (!best.candidate_name) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  $("champion-banner-name").textContent = best.candidate_name;

  const metric = (run.task_spec || {}).metric;
  const results = run.training_results || [];
  // NOTE: deliberately not `(best.tuning || {}).lower_is_better` (as the task
  // brief's snippet used) — src/training/dispatch.py hardcodes
  // tuning.lower_is_better to False whenever tuning was skipped/disabled for
  // the winning candidate (the common case), even for rmse/mae. That would
  // invert the delta sign and pick the wrong runner-up for regression runs.
  // Mirror the same rule src/graph/nodes.py:338 and
  // src/insights/auto_insights.py:140 use to pick `best` in the first place,
  // so ranking here stays consistent with how "best" was actually chosen.
  const lowerIsBetter = metric === "rmse" || metric === "mae";
  const sorted = [...results]
    .filter((r) => r.status === "succeeded" && metric && r.metrics && metric in r.metrics)
    .sort((a, b) => (lowerIsBetter ? a.metrics[metric] - b.metrics[metric] : b.metrics[metric] - a.metrics[metric]));
  const runnerUp = sorted.find((r) => r.run_id !== best.run_id);

  const stats = [];
  if (metric && best.metrics && metric in best.metrics) {
    const bestScore = Number(best.metrics[metric]);
    let deltaText = "";
    if (runnerUp) {
      const delta = lowerIsBetter ? runnerUp.metrics[metric] - bestScore : bestScore - runnerUp.metrics[metric];
      deltaText = ` <span class="champion-delta">${delta >= 0 ? "+" : ""}${delta.toFixed(3)} vs next best</span>`;
    }
    stats.push(`<div><span class="champion-stat-label">${escapeHtml(metric.toUpperCase())}</span><strong>${bestScore.toFixed(3)}</strong>${deltaText}</div>`);
  }
  if (best.duration_seconds != null) {
    stats.push(`<div><span class="champion-stat-label">Training Time</span><strong>${formatDuration(best.duration_seconds)}</strong></div>`);
  }
  if (best.cv_folds) {
    stats.push(`<div><span class="champion-stat-label">Cross Validation</span><strong>${best.cv_folds} Fold${best.resampling_applied ? ` + ${best.resampling_applied.replaceAll("_", " ")}` : ""}</strong></div>`);
  }
  $("champion-banner-stats").innerHTML = stats.join("");

  $("champion-download-btn").classList.toggle("hidden", !run.report);
}

$("champion-compare-btn").addEventListener("click", () => switchRunTab("experiments"));
$("champion-download-btn").addEventListener("click", (e) => {
  e.preventDefault();
  $("export-report-btn").click();
});

/* ================= journey condensed ================= */

function renderJourneyCondensed(run) {
  const done = new Set(run.stages_done || []);
  const durations = {};
  for (const rec of run.stage_timeline || []) durations[rec.node] = rec.duration_seconds;
  const best = run.best_model || {};

  $("journey-condensed").innerHTML = JOURNEY_GROUPS.map((group, i) => {
    const isDone = group.nodes.every((n) => done.has(n === "poll_training" ? "evaluate" : n));
    const lastNode = group.nodes[group.nodes.length - 1];
    const duration = durations[lastNode === "poll_training" ? "evaluate" : lastNode];
    const sub = group.label === "Champion Selected" && best.candidate_name ? best.candidate_name : "";
    return `
      <li class="${isDone ? "done" : "pending"}">
        <span class="journey-num">${isDone ? ICONS.check : i + 1}</span>
        <span class="journey-label">${i + 1}. ${escapeHtml(group.label)}</span>
        ${sub ? `<span class="muted small">${escapeHtml(sub)}</span>` : ""}
        ${isDone && duration != null ? `<span class="journey-time">${formatDuration(duration)}</span>` : ""}
      </li>`;
  }).join("");
}

$("journey-view-pipeline-btn").addEventListener("click", () => switchRunTab("experiments"));

$("nextstep-compare-btn").addEventListener("click", () => switchRunTab("experiments"));
$("nextstep-artifacts-btn").addEventListener("click", () => switchRunTab("artifacts"));
$("nextstep-share-btn").addEventListener("click", () => $("share-btn").click());

/* ================= leaderboard condensed ================= */

// Keyed by keyword found in the LLM-assigned candidate_name (case-insensitive),
// checked in order — first match wins. Falls back to 3 stars when nothing
// matches. This is a fuzzy, best-effort heuristic (candidate_name is free text,
// not an enum), not a per-run measurement.
const EXPLAINABILITY_KEYWORD_STARS = [
  { keywords: ["logistic", "linear", "ridge", "lasso", "elastic net", "elasticnet"], stars: 5 },
  { keywords: ["decision tree", "k-nearest", "knn", "naive bayes"], stars: 4 },
  { keywords: ["random forest", "extra trees", "extratrees"], stars: 3 },
  { keywords: ["gradient boost", "xgboost", "xgb", "lightgbm", "lgbm", "catboost"], stars: 2 },
];

function explainabilityStars(candidateName) {
  const name = (candidateName || "").toLowerCase();
  const match = EXPLAINABILITY_KEYWORD_STARS.find((entry) => entry.keywords.some((k) => name.includes(k)));
  const n = match ? match.stars : 3;
  return "★".repeat(n) + "☆".repeat(5 - n);
}

function renderLeaderboardCondensed(run) {
  const results = run.training_results || [];
  if (!results.length) { $("leaderboard-condensed-table").innerHTML = ""; $("leaderboard-view-all-btn").classList.add("hidden"); return; }

  const metric = (run.task_spec || {}).metric;
  const bestId = (run.best_model || {}).run_id;
  const metricNames = [...new Set(results.flatMap((r) => Object.keys(r.metrics || {})))];
  const primary = metric && metricNames.includes(metric) ? metric : metricNames[0];
  const secondary = metricNames.find((m) => m !== primary);

  const showAll = results.length <= 6;
  const champion = results.find((r) => r.run_id === bestId);
  const lowerIsBetter = primary === "rmse" || primary === "mae";
  const others = results
    .filter((r) => r.run_id !== bestId)
    .sort((a, b) => {
      const aHas = a.metrics && primary in a.metrics, bHas = b.metrics && primary in b.metrics;
      if (!aHas && !bHas) return 0;
      if (!aHas) return 1;
      if (!bHas) return -1;
      return lowerIsBetter ? a.metrics[primary] - b.metrics[primary] : b.metrics[primary] - a.metrics[primary];
    });
  const shown = showAll ? [champion, ...others].filter(Boolean) : [champion, ...others.slice(0, 5)].filter(Boolean);

  $("leaderboard-condensed-sub").textContent = primary ? `ranked by ${primary}` : "";
  $("leaderboard-view-all-btn").classList.toggle("hidden", showAll);

  let html = `<tr><th>Model</th>${primary ? `<th>${escapeHtml(primary)}</th>` : ""}${secondary ? `<th>${escapeHtml(secondary)}</th>` : ""}<th>Training Time</th><th>Explainability</th><th>Champion</th></tr>`;
  for (const r of shown) {
    const isBest = r.run_id === bestId;
    html += `<tr class="${isBest ? "best" : ""}">
      <td>${escapeHtml(r.candidate_name)}</td>
      ${primary ? `<td class="num">${r.metrics && primary in r.metrics ? Number(r.metrics[primary]).toFixed(3) : "—"}</td>` : ""}
      ${secondary ? `<td class="num">${r.metrics && secondary in r.metrics ? Number(r.metrics[secondary]).toFixed(3) : "—"}</td>` : ""}
      <td>${r.duration_seconds != null ? formatDuration(r.duration_seconds) : "—"}</td>
      <td class="stars" title="Approximate rating based on the model's name, not a per-run measurement">${explainabilityStars(r.candidate_name)}</td>
      <td>${isBest ? `<span class="winner-tag">★ CHAMPION</span>` : ""}</td>
    </tr>`;
  }
  $("leaderboard-condensed-table").innerHTML = html;
}

$("leaderboard-view-all-btn").addEventListener("click", () => switchRunTab("experiments"));

/* ================= model rationale (why this / why not others) ================= */

function renderModelRationale(run) {
  const best = run.best_model || {};
  const results = run.training_results || [];
  if (!best.candidate_name || results.length < 2) {
    $("why-this-model-list").innerHTML = "";
    $("why-others-table").innerHTML = "";
    return;
  }
  const metric = (run.task_spec || {}).metric;
  // NOTE: not `(best.tuning || {}).lower_is_better` — src/training/dispatch.py
  // hardcodes tuning.lower_is_better to False whenever tuning was skipped for
  // the winning candidate, even for rmse/mae. Mirror the same rule used in
  // renderChampionBanner (and src/graph/nodes.py:338 /
  // src/insights/auto_insights.py:140) so deltas here stay consistent with how
  // "best" was actually chosen.
  const lowerIsBetter = metric === "rmse" || metric === "mae";
  const bestScore = metric && best.metrics ? Number(best.metrics[metric]) : null;

  const whyThis = [];
  if (bestScore != null) whyThis.push(`Highest ${escapeHtml(metric)} (${bestScore.toFixed(3)}) among all candidates`);
  const bestCv = best.cv_metrics && metric && best.cv_metrics[metric];
  if (bestCv) whyThis.push(`Stable performance across folds (CV std ${bestCv.std.toFixed(3)})`);
  const fastestId = [...results].sort((a, b) => (a.duration_seconds ?? Infinity) - (b.duration_seconds ?? Infinity))[0]?.run_id;
  if (fastestId === best.run_id) whyThis.push("Fastest training time among all candidates");
  if ((best.tuning || {}).enabled) whyThis.push(`Hyperparameters tuned over ${best.tuning.trials_done} trial(s)`);
  $("why-this-model-list").innerHTML = whyThis.map((t) => `<li>${ICONS.check}<span>${t}</span></li>`).join("") ||
    `<li class="muted small">No further rationale available for this run.</li>`;

  const others = results.filter((r) => r.run_id !== best.run_id && r.status === "succeeded");
  let html = `<tr><th>Model</th><th>Reason</th><th>Impact</th></tr>`;
  for (const r of others) {
    const delta = metric && r.metrics && bestScore != null && metric in r.metrics
      ? (lowerIsBetter ? r.metrics[metric] - bestScore : bestScore - r.metrics[metric])
      : null;
    const durRatio = best.duration_seconds && r.duration_seconds ? r.duration_seconds / best.duration_seconds : null;
    let impact = "Marginal gain";
    if (durRatio != null && durRatio > 2) impact = "High Cost";
    else if (durRatio != null && durRatio > 1.3) impact = "Medium Cost";
    // delta >= 0 always means "worse than champion", but whether that means
    // a higher or lower raw metric value depends on lowerIsBetter (rmse/mae
    // vs. accuracy/f1/etc.) — don't conflate the two.
    const reason = delta != null
      ? `${(lowerIsBetter === (delta >= 0)) ? "Higher" : "Lower"} ${escapeHtml(metric)} (${Math.abs(delta).toFixed(3)} difference)${durRatio != null && durRatio > 1.3 ? " and slower training" : ""}`
      : "Did not outperform the champion";
    html += `<tr><td>${escapeHtml(r.candidate_name)}</td><td>${reason}</td><td><span class="chip flagged">${impact}</span></td></tr>`;
  }
  $("why-others-table").innerHTML = html;
}

const FEATURE_OP_LABELS = {
  impute: "Imputed missing values",
  encode: "Encoded categorical values",
  scale: "Standardized numerical features",
  bin: "Binned continuous values",
  datetime_decompose: "Decomposed datetime columns",
  drop: "Removed columns",
  custom_code: "Applied a custom transformation",
};

function renderPipelineActions(run) {
  const steps = ((run.feature_plan || {}).steps) || [];
  const card = $("pipeline-actions-card");
  const items = [];

  for (const step of steps) {
    const label = FEATURE_OP_LABELS[step.op] || step.op;
    const cols = (step.columns || []).slice(0, 3).join(", ") + (step.columns && step.columns.length > 3 ? ", …" : "");
    items.push(`<li>${ICONS.check}<span><strong>${escapeHtml(label)}</strong>${cols ? ` — ${escapeHtml(cols)}` : ""}</span></li>`);
  }
  const resamplingApplied = (run.training_results || []).find((r) => r.resampling_applied)?.resampling_applied;
  if (resamplingApplied) {
    items.push(`<li>${ICONS.check}<span><strong>Applied ${escapeHtml(resamplingApplied.replaceAll("_", " "))}</strong> to correct class imbalance</span></li>`);
  }
  const fs = run.feature_selection;
  if (fs && fs.n_features_selected != null) {
    items.push(`<li>${ICONS.check}<span><strong>Feature selection</strong> kept ${fs.n_features_selected} of ${fs.n_features_total} features</span></li>`);
  }

  if (!items.length) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  $("pipeline-actions-list").innerHTML = items.join("");
}

/* ================= stat cards ================= */

function renderStatCards(run) {
  const summary = run.profile_summary || {};
  const best = run.best_model || {};
  const spec = run.task_spec || {};
  const results = run.training_results || [];
  const succeeded = results.filter((r) => r.status === "succeeded").length;
  const terminal = ["completed", "failed", "cancelled"].includes(run.status);
  const metric = spec.metric;
  const bestScore = metric && best.metrics && metric in best.metrics ? Number(best.metrics[metric]).toFixed(3) : null;

  const cards = [
    {
      icon: "db", tint: "violet", label: "Dataset",
      value: summary.row_count != null ? Number(summary.row_count).toLocaleString() + " rows" : "…",
      sub: summary.column_count != null ? `${summary.column_count} columns <span class="good">✓ profiled</span>` : "profiling…",
    },
    {
      icon: "trophy", tint: "violet", label: "Best model",
      value: best.candidate_name || "—",
      sub: bestScore ? `${metric}: <span class="good">${bestScore}</span>` : "pending evaluation",
    },
    {
      icon: "layers", tint: "green", label: "Candidates trained",
      value: String(results.length || 0),
      sub: results.length ? `${succeeded} succeeded` : "not started",
    },
    {
      icon: "clock", tint: "amber", label: terminal ? "Total runtime" : "Running for",
      value: formatDuration(run.elapsed_seconds),
      sub: run.status.replaceAll("_", " "),
    },
    {
      icon: "sparkle", tint: "violet", label: "LLM calls",
      value: String(run.llm_call_count || 0),
      sub: "fully trace-logged",
    },
  ];
  const quality = summary.quality;
  if (quality) {
    cards.push({
      icon: "shield", tint: "green", label: "Data quality",
      value: `${Math.round(quality.overall * 100)}%`,
      sub: "overall score",
    });
  }
  const insights = run.insights || [];
  if (insights.length) {
    cards.push({
      icon: "bulb", tint: "violet", label: "Auto insights",
      value: String(insights.length),
      sub: "generated from your data",
    });
  }
  if (summary.pii_columns_detected) {
    cards.push({
      icon: "shield", tint: "amber", label: "PII columns",
      value: String(summary.pii_columns_detected),
      sub: "redacted before any AI step",
    });
  }

  $("stat-cards").innerHTML = cards
    .map(
      (c) => `
      <div class="stat-card">
        <span class="stat-icon ${c.tint}">${ICONS[c.icon]}</span>
        <div class="stat-body">
          <div class="stat-label">${c.label}</div>
          <div class="stat-value">${escapeHtml(c.value)}</div>
          <div class="stat-sub">${c.sub}</div>
        </div>
      </div>`
    )
    .join("");
}

/* ================= stage tracker ================= */

function renderStageTracker(run) {
  const done = new Set(run.stages_done);
  const durations = {};
  for (const rec of run.stage_timeline || []) durations[rec.node] = rec.duration_seconds;

  const tracker = $("stage-tracker");
  tracker.innerHTML = "";
  const terminal = ["completed", "failed", "cancelled"].includes(run.status);
  tracker.classList.toggle("compact", terminal);
  $("pipeline-sub").textContent = terminal
    ? `finished in ${formatDuration(run.elapsed_seconds)}`
    : run.status === "awaiting_confirmation"
      ? "paused — waiting for your confirmation"
      : run.status === "awaiting_feature_approval"
        ? "paused — waiting for your feature-plan review"
        : "running";

  const CHECKPOINT_STAGES = { confirm: "awaiting_confirmation", feature_approval: "awaiting_feature_approval" };
  let activeAssigned = false;
  for (const stage of STAGES) {
    const li = document.createElement("li");
    li.className = "stage";
    const isCheckpoint = CHECKPOINT_STAGES[stage.node] === run.status;
    const stageDone = stage.node === "poll_training" ? done.has("evaluate") : done.has(stage.node);

    let stateClass = "pending";
    let statusText = "Pending";
    if (stageDone) {
      stateClass = "done";
      statusText = "Completed";
    } else if (isCheckpoint) {
      stateClass = "needs_input";
      statusText = "Needs input";
      activeAssigned = true;
    } else if (!activeAssigned && !terminal) {
      stateClass = "active";
      statusText = "Running";
      activeAssigned = true;
    } else if (!activeAssigned && run.status === "failed") {
      stateClass = "failed";
      statusText = "Failed";
      activeAssigned = true;
    } else if (terminal && !stageDone) {
      statusText = "Skipped";
    }

    li.classList.add(stateClass);
    const duration = durations[stage.node === "poll_training" ? "poll_training" : stage.node];
    const retry = (run.retry_count || {}).feature_engineering;
    const showRetry = stateClass === "active" && ["feature_engineering", "apply_feature_plan"].includes(stage.node) && retry;

    li.innerHTML = `
      <span class="stage-dot">${stateClass === "done" ? ICONS.check : ICONS[stage.icon]}</span>
      <span class="stage-label">${stage.label}</span>
      <span class="stage-status">${statusText}</span>
      ${stageDone && duration != null ? `<span class="stage-time">${formatDuration(duration)}</span>` : ""}
      ${showRetry ? `<span class="stage-retry">Attempt ${retry + 1} of 4</span>` : ""}
    `;
    tracker.appendChild(li);
  }
}

function renderTrainProgress(run) {
  const box = $("train-progress");
  const results = run.training_results || [];
  const inTraining =
    run.status === "running" &&
    (run.stages_done || []).includes("dispatch_training") &&
    !(run.stages_done || []).includes("evaluate");
  if (!inTraining || !results.length) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");
  const finished = results.filter((r) => ["succeeded", "failed", "timed_out"].includes(r.status)).length;
  const pct = Math.round((finished / results.length) * 100);
  const runningNames = results.filter((r) => r.status === "running").map((r) => r.candidate_name);
  $("train-progress-text").innerHTML = `
    <span>Training ${results.length} candidate model(s)${runningNames.length ? ` — now: ${escapeHtml(runningNames.join(", "))}` : ""}</span>
    <span class="mono">${finished} of ${results.length} finished · ${pct}%</span>`;
  $("train-progress-fill").style.transform = `scaleX(${Math.max(pct, 4) / 100})`;
  renderTuningProgress(results);
}

function tuningStatusText(result) {
  const t = result.tuning || {};
  if (result.status === "pending") return "queued";
  if (result.status === "timed_out") return "timed out — still training in the background, no longer tracked";
  if (t.note) return t.note;
  if (!t.enabled) return result.status === "running" ? "training…" : result.status;
  const last = t.history[t.history.length - 1];
  const best = last ? `best ${t.metric} ${last.best_score.toFixed(3)}` : "starting…";
  const doneTraining = ["succeeded", "failed", "timed_out"].includes(result.status);
  return doneTraining ? `${t.trials_done} trial(s) · ${best}` : `trial ${t.trials_done} of ${t.trials_total} · ${best}`;
}

function renderTuningProgress(results) {
  $("tuning-progress-list").innerHTML = results
    .map((r) => {
      const t = r.tuning || {};
      const pct = t.enabled && t.trials_total ? Math.round((t.trials_done / t.trials_total) * 100) : r.status === "succeeded" ? 100 : 0;
      return `
      <div class="tuning-row">
        <span class="tuning-name" title="${escapeHtml(r.candidate_name)}">${escapeHtml(r.candidate_name)}</span>
        <span class="tuning-track"><span class="tuning-fill ${["failed", "timed_out"].includes(r.status) ? "failed" : ""}" style="transform:scaleX(${Math.max(pct, 3) / 100})"></span></span>
        <span class="tuning-status mono small">${escapeHtml(tuningStatusText(r))}</span>
      </div>`;
    })
    .join("");
}

/* ================= header tags ================= */

function renderHeaderTags(run) {
  const box = $("header-tags");
  const spec = run.task_spec || {};
  if (!spec.task_type) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  const tags = [spec.task_type.charAt(0).toUpperCase() + spec.task_type.slice(1)];
  if (spec.task_type === "classification") {
    const target = (run.profile_columns || []).find((c) => c.name === spec.target_column);
    if (target && target.n_unique != null) tags.push(target.n_unique === 2 ? "Binary" : "Multiclass");
  }
  if (spec.metric) tags.push(spec.metric.toUpperCase());
  box.innerHTML = tags.map((t) => `<span class="header-tag">${escapeHtml(t)}</span>`).join("");
  box.classList.remove("hidden");
}

function renderBreadcrumb(run) {
  $("run-breadcrumb").classList.remove("hidden");
  $("run-breadcrumb-name").textContent = run.filename;
}

/* ================= reasoning rail ================= */

const LIVE_RUN_STATUSES = ["profiling", "running", "awaiting_confirmation", "awaiting_feature_approval"];

/* Mirrors renderStageTracker's precedence (first not-yet-done stage is the
   current one) to find which stage the reasoning rail should describe,
   without duplicating that function's DOM rendering. */
function reasoningStageIndex(run) {
  const done = new Set(run.stages_done || []);
  const terminal = !LIVE_RUN_STATUSES.includes(run.status);
  const CHECKPOINT_STAGES = { confirm: "awaiting_confirmation", feature_approval: "awaiting_feature_approval" };
  for (let i = 0; i < STAGES.length; i++) {
    const stage = STAGES[i];
    const isCheckpoint = CHECKPOINT_STAGES[stage.node] === run.status;
    const stageDone = stage.node === "poll_training" ? done.has("evaluate") : done.has(stage.node);
    if (stageDone) continue;
    if (isCheckpoint || !terminal) return i;
  }
  return STAGES.length - 1;
}

function renderReasoningRail(run) {
  const rail = $("reasoning-rail");
  if (!LIVE_RUN_STATUSES.includes(run.status)) {
    rail.classList.add("hidden");
    return;
  }
  rail.classList.remove("hidden");

  const idx = reasoningStageIndex(run);
  const stage = STAGES[idx];
  const details = STAGE_DETAILS[stage.node] || {};
  const stepNum = idx + 1;
  $("reasoning-step-count").textContent = `Step ${stepNum} of ${STAGES.length}`;
  $("reasoning-progress-fill").style.width = `${Math.round((stepNum / STAGES.length) * 100)}%`;
  $("reasoning-stage-label").textContent = stage.label;
  $("reasoning-what").textContent = details.what || "";

  const results = run.training_results || [];
  const inTraining = ["dispatch_training", "poll_training"].includes(stage.node) && results.length;
  const retry = (run.retry_count || {})[stage.node];
  const section = $("reasoning-underhood-section");
  const checklist = $("reasoning-checklist");
  if (inTraining) {
    section.classList.remove("hidden");
    checklist.innerHTML = results
      .map((r) => {
        const cls =
          r.status === "succeeded" ? "done"
          : r.status === "failed" || r.status === "timed_out" ? "failed"
          : r.status === "running" ? "active"
          : "";
        const icon =
          r.status === "succeeded" ? ICONS.check
          : r.status === "failed" || r.status === "timed_out" ? ICONS.error
          : r.status === "running" ? '<span class="stage-spinner"></span>'
          : ICONS.clock;
        const label = r.status === "timed_out" ? `${r.candidate_name} — timed out` : r.candidate_name;
        return `<li class="${cls}">${icon}<span>${escapeHtml(label)}</span></li>`;
      })
      .join("");
  } else if (retry) {
    section.classList.remove("hidden");
    checklist.innerHTML = `<li class="active"><span class="stage-spinner"></span><span>Retrying — attempt ${retry + 1} of 4</span></li>`;
  } else {
    section.classList.add("hidden");
    checklist.innerHTML = "";
  }

  $("reasoning-why").innerHTML = `
    <strong>Why this step matters</strong>
    ${escapeHtml(details.why || "")}
    <span class="trust-note">${ICONS.shield} Only statistical summaries and schemas reach the AI model — never your raw rows.</span>`;

  renderEventsLog($("reasoning-log"), run.events || []);
}

function renderEventsLog(container, events) {
  container.innerHTML = events.length
    ? [...events]
        .reverse()
        .map(
          (e) => `<li>${ICONS.check}<span>${escapeHtml(e.message)}</span>${
            e.timestamp ? `<span class="reasoning-log-time">${new Date(e.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>` : ""
          }</li>`
        )
        .join("")
    : `<li class="muted small">No stages completed yet.</li>`;
}

function renderLogsTab(run) {
  renderEventsLog($("logs-tab-events"), run.events || []);
}

$("share-btn").addEventListener("click", async () => {
  const btn = $("share-btn");
  const original = btn.innerHTML;
  const url = `${window.location.origin}${window.location.pathname}?run=${encodeURIComponent(currentRunId)}`;
  try {
    await navigator.clipboard.writeText(url);
    btn.textContent = "Link copied";
  } catch {
    btn.textContent = "Copy failed";
  }
  setTimeout(() => { btn.innerHTML = original; }, 1600);
});

$("export-report-btn").addEventListener("click", () => {
  if (!lastRun || !lastRun.report) return;
  const blob = new Blob([lastRun.report], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${lastRun.filename.replace(/\.[^.]+$/, "")}-report.md`;
  a.click();
  URL.revokeObjectURL(url);
});

/* ================= confirm checkpoint ================= */

function renderConfirm(run) {
  const card = $("confirm-card");
  if (run.status !== "awaiting_confirmation") {
    card.classList.add("hidden");
    return;
  }
  card.classList.remove("hidden");

  const spec = run.task_spec || {};
  $("confirm-reason").textContent = spec.ambiguity_reason
    ? spec.ambiguity_reason
    : "Review the inferred task before compute-heavy work begins — every field below is editable.";

  const summary = run.profile_summary || {};
  $("dataset-summary").innerHTML = `
    <span class="chip detected">${ICONS.check} ${summary.row_count ?? "?"} rows</span>
    <span class="chip detected">${ICONS.check} ${summary.column_count ?? "?"} columns</span>
    ${summary.pii_columns_detected ? `<span class="chip flagged" title="Redacted from every AI-facing step">${ICONS.warning} ${summary.pii_columns_detected} PII column(s) redacted</span>` : ""}`;

  const targetSelect = $("target-select");
  if (!targetSelect.dataset.filled) {
    targetSelect.innerHTML = "";
    const timeSelect = $("time-select");
    timeSelect.innerHTML = `<option value="">none — rows are not time-ordered</option>`;
    for (const col of run.profile_columns || []) {
      const opt = document.createElement("option");
      opt.value = col.name;
      opt.textContent = `${col.name} (${col.dtype}${col.is_pii ? ", PII" : ""})`;
      if (col.name === spec.target_column) opt.selected = true;
      targetSelect.appendChild(opt);
      const timeOpt = opt.cloneNode(true);
      timeOpt.selected = col.name === spec.time_column;
      timeSelect.appendChild(timeOpt);
    }
    targetSelect.dataset.filled = "1";
    if (spec.task_type) $("tasktype-select").value = spec.task_type;
    fillMetrics(spec.metric);
  }
}

function fillMetrics(preselect) {
  const task = $("tasktype-select").value;
  const metricSelect = $("metric-select");
  metricSelect.innerHTML = "";
  for (const metric of METRICS[task] || []) {
    const opt = document.createElement("option");
    opt.value = metric;
    opt.textContent = metric;
    if (metric === preselect) opt.selected = true;
    metricSelect.appendChild(opt);
  }
}
$("tasktype-select").addEventListener("change", () => fillMetrics());

$("cv-enabled-input").addEventListener("change", (e) => {
  $("cv-folds-input").disabled = !e.target.checked;
});

$("confirm-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const cvEnabled = $("cv-enabled-input").checked;
  const body = {
    target_column: $("target-select").value,
    task_type: $("tasktype-select").value,
    metric: $("metric-select").value,
    time_column: $("time-select").value || null,
    constraints: [],
    cv_enabled: cvEnabled,
    cv_folds: cvEnabled ? Math.max(2, Math.min(10, Number($("cv-folds-input").value) || 5)) : 0,
    tuning_enabled: $("tuning-enabled-input").checked,
    feature_selection_enabled: $("feature-selection-input").checked,
  };
  const res = await authFetch(`/api/runs/${currentRunId}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    alert("Confirmation failed: " + ((await res.json()).detail || res.statusText));
    return;
  }
  $("confirm-card").classList.add("hidden");
  $("target-select").dataset.filled = "";
});

/* ================= feature engineering approval ================= */

function renderFeatureApproval(run) {
  const card = $("feature-approval-card");
  if (run.status !== "awaiting_feature_approval") {
    card.classList.add("hidden");
    return;
  }
  card.classList.remove("hidden");

  const list = $("feature-step-list");
  if (!list.dataset.filled) {
    const eda = run.eda_report || {};
    $("eda-insights-list").innerHTML = (eda.insights || [])
      .map((i) => `<li class="insight-${i.tone}">${ICONS[INSIGHT_ICON[i.tone]] || ICONS.sparkle}<span>${escapeHtml(i.message)}</span></li>`)
      .join("");

    const steps = (run.feature_plan || {}).steps || [];
    list.innerHTML = steps.length
      ? steps
          .map((step, i) => {
            const origin = step.source === "eda" ? "data analysis" : "AI planner";
            const cols = (step.columns || []).map(escapeHtml).join(", ");
            return `<li>
              <input type="checkbox" data-step-index="${i}" checked />
              <span>
                <span class="step-op">${escapeHtml(step.op)}</span> on <span class="mono">${cols}</span>
                <span class="chip inferred step-source">${escapeHtml(origin)}</span>
                <span class="step-rationale">${escapeHtml(step.rationale || "")}</span>
              </span>
            </li>`;
          })
          .join("")
      : `<li><span>No transformation steps were suggested — training will use the raw numeric columns as-is.</span></li>`;

    const resampling = run.resampling_suggestion || { suggested: false };
    const block = $("resampling-block");
    const spec = run.task_spec || {};
    if (spec.task_type === "classification") {
      block.classList.remove("hidden");
      $("resampling-enabled-input").checked = !!resampling.suggested;
      $("resampling-method-select").value = resampling.method && resampling.method !== "none" ? resampling.method : "smote";
      $("resampling-method-select").disabled = !resampling.suggested;
      $("resampling-reason").textContent = resampling.suggested
        ? resampling.reason
        : "Not suggested for this dataset — enable manually if you still want to balance classes during training.";
    } else {
      block.classList.add("hidden");
    }

    list.dataset.filled = "1";
  }
}

$("resampling-enabled-input").addEventListener("change", (e) => {
  $("resampling-method-select").disabled = !e.target.checked;
});

$("approve-features-btn").addEventListener("click", async () => {
  const approvedIndices = Array.from(document.querySelectorAll('#feature-step-list input[type="checkbox"]:checked')).map(
    (el) => Number(el.dataset.stepIndex)
  );
  const resamplingEnabled = $("resampling-enabled-input").checked;
  const body = {
    approved_step_indices: approvedIndices,
    resampling_enabled: resamplingEnabled,
    resampling_method: resamplingEnabled ? $("resampling-method-select").value : "none",
  };
  const btn = $("approve-features-btn");
  btn.disabled = true;
  try {
    const res = await authFetch(`/api/runs/${currentRunId}/approve-features`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    $("feature-approval-card").classList.add("hidden");
    $("feature-step-list").dataset.filled = "";
  } catch (err) {
    alert("Approval failed: " + err.message);
  } finally {
    btn.disabled = false;
  }
});

/* ================= leakage ================= */

function renderLeakage(run) {
  const flags = run.leakage_flags || [];
  const card = $("leakage-card");
  if (!flags.length) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  $("leakage-list").innerHTML = flags
    .map(
      (f) =>
        `<li>${ICONS.warning}<span><strong>${escapeHtml(f.column)}</strong> — ${escapeHtml(f.reason)} <span class="chip flagged severity-${f.severity}">${f.severity}</span></span></li>`
    )
    .join("");
}

/* ================= dataset summary donut ================= */

function classifyDtype(dtype) {
  const d = String(dtype || "").toLowerCase();
  if (d.includes("int") || d.includes("float")) return "Numeric";
  if (d.includes("datetime") || d.includes("date")) return "Date/time";
  if (d.includes("bool")) return "Boolean";
  return "Categorical / text";
}

function renderDatasetSummary(run) {
  const card = $("dataset-card");
  const columns = run.profile_columns || [];
  if (!columns.length) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  const summary = run.profile_summary || {};
  $("dataset-sub").textContent = `${Number(summary.row_count || 0).toLocaleString()} rows`;

  const groups = {};
  for (const col of columns) {
    const kind = classifyDtype(col.dtype);
    groups[kind] = (groups[kind] || 0) + 1;
  }
  const order = ["Numeric", "Categorical / text", "Date/time", "Boolean"];
  const entries = order.filter((k) => groups[k]).map((k) => [k, groups[k]]);
  const total = entries.reduce((acc, [, n]) => acc + n, 0);

  const styles = getComputedStyle(document.documentElement);
  const palette = DONUT_KEYS.map((k) => styles.getPropertyValue(k).trim());

  // donut: SVG circle strokes with a 2px surface gap between segments
  const R = 44, C = 2 * Math.PI * R;
  const gapPx = total > 1 ? 3 : 0;
  let offset = 0;
  let svg = "";
  entries.forEach(([, count], i) => {
    const frac = count / total;
    const len = Math.max(frac * C - gapPx, 1);
    svg += `<circle cx="60" cy="60" r="${R}" fill="none" stroke="${palette[i % palette.length]}"
      stroke-width="14" stroke-linecap="butt"
      stroke-dasharray="${len} ${C - len}" stroke-dashoffset="${-offset}"
      transform="rotate(-90 60 60)"/>`;
    offset += frac * C;
  });
  $("donut").innerHTML = svg;
  $("donut-center").innerHTML = `${total}<small>columns</small>`;

  $("donut-legend").innerHTML = entries
    .map(
      ([kind, count], i) => `
      <li><span class="swatch" style="background:${palette[i % palette.length]}"></span>
      ${escapeHtml(kind)}<span class="count">${count} (${Math.round((count / total) * 100)}%)</span></li>`
    )
    .join("");

  const worstNull = columns.reduce((max, col) => Math.max(max, col.null_rate || 0), 0);
  const piiCount = columns.filter((c) => c.is_pii).length;
  const targetCol = (run.task_spec || {}).target_column;
  $("dataset-chips").innerHTML = `
    <span class="chip detected">${ICONS.check} worst null rate: ${(worstNull * 100).toFixed(1)}%</span>
    ${targetCol ? `<span class="chip detected">${ICONS.check} target: <span class="mono">${escapeHtml(targetCol)}</span></span>` : ""}
    ${piiCount ? `<span class="chip flagged" title="Redacted from every AI-facing step">${ICONS.warning} ${piiCount} PII column(s)</span>` : ""}`;
}

/* ================= class distribution (classification targets) ================= */

function renderClassDistribution(run) {
  const card = $("classdist-card");
  const spec = run.task_spec || {};
  const confirmed = (run.stages_done || []).includes("confirm");
  const target = (run.profile_columns || []).find((c) => c.name === spec.target_column);
  const entries = target && target.top_values ? Object.entries(target.top_values).sort((a, b) => b[1] - a[1]) : [];
  if (spec.task_type !== "classification" || !confirmed || entries.length < 2) {
    card.classList.add("hidden");
    return;
  }
  card.classList.remove("hidden");
  $("classdist-sub").textContent = `target: ${spec.target_column}`;

  const total = entries.reduce((acc, [, n]) => acc + n, 0);
  const styles = getComputedStyle(document.documentElement);
  const palette = DONUT_KEYS.map((k) => styles.getPropertyValue(k).trim());
  const R = 44, C = 2 * Math.PI * R;
  const gapPx = entries.length > 1 ? 3 : 0;
  let offset = 0;
  let svg = "";
  entries.forEach(([, count], i) => {
    const frac = count / total;
    const len = Math.max(frac * C - gapPx, 1);
    svg += `<circle cx="60" cy="60" r="${R}" fill="none" stroke="${palette[i % palette.length]}"
      stroke-width="14" stroke-linecap="butt"
      stroke-dasharray="${len} ${C - len}" stroke-dashoffset="${-offset}"
      transform="rotate(-90 60 60)"/>`;
    offset += frac * C;
  });
  $("classdist-donut").innerHTML = svg;
  $("classdist-center").innerHTML = `${entries.length}<small>classes</small>`;

  $("classdist-legend").innerHTML = entries
    .map(
      ([label, count], i) => `
      <li><span class="swatch" style="background:${palette[i % palette.length]}"></span>
      ${escapeHtml(label)}<span class="count">${count.toLocaleString()} (${((count / total) * 100).toFixed(1)}%)</span></li>`
    )
    .join("");

  const covered = target.n_unique <= entries.length; // top_values holds every class
  const majority = entries[0][1];
  const minority = entries[entries.length - 1][1];
  const ratio = covered && minority > 0 ? majority / minority : null;
  const plan = run.resampling_plan || {};
  $("classdist-chips").innerHTML = `
    ${ratio != null ? `<span class="chip ${ratio >= 3 ? "flagged" : "detected"}">${ratio >= 3 ? ICONS.warning : ICONS.check} imbalance ratio ${ratio.toFixed(1)} : 1</span>` : ""}
    ${!covered ? `<span class="chip detected">top ${entries.length} of ${target.n_unique} classes shown</span>` : ""}
    ${plan.enabled ? `<span class="chip detected">${ICONS.check} ${escapeHtml(String(plan.method || "").replaceAll("_", " "))} applied during training</span>` : ""}`;
}

/* ================= data quality overview ================= */

function renderQuality(run) {
  const quality = (run.profile_summary || {}).quality;
  const card = $("quality-card");
  if (!quality) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  const overallPct = Math.round(quality.overall * 100);
  $("quality-sub").textContent = `${quality.duplicate_row_count.toLocaleString()} duplicate row(s)`;

  const R = 34, C = 2 * Math.PI * R;
  $("quality-ring").innerHTML = `
    <svg viewBox="0 0 80 80" role="img" aria-label="Overall data quality ${overallPct}%">
      <circle cx="40" cy="40" r="${R}" fill="none" stroke="var(--border-subtle)" stroke-width="8"/>
      <circle cx="40" cy="40" r="${R}" fill="none" stroke="var(--accent-success)" stroke-width="8" stroke-linecap="round"
        stroke-dasharray="${((overallPct / 100) * C).toFixed(2)} ${C.toFixed(2)}" transform="rotate(-90 40 40)"/>
    </svg>
    <div class="quality-ring-label"><strong>${overallPct}%</strong><small>overall</small></div>`;

  const dims = [
    { label: "Completeness", value: quality.completeness, hint: "share of cells that are not null" },
    { label: "Uniqueness", value: quality.uniqueness, hint: "share of rows that are not exact duplicates" },
  ];
  $("quality-bars").innerHTML = dims
    .map(
      (d) => `
      <div class="quality-row" title="${d.hint}">
        <span class="quality-name">${d.label}</span>
        <span class="fi-track"><span class="fi-fill quality-fill" style="width:${(d.value * 100).toFixed(1)}%"></span></span>
        <span class="quality-value mono">${Math.round(d.value * 100)}%</span>
      </div>`
    )
    .join("");
}

/* ================= experiments tab ================= */

function renderExperimentsTab(run) {
  renderExperimentsStatCards(run);
  renderExperimentsBarChart(run);
  renderExperimentsTrend(run);
  renderExperimentsTable(run);
  renderExperimentsDonuts(run);
  renderExperimentsBestPanel(run);
}

function renderExperimentsStatCards(run) {
  const results = run.training_results || [];
  const best = run.best_model || {};
  const metric = (run.task_spec || {}).metric;
  const totalTrials = results.reduce((sum, r) => sum + (r.tuning?.trials_done || 1), 0);
  const avgDuration = results.length
    ? results.reduce((sum, r) => sum + (r.duration_seconds || 0), 0) / results.length
    : 0;
  const totalDuration = results.reduce((sum, r) => sum + (r.duration_seconds || 0), 0);
  const bestCv = metric && best.cv_metrics && best.cv_metrics[metric];
  const bestScore = bestCv ? bestCv.mean : (metric && best.metrics && metric in best.metrics ? Number(best.metrics[metric]) : null);

  const cards = [
    { icon: "layers", tint: "violet", label: "Total Experiments", value: String(totalTrials), sub: "trials across all candidates" },
    { icon: "grid", tint: "violet", label: "Models Evaluated", value: String(results.length), sub: "unique candidates" },
    { icon: "trophy", tint: "amber", label: `Best ${metric ? metric.toUpperCase() : "Score"}${bestCv ? " (CV)" : ""}`,
      value: bestScore != null ? bestScore.toFixed(3) : "—", sub: escapeHtml(best.candidate_name || "—") },
    { icon: "clock", tint: "green", label: "Avg. Training Time", value: formatDuration(avgDuration), sub: "per candidate" },
    { icon: "cpu", tint: "green", label: "Total Compute Time", value: formatDuration(totalDuration), sub: "total wall time" },
  ];

  $("exp-stat-cards").innerHTML = cards
    .map(
      (c) => `
      <div class="stat-card">
        <span class="stat-icon ${c.tint}">${ICONS[c.icon]}</span>
        <div class="stat-body">
          <div class="stat-label">${escapeHtml(c.label)}</div>
          <div class="stat-value">${c.value}</div>
          <div class="stat-sub">${c.sub}</div>
        </div>
      </div>`
    )
    .join("");
}

function renderExperimentsBarChart(run) {
  const results = (run.training_results || []).filter((r) => r.status === "succeeded");
  const metric = (run.task_spec || {}).metric;
  const bestId = (run.best_model || {}).run_id;
  const card = $("exp-bar-chart").closest(".card");
  if (!results.length || !metric) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  const lowerIsBetter = metric === "rmse" || metric === "mae";
  const withMetric = results.filter((r) => r.metrics && metric in r.metrics);
  const sorted = [...withMetric].sort((a, b) =>
    lowerIsBetter ? a.metrics[metric] - b.metrics[metric] : b.metrics[metric] - a.metrics[metric]
  );
  $("exp-bar-sub").textContent = `ranked by ${metric} (held-out)`;

  const values = sorted.map((r) => Number(r.metrics[metric]));
  const allUnitRange = values.every((v) => v >= 0 && v <= 1);
  const scaleMax = allUnitRange ? 1 : Math.max(...values) * 1.15;

  $("exp-bar-chart").innerHTML = sorted
    .map((r) => {
      const value = Number(r.metrics[metric]);
      const pct = Math.max((value / scaleMax) * 100, 1);
      const isBest = r.run_id === bestId;
      return `
      <div class="exp-bar-col" title="${escapeHtml(r.candidate_name)}: ${value.toFixed(3)}">
        <span class="exp-bar-value">${value.toFixed(3)}</span>
        <div class="exp-bar-track"><div class="exp-bar-fill ${isBest ? "champion" : ""}" style="height:${pct.toFixed(1)}%"></div></div>
        <span class="exp-bar-name">${escapeHtml(r.candidate_name)}</span>
      </div>`;
    })
    .join("");
}

/* ================= results table ================= */

function renderResults(run) {
  const results = run.training_results || [];

  const cvConfig = run.cv_config || {};
  const cvChip = $("cv-config-chip");
  if (cvConfig.enabled) {
    cvChip.className = "chip detected cv-config-chip";
    cvChip.innerHTML = `${ICONS.check}${cvConfig.requested_folds}-fold cross-validation requested`;
  } else {
    cvChip.className = "chip cv-config-chip";
    cvChip.innerHTML = `Cross-validation disabled for this run`;
  }

  const resamplingChip = $("resampling-config-chip");
  const resamplingPlan = run.resampling_plan || {};
  if (resamplingPlan.enabled) {
    const applied = results.find((r) => r.resampling_applied)?.resampling_applied || resamplingPlan.method;
    const note = results.find((r) => r.resampling_note)?.resampling_note;
    resamplingChip.classList.remove("hidden");
    resamplingChip.className = "chip detected cv-config-chip";
    resamplingChip.title = note || "";
    resamplingChip.innerHTML = `${ICONS.check}${escapeHtml(applied.replaceAll("_", " "))} applied to training data`;
  } else {
    resamplingChip.classList.add("hidden");
  }

  const fsChip = $("fs-config-chip");
  const fsConfig = run.feature_selection_config || {};
  if (fsConfig.enabled) {
    const fs = run.feature_selection || {};
    fsChip.classList.remove("hidden");
    fsChip.className = "chip detected cv-config-chip";
    if (fs.enabled && fs.n_features_selected != null) {
      fsChip.title = `Selected with ${fs.basic_model || "a basic model"}: ${(fs.selected_features || []).join(", ")}`;
      fsChip.innerHTML = `${ICONS.check}RFE kept ${fs.n_features_selected} of ${fs.n_features_total} features (all models)`;
    } else {
      fsChip.title = fs.note || "";
      fsChip.innerHTML = fs.note ? `Feature selection skipped` : `Feature selection (RFE) requested`;
    }
  } else {
    fsChip.classList.add("hidden");
  }
}

function cvCell(result, metric) {
  const cv = result.cv_metrics && result.cv_metrics[metric];
  if (!cv) {
    return result.cv_note ? `<span class="muted small" title="${escapeHtml(result.cv_note)}">n/a</span>` : "—";
  }
  return `<span title="${result.cv_folds}-fold cross-validation">${cv.mean.toFixed(4)} ± ${cv.std.toFixed(3)}</span>`;
}

/* ================= experiment trend chart ================= */

const EXP_TREND_COLOR_KEYS = ["--cat-1", "--cat-2", "--cat-3", "--cat-4", "--cat-5", "--cat-6", "--cat-7", "--cat-8"];

function renderExperimentsTrend(run) {
  const results = run.training_results || [];
  const bestId = (run.best_model || {}).run_id;
  const tuned = results.filter((r) => r.tuning && r.tuning.enabled && (r.tuning.history || []).length > 1);

  $("exp-trend-empty").classList.toggle("hidden", tuned.length > 0);
  $("exp-trend-chart").classList.toggle("hidden", tuned.length === 0);
  $("exp-trend-legend").classList.toggle("hidden", tuned.length === 0);
  if (!tuned.length) {
    $("exp-trend-sub").textContent = "";
    return;
  }

  // Stable partition: pull the champion (if present) to the front so it can
  // never be pushed into the overflow fold just because of dispatch order.
  const champFirst = [...tuned].sort((a, b) => (a.run_id === bestId ? -1 : 0) - (b.run_id === bestId ? -1 : 0));
  const named = champFirst.slice(0, 8);
  const overflow = champFirst.slice(8);
  $("exp-trend-sub").textContent = `best-so-far score per trial · ${tuned.length} of ${results.length} candidate(s) shown`;

  const styles = getComputedStyle(document.documentElement);
  const palette = EXP_TREND_COLOR_KEYS.map((k) => styles.getPropertyValue(k).trim());

  const W = 680, H = 240, padL = 50, padR = 16, padT = 14, padB = 30;
  const maxTrials = Math.max(...tuned.map((r) => r.tuning.history.length));
  const allScores = tuned.flatMap((r) => r.tuning.history.map((h) => h.best_score));
  let lo = Math.min(...allScores), hi = Math.max(...allScores);
  if (hi - lo < 1e-9) { hi += 0.001; lo -= 0.001; }
  const span = hi - lo;
  lo -= span * 0.08; hi += span * 0.08;
  const x = (i) => padL + (maxTrials <= 1 ? 0 : (i / (maxTrials - 1)) * (W - padL - padR));
  const y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);

  const gridValues = [lo + (hi - lo) * 0.1, lo + (hi - lo) * 0.5, lo + (hi - lo) * 0.9];
  const grid = gridValues
    .map((v) => `
      <line x1="${padL}" y1="${y(v)}" x2="${W - padR}" y2="${y(v)}" class="tt-grid"></line>
      <text x="${padL - 6}" y="${y(v) + 3}" class="tt-axis" text-anchor="end">${v.toFixed(3)}</text>`)
    .join("");

  let linesSvg = "";
  let finalLabel = "";
  named.forEach((r, i) => {
    const isBest = r.run_id === bestId;
    const color = palette[i % palette.length];
    const points = r.tuning.history.map((h, hi2) => `${x(hi2)},${y(h.best_score)}`).join(" ");
    linesSvg += `<polyline points="${points}" class="tt-line ${isBest ? "champion" : ""}" style="stroke:${color}"></polyline>`;
    if (isBest) {
      const last = r.tuning.history[r.tuning.history.length - 1];
      finalLabel = `<text x="${x(r.tuning.history.length - 1) - 6}" y="${y(last.best_score) - 8}" class="tt-final" text-anchor="end">${last.best_score.toFixed(4)}</text>`;
    }
  });
  overflow.forEach((r) => {
    const points = r.tuning.history.map((h, hi2) => `${x(hi2)},${y(h.best_score)}`).join(" ");
    linesSvg += `<polyline points="${points}" class="tt-line other-fold"></polyline>`;
  });

  $("exp-trend-chart").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Best-so-far tuning score per trial, one line per candidate" style="width:100%;height:auto">
      ${grid}
      ${linesSvg}
      ${finalLabel}
      <text x="${(padL + W - padR) / 2}" y="${H - 8}" class="tt-axis" text-anchor="middle">trial</text>
    </svg>`;

  $("exp-trend-legend").innerHTML = named
    .map((r, i) => `<li><span class="tt-chip" style="background:${palette[i % palette.length]}"></span>${escapeHtml(r.candidate_name)}${r.run_id === bestId ? " (champion)" : ""}</li>`)
    .join("") + (overflow.length ? `<li><span class="tt-chip" style="background:var(--text-secondary);opacity:0.5"></span>${overflow.length} other candidate(s)</li>` : "");
}

/* ================= experiments table ================= */

function renderExperimentsTable(run) {
  const results = run.training_results || [];
  const metric = (run.task_spec || {}).metric;
  const bestId = (run.best_model || {}).run_id;
  const lowerIsBetter = metric === "rmse" || metric === "mae";
  const metricNames = [...new Set(results.flatMap((r) => Object.keys(r.metrics || {})))];
  const primary = metric && metricNames.includes(metric) ? metric : metricNames[0];
  const secondary = metricNames.find((m) => m !== primary);
  const hasCv = primary && results.some((r) => r.cv_metrics && primary in r.cv_metrics);

  // Rank by the SAME value the primary column actually displays (CV mean
  // when hasCv makes the column show cvCell's mean instead of the raw
  // metric, else the raw metric) — sorting by the raw metric while
  // displaying the CV mean produces a table that visibly contradicts its
  // own "ranked by {primary}" sub-label whenever the held-out test score
  // and the CV mean disagree on ordering (real, not hypothetical: seen on
  // an actual pipeline run where CV-mean roc_auc for the champion was
  // lower than two non-champion candidates' CV means, while its raw
  // held-out score was still the highest). Found during Task 8 end-to-end
  // verification against a real run — a fixture with raw==CV-mean by
  // coincidence had masked this.
  const rankValue = (r) => {
    if (hasCv) {
      const cv = r.cv_metrics && r.cv_metrics[primary];
      return cv ? cv.mean : undefined;
    }
    return r.metrics && primary in r.metrics ? r.metrics[primary] : undefined;
  };
  const ranked = [...results].sort((a, b) => {
    const aVal = rankValue(a), bVal = rankValue(b);
    const aHas = aVal !== undefined, bHas = bVal !== undefined;
    if (!aHas && !bHas) return 0;
    if (!aHas) return 1;
    if (!bHas) return -1;
    return lowerIsBetter ? aVal - bVal : bVal - aVal;
  });

  $("exp-table-sub").textContent = primary ? `${results.length} candidate(s) · ranked by ${primary}` : `${results.length} candidate(s)`;

  let html = `<tr><th>Rank</th><th>Model</th><th>Trials</th>${primary ? `<th>${escapeHtml(primary)}${hasCv ? " (CV)" : ""}</th>` : ""}${secondary ? `<th>${escapeHtml(secondary)}</th>` : ""}<th>Training Time</th><th>Status</th></tr>`;
  ranked.forEach((r, i) => {
    const isBest = r.run_id === bestId;
    const trials = r.tuning && r.tuning.enabled ? `${r.tuning.trials_done}/${r.tuning.trials_total}` : "no tuning";
    html += `<tr class="${isBest ? "best" : ""}">
      <td>${i + 1}</td>
      <td>${escapeHtml(r.candidate_name)}${isBest ? '<span class="winner-tag">★ CHAMPION</span>' : ""}</td>
      <td>${escapeHtml(trials)}</td>
      ${primary ? `<td class="num">${r.metrics && primary in r.metrics ? (hasCv ? cvCell(r, primary) : Number(r.metrics[primary]).toFixed(4)) : "—"}</td>` : ""}
      ${secondary ? `<td class="num">${r.metrics && secondary in r.metrics ? Number(r.metrics[secondary]).toFixed(4) : "—"}</td>` : ""}
      <td>${r.duration_seconds != null ? formatDuration(r.duration_seconds) : "—"}</td>
      <td>${escapeHtml(r.status.replaceAll("_", " "))}${r.error ? errorDisclosure(r.error) : ""}</td>
    </tr>`;
  });
  $("exp-table").innerHTML = html;
}

/* ================= experiments: distribution donuts ================= */

function drawDonut(svgId, centerId, legendId, entries, colors, centerLabel) {
  const total = entries.reduce((acc, [, n]) => acc + n, 0);
  if (!total) { $(svgId).innerHTML = ""; $(centerId).innerHTML = ""; $(legendId).innerHTML = ""; return; }
  const R = 44, C = 2 * Math.PI * R;
  const gapPx = entries.length > 1 ? 3 : 0;
  let offset = 0;
  let svg = "";
  entries.forEach(([, count], i) => {
    const frac = count / total;
    const len = Math.max(frac * C - gapPx, 1);
    svg += `<circle cx="60" cy="60" r="${R}" fill="none" stroke="${colors[i % colors.length]}"
      stroke-width="14" stroke-linecap="butt"
      stroke-dasharray="${len} ${C - len}" stroke-dashoffset="${-offset}"
      transform="rotate(-90 60 60)"/>`;
    offset += frac * C;
  });
  $(svgId).innerHTML = svg;
  $(centerId).innerHTML = centerLabel;
  $(legendId).innerHTML = entries
    .map(([label, count], i) => `
      <li><span class="swatch" style="background:${colors[i % colors.length]}"></span>
      ${escapeHtml(label)}<span class="count">${count} (${Math.round((count / total) * 100)}%)</span></li>`)
    .join("");
}

function renderExperimentsDonuts(run) {
  const results = run.training_results || [];
  const bestId = (run.best_model || {}).run_id;
  const metric = (run.task_spec || {}).metric;
  const styles = getComputedStyle(document.documentElement);
  const cssVar = (k) => styles.getPropertyValue(k).trim();

  // By Model — trial-count share per candidate, 8 named + "Other" fold-in
  const named = results.slice(0, 8);
  const overflow = results.slice(8);
  const modelEntries = named.map((r) => [r.candidate_name, r.tuning?.trials_done || 1]);
  const modelColors = EXP_TREND_COLOR_KEYS.map(cssVar);
  if (overflow.length) {
    modelEntries.push(["Other", overflow.reduce((s, r) => s + (r.tuning?.trials_done || 1), 0)]);
    modelColors.push(cssVar("--text-secondary"));
  }
  $("exp-donut-model-sub").textContent = `${results.length} candidate(s)`;
  drawDonut("exp-donut-model", "exp-donut-model-center", "exp-donut-model-legend", modelEntries, modelColors, `${results.length}<small>models</small>`);

  // By Status — fixed status colors, never categorical
  const statusOrder = ["succeeded", "failed", "timed_out", "running", "pending"];
  const statusColor = { succeeded: cssVar("--accent-success"), failed: cssVar("--accent-danger"), timed_out: cssVar("--accent-warning"), running: cssVar("--text-secondary"), pending: cssVar("--text-secondary") };
  const statusCounts = {};
  for (const r of results) statusCounts[r.status] = (statusCounts[r.status] || 0) + 1;
  const statusEntries = statusOrder.filter((s) => statusCounts[s]).map((s) => [s.replaceAll("_", " "), statusCounts[s]]);
  const statusColors = statusOrder.filter((s) => statusCounts[s]).map((s) => statusColor[s]);
  $("exp-donut-status-sub").textContent = `${results.length} total`;
  drawDonut("exp-donut-status", "exp-donut-status-center", "exp-donut-status-legend", statusEntries, statusColors, `${results.length}<small>total</small>`);

  // By Outcome vs champion — champion / close contender (within 2% relative) / trailed
  const best = run.best_model || {};
  const bestScore = metric && best.metrics && metric in best.metrics ? Number(best.metrics[metric]) : null;
  const lowerIsBetter = metric === "rmse" || metric === "mae";
  let champCount = 0, closeCount = 0, trailedCount = 0;
  for (const r of results) {
    if (r.run_id === bestId) { champCount += 1; continue; }
    if (bestScore == null || !r.metrics || !(metric in r.metrics)) { trailedCount += 1; continue; }
    const rel = Math.abs(r.metrics[metric] - bestScore) / Math.abs(bestScore || 1);
    const better = lowerIsBetter ? r.metrics[metric] < bestScore : r.metrics[metric] > bestScore;
    if (rel <= 0.02 || better) closeCount += 1; else trailedCount += 1;
  }
  const outcomeEntries = [["Champion", champCount], ["Close contender", closeCount], ["Trailed", trailedCount]].filter(([, n]) => n > 0);
  const outcomeColorMap = { Champion: cssVar("--accent-primary"), "Close contender": cssVar("--accent-success"), Trailed: cssVar("--text-secondary") };
  const outcomeColors = outcomeEntries.map(([label]) => outcomeColorMap[label]);
  $("exp-donut-outcome-sub").textContent = `vs ${escapeHtml(best.candidate_name || "champion")}`;
  drawDonut("exp-donut-outcome", "exp-donut-outcome-center", "exp-donut-outcome-legend", outcomeEntries, outcomeColors, `${results.length}<small>total</small>`);

  // By Compute Time — ordinal buckets, one hue at 3 opacities
  const buckets = [
    { label: "< 30s", test: (d) => d < 30 },
    { label: "30s - 2m", test: (d) => d >= 30 && d <= 120 },
    { label: "> 2m", test: (d) => d > 120 },
  ];
  const base = cssVar("--cat-2");
  const toRgba = (hex, alpha) => {
    const n = parseInt(hex.replace("#", ""), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
  };
  const computeCounts = buckets.map((b) => results.filter((r) => r.duration_seconds != null && b.test(r.duration_seconds)).length);
  const computeEntries = buckets.map((b, i) => [b.label, computeCounts[i]]).filter(([, n]) => n > 0);
  const computeColors = [toRgba(base, 1), toRgba(base, 0.65), toRgba(base, 0.35)].filter((_, i) => computeCounts[i] > 0);
  $("exp-donut-compute-sub").textContent = `${results.length} candidate(s)`;
  drawDonut("exp-donut-compute", "exp-donut-compute-center", "exp-donut-compute-legend", computeEntries, computeColors, `${results.length}<small>total</small>`);
}

function classImbalanceRatioLabel(run) {
  const spec = run.task_spec || {};
  const target = (run.profile_columns || []).find((c) => c.name === spec.target_column);
  const entries = target && target.top_values ? Object.entries(target.top_values).sort((a, b) => b[1] - a[1]) : [];
  if (spec.task_type !== "classification" || entries.length < 2) return null;
  const majority = entries[0][1], minority = entries[entries.length - 1][1];
  if (!minority) return null;
  const total = majority + minority;
  return `${Math.round((majority / total) * 100)}:${Math.round((minority / total) * 100)}`;
}

function renderExperimentsBestPanel(run) {
  const best = run.best_model || {};
  const panel = $("exp-best-panel");
  if (!best.candidate_name) { panel.innerHTML = `<p class="muted small">No champion selected yet.</p>`; return; }

  const metric = (run.task_spec || {}).metric;
  const results = run.training_results || [];
  const metricNames = [...new Set(results.flatMap((r) => Object.keys(r.metrics || {})))];
  const secondary = metricNames.find((m) => m !== metric);
  const trials = best.tuning && best.tuning.enabled ? `${best.tuning.trials_done}/${best.tuning.trials_total}` : "no tuning";

  const metricRows = [metric, secondary]
    .filter(Boolean)
    .map((m) => {
      const cv = best.cv_metrics && best.cv_metrics[m];
      const value = cv ? `${cv.mean.toFixed(3)} ± ${cv.std.toFixed(3)}` : (best.metrics && m in best.metrics ? Number(best.metrics[m]).toFixed(3) : "—");
      return `<div class="exp-best-row"><span class="label">${escapeHtml(m)}${cv ? " (CV mean)" : ""}</span><span class="value">${value}</span></div>`;
    })
    .join("");

  const ratio = classImbalanceRatioLabel(run);
  const resampling = run.resampling_plan || {};
  const dataUsed = resampling.enabled && best.resampling_applied
    ? `${best.resampling_applied.replaceAll("_", " ")}${ratio ? ` (${ratio})` : ""}`
    : "no resampling";

  panel.innerHTML = `
    <div class="exp-best-header">
      <span class="stat-icon amber">${ICONS.trophy}</span>
      <div>
        <div class="stat-label">Best Experiment</div>
        <h3>${escapeHtml(best.candidate_name)}</h3>
      </div>
    </div>
    <div class="exp-best-row"><span class="label">Trials</span><span class="value">${escapeHtml(trials)}</span></div>
    <div class="exp-best-row"><span class="label">Status</span><span class="value">${escapeHtml((best.status || "succeeded").replaceAll("_", " "))}</span></div>
    <div class="exp-best-section">
      <h4>Key Metrics</h4>
      ${metricRows}
    </div>
    <div class="exp-best-section">
      <h4>Training Info</h4>
      <div class="exp-best-row"><span class="label">Training Time</span><span class="value">${best.duration_seconds != null ? formatDuration(best.duration_seconds) : "—"}</span></div>
      <div class="exp-best-row"><span class="label">Start Time</span><span class="value">${run.created_at ? new Date(run.created_at * 1000).toLocaleString() : "—"}</span></div>
      <div class="exp-best-row"><span class="label">Data Used</span><span class="value">${escapeHtml(dataUsed)}</span></div>
      <div class="exp-best-row"><span class="label">Folds</span><span class="value">${best.cv_folds ? `${best.cv_folds}-fold CV` : "no CV"}</span></div>
    </div>`;
}

/* ================= AI assistant panel ================= */

let chatPendingQuestion = null;

function renderChat(run) {
  const ready = ["completed", "failed"].includes(run.status);
  $("chat-placeholder").classList.toggle("hidden", ready);
  $("chat-thread").classList.toggle("hidden", !ready);
  $("chat-suggestions").classList.toggle("hidden", !ready);
  $("chat-form").classList.toggle("hidden", !ready);
  if (!ready) return;

  const history = run.chat_history || [];
  const bubbles = history.map(
    (m) => `
      <div class="chat-msg chat-${m.role}">
        <span class="chat-role">${m.role === "user" ? "You" : "Assistant"}</span>
        <p>${escapeHtml(m.content)}</p>
      </div>`
  );
  if (chatPendingQuestion != null) {
    bubbles.push(`
      <div class="chat-msg chat-user">
        <span class="chat-role">You</span>
        <p>${escapeHtml(chatPendingQuestion)}</p>
      </div>`);
    bubbles.push(`
      <div class="chat-msg chat-assistant chat-thinking">
        <span class="chat-role">Assistant</span>
        <p>Thinking…</p>
      </div>`);
  }
  $("chat-thread").innerHTML = bubbles.length
    ? bubbles.join("")
    : `<p class="muted small">Ask anything about this run's data, decisions, or results.</p>`;
  $("chat-thread").scrollTop = $("chat-thread").scrollHeight;

  const suggestions = run.suggested_questions || [];
  $("chat-suggestions").innerHTML = suggestions
    .map((q) => `<button type="button" class="suggestion-chip">${escapeHtml(q)}</button>`)
    .join("");
  $("chat-suggestions").querySelectorAll(".suggestion-chip").forEach((chip, i) => {
    chip.addEventListener("click", () => {
      $("chat-input").value = suggestions[i];
      $("chat-input").focus();
    });
  });
}

$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("chat-input");
  const question = input.value.trim();
  if (!question || !currentRunId || chatPendingQuestion != null) return;
  input.value = "";
  $("chat-error").classList.add("hidden");
  chatPendingQuestion = question;
  if (lastRun) renderChat(lastRun); // show the question + thinking bubble immediately
  $("chat-send-btn").disabled = true;
  try {
    const res = await authFetch(`/api/runs/${currentRunId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    chatPendingQuestion = null;
    await poll(); // re-fetch: chat_history now contains both messages
  } catch (err) {
    chatPendingQuestion = null;
    if (lastRun) renderChat(lastRun);
    $("chat-error").textContent = "Could not get an answer: " + err.message;
    $("chat-error").classList.remove("hidden");
  } finally {
    $("chat-send-btn").disabled = false;
  }
});

/* ================= feature importance ================= */

function renderFeatureImportance(run) {
  const importance = (run.best_model || {}).feature_importance || [];
  const card = $("fi-card");
  if (!importance.length) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  const max = Math.max(...importance.map((f) => f.importance), 0.0001);
  $("fi-list").innerHTML = importance
    .map(
      (f) => `
      <div class="fi-row">
        <span class="fi-name" title="${escapeHtml(f.feature)}">${escapeHtml(f.feature)}</span>
        <span class="fi-track"><span class="fi-fill" style="width:${((f.importance / max) * 100).toFixed(1)}%"></span></span>
        <span class="fi-value">${(f.importance * 100).toFixed(1)}%</span>
      </div>`
    )
    .join("");
}

async function loadExplainabilityTab(run) {
  if (!(run.best_model || {}).model_path) {
    $("explainability-narrative").textContent = "";
    $("explainability-plots").innerHTML = "";
    $("explainability-empty").textContent = "No trained model is available for this run yet.";
    $("explainability-empty").classList.remove("hidden");
    return;
  }
  if (explainabilityLoadedFor === run.run_id) return;
  explainabilityLoadedFor = run.run_id;

  $("explainability-narrative").textContent = "";
  $("explainability-plots").innerHTML = "";
  $("explainability-empty").textContent = "Loading explainability data…";
  $("explainability-empty").classList.remove("hidden");

  try {
    const data = await (await authFetch(`/api/runs/${run.run_id}/explainability`)).json();
    renderExplainability(data);
  } catch {
    $("explainability-empty").textContent = "Could not load explainability data for this run.";
    $("explainability-empty").classList.remove("hidden");
    explainabilityLoadedFor = null;
  }
}

function renderShapPlot(plot) {
  return `
    <div class="shap-plot">
      <img src="data:image/png;base64,${plot.image_base64}" alt="${escapeHtml(plot.title)}" />
      ${plot.caption ? `<p class="shap-plot-caption">${escapeHtml(plot.caption)}</p>` : ""}
    </div>`;
}

function renderExplainability(data) {
  const empty = $("explainability-empty");
  const plotsEl = $("explainability-plots");
  const narrative = $("explainability-narrative");

  const hasPlots = data.summary_plot || data.bar_plot || (data.dependence_plots && data.dependence_plots.length);
  if (data.method === "unavailable" || !hasPlots) {
    plotsEl.innerHTML = "";
    narrative.textContent = "";
    empty.textContent = data.note || "SHAP-based impact analysis isn't available for this run.";
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  narrative.textContent = data.narrative || "";

  let html = "";
  if (data.bar_plot) html += renderShapPlot(data.bar_plot);
  if (data.summary_plot) html += renderShapPlot(data.summary_plot);
  if (data.dependence_plots && data.dependence_plots.length) {
    html += `<div class="shap-dependence-grid">${data.dependence_plots.map(renderShapPlot).join("")}</div>`;
  }
  plotsEl.innerHTML = html;
}

/* ================= activity feed ================= */

function renderActivity(run) {
  const events = run.events || [];
  const card = $("activity-card");
  if (!events.length) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  $("activity-list").innerHTML = [...events]
    .reverse()
    .map(
      (e) => `<li>${ICONS.check}<span>${escapeHtml(e.message)}</span>
        ${e.timestamp ? `<span class="activity-time">${relativeTime(e.timestamp)}</span>` : ""}</li>`
    )
    .join("");
}

/* ================= auto insights ================= */

const INSIGHT_ICON = { info: "sparkle", success: "check", warning: "warning", danger: "error" };

function renderInsights(run) {
  const insights = run.insights || [];
  const card = $("insights-card");
  if (!insights.length) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  $("insights-sub").textContent = `${insights.length} suggested based on your data and goal`;
  $("insights-list").innerHTML = insights
    .map(
      (i) => `<li class="insight-${i.tone}">${ICONS[INSIGHT_ICON[i.tone]] || ICONS.sparkle}<span>${escapeHtml(i.message)}</span></li>`
    )
    .join("");
}

/* ================= report / test tabs ================= */

function renderReport(run) {
  $("test-model-btn").classList.toggle("hidden", !run.report);
  const card = $("ai-summary-card");
  if (!run.report) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  const lines = run.report.split("\n").filter((l) => l.trim());
  const lede = lines[0] || "";
  $("report-lede").textContent = lede;
  $("report-lede").classList.toggle("hidden", !lede);
  $("report-body").textContent = lines.slice(1).join("\n").trim() || run.report;

  const hasModel = (run.best_model || {}).model_path;
  $("download-btn").style.display = hasModel ? "" : "none";
  $("download-btn").href = `/api/runs/${run.run_id}/model`;
  $("download-script-btn").style.display = hasModel ? "" : "none";
  $("download-script-btn").href = `/api/runs/${run.run_id}/script`;

  if (!$("tab-test-panel").classList.contains("hidden")) loadPredictTab(run);
}

$("trace-toggle-btn").addEventListener("click", async () => {
  const details = $("trace-details");
  const wasHidden = details.classList.contains("hidden");
  details.classList.toggle("hidden");
  details.open = wasHidden;
  if (wasHidden && !$("trace-body").textContent) {
    const trace = await (await authFetch(`/api/runs/${currentRunId}/trace`)).json();
    $("trace-body").textContent = JSON.stringify(trace, null, 2);
  }
});

const RUN_TABS = ["overview", "experiments", "explainability", "artifacts", "logs"];

function switchRunTab(name) {
  for (const tab of RUN_TABS) {
    const isActive = tab === name;
    $(`tab-${tab}-btn`).classList.toggle("active", isActive);
    $(`tab-${tab}-btn`).setAttribute("aria-selected", String(isActive));
    $(`tab-${tab}-panel`).classList.toggle("hidden", !isActive);
  }
  $("run-rail").classList.toggle("hidden", name !== "overview");
  $("run-layout").classList.toggle("no-rail", name !== "overview");
  if (name === "explainability" && lastRun) loadExplainabilityTab(lastRun);
}
for (const tab of RUN_TABS) {
  $(`tab-${tab}-btn`).addEventListener("click", () => switchRunTab(tab));
}
$("tab-data-btn").addEventListener("click", () => {
  if (lastRun) openDatasetDetail(lastRun.source_run_id || lastRun.run_id);
});

function toggleTestModelPanel() {
  const panel = $("tab-test-panel");
  const wasHidden = panel.classList.contains("hidden");
  panel.classList.toggle("hidden");
  if (wasHidden && lastRun) loadPredictTab(lastRun);
}
$("test-model-btn").addEventListener("click", toggleTestModelPanel);

async function loadPredictTab(run) {
  if (!(run.best_model || {}).model_path) {
    $("predict-form").innerHTML = `<p class="muted small">No trained model is available for this run — nothing to test.</p>`;
    return;
  }
  if (predictFormLoadedFor === run.run_id) return;
  predictFormLoadedFor = run.run_id;

  const form = $("predict-form");
  form.innerHTML = `<p class="muted small">Loading model inputs…</p>`;
  try {
    const schema = await (await authFetch(`/api/runs/${run.run_id}/model/schema`)).json();
    const types = schema.feature_types || {};
    form.innerHTML = schema.feature_columns
      .map((col) => {
        const isText = types[col] === "text";
        return `
        <label class="field">
          <span class="mono small">${escapeHtml(col)}</span>
          <input type="${isText ? "text" : "number"}" ${isText ? "" : 'step="any"'} name="${escapeHtml(col)}" data-kind="${isText ? "text" : "number"}" placeholder="${isText ? "category" : "0"}" />
        </label>`;
      })
      .join("");
  } catch {
    form.innerHTML = `<p class="muted small">Could not load model inputs for this run.</p>`;
    predictFormLoadedFor = null;
  }
}

$("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const values = {};
  for (const input of e.target.querySelectorAll("input[name]")) {
    if (input.value === "") continue; // omitted values are imputed by the model pipeline
    values[input.name] = input.dataset.kind === "text" ? input.value : Number(input.value);
  }
  const resultBox = $("predict-result");
  resultBox.classList.remove("hidden", "is-error");
  resultBox.innerHTML = `<p class="muted small">Scoring…</p>`;
  try {
    const res = await authFetch(`/api/runs/${currentRunId}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    renderPredictResult(data);
  } catch (err) {
    resultBox.classList.add("is-error");
    resultBox.innerHTML = `<p>${escapeHtml(err.message)}</p>`;
  }
});

const WATERFALL_CAPTION =
  "Each bar shows how much that feature pushed this specific prediction up or down from the model's average output. The final value (f(x)) is this row's predicted output.";

function renderPredictResult(data) {
  const resultBox = $("predict-result");
  let html = `<div class="predict-headline">Prediction: ${escapeHtml(String(data.prediction))}</div>`;
  if (data.probabilities) {
    const max = Math.max(...Object.values(data.probabilities), 0.0001);
    html += Object.entries(data.probabilities)
      .sort((a, b) => b[1] - a[1])
      .map(
        ([label, p]) => `
        <div class="predict-proba-row">
          <span>${escapeHtml(label)}</span>
          <span class="fi-track"><span class="fi-fill" style="width:${((p / max) * 100).toFixed(1)}%"></span></span>
          <span class="mono">${(p * 100).toFixed(1)}%</span>
        </div>`
      )
      .join("");
  }
  if (data.waterfall_plot_base64) {
    html += `
      <div class="shap-plot" style="margin-top:10px">
        <p class="muted small">Why:</p>
        <img src="data:image/png;base64,${data.waterfall_plot_base64}" alt="Waterfall plot of this prediction's SHAP contributions" />
        <p class="shap-plot-caption">${escapeHtml(WATERFALL_CAPTION)}</p>
      </div>`;
  }
  resultBox.innerHTML = html;
}

/* ================= caveats + errors ================= */

function renderCaveats(run) {
  const card = $("caveats-card");
  const items = [];
  const flags = run.leakage_flags || [];

  if (run.stages_done && run.stages_done.includes("leakage_check")) {
    items.push(
      flags.length
        ? `${flags.length} possible target-leakage column(s) were flagged but not automatically resolved — review before trusting this model in production.`
        : "No target-leakage signals were detected, but this check is heuristic and can miss cases."
    );
  }
  if (run.report) {
    items.push("Leakage detection, like every automated decision here, is best-effort — it is not a guarantee of correctness.");
  }

  card.classList.toggle("hidden", items.length === 0 || !run.report);
  if (items.length) {
    $("caveats-list").innerHTML = items.map((text) => `<li>${ICONS.warning}<span>${escapeHtml(text)}</span></li>`).join("");
  }
}

function renderErrors(run) {
  const errors = run.errors || [];
  const card = $("error-card");
  if (!errors.length) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  $("error-list").innerHTML = errors.map((e) => `<li>${ICONS.error}<span>${escapeHtml(e)}</span></li>`).join("");
}

/* ================= helpers ================= */

function formatDuration(totalSeconds) {
  const seconds = Math.max(0, Math.round(totalSeconds || 0));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${seconds % 60}s`;
}

function formatBytes(bytes) {
  if (bytes == null) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function relativeTime(epochSeconds) {
  const delta = Math.max(0, Math.round(Date.now() / 1000 - epochSeconds));
  if (delta < 60) return `${delta}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  return `${Math.floor(delta / 3600)}h ago`;
}

/* Training errors (esp. sklearn's) can run to hundreds of characters and
   must never blow up the table or dump a raw stack trace onto a
   non-technical user (DESIGN.md §6 / PRODUCT.md 4.3) — show a short summary
   line with the full text behind a native disclosure widget. */
function errorDisclosure(error) {
  const oneLine = error.replace(/\s+/g, " ").trim();
  const short = oneLine.length > 80 ? oneLine.slice(0, 80) + "…" : oneLine;
  return `
    <details class="error-details">
      <summary>${escapeHtml(short)}</summary>
      <div class="trace-body">${escapeHtml(error)}</div>
    </details>`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

/* ================= collapsible rail cards ================= */

function initCollapsible(cardId) {
  const card = $(cardId);
  const toggle = card.querySelector(".collapsible-toggle");
  const body = card.querySelector(".card-collapsible-body");
  const storageKey = `collapse:${cardId}`;
  const expanded = localStorage.getItem(storageKey) === "true";
  toggle.setAttribute("aria-expanded", String(expanded));
  body.classList.toggle("expanded", expanded);
  toggle.addEventListener("click", () => {
    const next = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(next));
    body.classList.toggle("expanded", next);
    localStorage.setItem(storageKey, String(next));
  });
}
initCollapsible("ai-summary-card");
initCollapsible("activity-card");

$("logout-btn").addEventListener("click", async () => {
  try {
    await window.fetch("/api/auth/logout", { method: "POST" });
  } finally {
    window.location.href = "/login.html";
  }
});

// A shared/bookmarked "Share" link carries ?run=<id>; open straight into
// that run instead of the intake screen. openRun() calls loadRecentRuns()
// itself, so only the no-deep-link path needs the initial call here.
const deepLinkRunId = new URLSearchParams(window.location.search).get("run");
if (deepLinkRunId) {
  openRun(deepLinkRunId);
} else {
  loadRecentRuns();
}
// independent of the per-run poll loop, so the sidebar stays eventually
// consistent regardless of which trigger points did or didn't fire
setInterval(loadRecentRuns, 4000);
