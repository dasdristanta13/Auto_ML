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
};

const STAGES = [
  { node: "profile", label: "Data profiling", icon: "db" },
  { node: "understand_usecase", label: "Understanding", icon: "chat" },
  { node: "confirm", label: "Your review", icon: "userCheck" },
  { node: "leakage_check", label: "Leakage check", icon: "search" },
  { node: "feature_engineering", label: "Feature plan", icon: "sliders" },
  { node: "apply_feature_plan", label: "Transform", icon: "grid" },
  { node: "model_selection", label: "Model search", icon: "layers" },
  { node: "dispatch_training", label: "Dispatch", icon: "send" },
  { node: "poll_training", label: "Training", icon: "cpu" },
  { node: "evaluate", label: "Evaluation", icon: "gauge" },
  { node: "report", label: "Report", icon: "file" },
];

const METRICS = {
  classification: ["f1", "accuracy", "roc_auc"],
  regression: ["rmse", "mae", "r2"],
  forecasting: ["rmse", "mae", "r2"],
};

/* donut palette — set via CSS custom properties so it re-validates per theme */
const DONUT_KEYS = ["--cat-1", "--cat-2", "--cat-3", "--cat-4"];

const $ = (id) => document.getElementById(id);
let pollTimer = null;
let currentRunId = null;
let currentRunStatus = null;
let selectedFile = null;
let lastRun = null;
let predictFormLoadedFor = null;

/* ================= theme ================= */

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("automl-theme", theme);
  const isDark = theme === "dark";
  $("theme-label").textContent = isDark ? "Light mode" : "Dark mode";
  document.querySelector(".icon-moon").classList.toggle("hidden", isDark);
  document.querySelector(".icon-sun").classList.toggle("hidden", !isDark);
  $("theme-toggle").setAttribute("aria-pressed", String(isDark));
  if (lastRun) renderDatasetSummary(lastRun); // re-tint donut for the new surface
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

function setActiveNav(id) {
  document.querySelectorAll(".nav-item.active").forEach((el) => el.classList.remove("active"));
  $(id).classList.add("active");
}

function showIntakeView() {
  stopPolling();
  currentRunId = null;
  $("run-view").classList.add("hidden");
  $("intake-view").classList.remove("hidden");
  setActiveNav("nav-new");
  $("header-eyebrow").textContent = "Agentic AutoML";
  $("run-title").textContent = "Start an experiment";
  $("run-desc").textContent = "Upload data, describe your goal, get a model — explained.";
  $("status-badge").classList.add("hidden");
  $("cancel-btn").classList.add("hidden");
  selectedFile = null;
  dropzone.classList.remove("has-file");
  $("dropzone-label").innerHTML = "<strong>Drop a CSV here</strong> or click to browse";
  $("estimate-row").innerHTML = "";
  $("description").value = "";
  updateSubmit();
  loadRecentRuns();
}

function showRunView() {
  $("intake-view").classList.add("hidden");
  $("run-view").classList.remove("hidden");
  setActiveNav("nav-dashboard");
}

/* ================= recent runs (sidebar) ================= */

async function loadRecentRuns() {
  const box = $("nav-runs");
  try {
    const runs = await (await fetch("/api/runs")).json();
    if (!runs.length) {
      box.innerHTML = `<span class="nav-runs-empty">No runs yet</span>`;
      return;
    }
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
  } catch {
    box.innerHTML = `<span class="nav-runs-empty">No runs yet</span>`;
  }
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
    const res = await fetch("/api/runs", { method: "POST", body: form });
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

$("cancel-btn").addEventListener("click", async () => {
  if (!confirm("Cancel this run? Work completed so far is kept, but no further stages will run.")) return;
  try {
    await fetch(`/api/runs/${currentRunId}/cancel`, { method: "POST" });
  } catch { /* poll() reflects the outcome regardless */ }
});

/* ================= run view + polling ================= */

function openRun(runId) {
  currentRunId = runId;
  currentRunStatus = null;
  lastRun = null;
  showRunView();
  $("trace-details").classList.add("hidden");
  $("trace-body").textContent = "";
  predictFormLoadedFor = null;
  $("predict-result").classList.add("hidden");
  switchTab("report");
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
    const res = await fetch(`/api/runs/${currentRunId}`);
    if (!res.ok) return;
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

  $("header-eyebrow").textContent = `Run ${run.run_id}`;
  $("run-title").textContent = run.filename;
  $("run-desc").textContent = run.description || "";

  const badge = $("status-badge");
  badge.classList.remove("hidden");
  badge.className = `status-badge ${run.status}`;
  $("status-badge-text").textContent = run.status.replaceAll("_", " ");

  $("cancel-btn").classList.toggle("hidden", !["profiling", "running", "awaiting_confirmation"].includes(run.status));

  renderStatCards(run);
  renderStageTracker(run);
  renderTrainProgress(run);
  renderConfirm(run);
  renderLeakage(run);
  renderDatasetSummary(run);
  renderResults(run);
  renderFeatureImportance(run);
  renderActivity(run);
  renderReport(run);
  renderCaveats(run);
  renderErrors(run);
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
  $("pipeline-sub").textContent = terminal
    ? `finished in ${formatDuration(run.elapsed_seconds)}`
    : run.status === "awaiting_confirmation"
      ? "paused — waiting for your confirmation"
      : "running";

  let activeAssigned = false;
  for (const stage of STAGES) {
    const li = document.createElement("li");
    li.className = "stage";
    const isConfirm = stage.node === "confirm";
    const stageDone = stage.node === "poll_training" ? done.has("evaluate") : done.has(stage.node);

    let stateClass = "pending";
    let statusText = "Pending";
    if (stageDone) {
      stateClass = "done";
      statusText = "Completed";
    } else if (isConfirm && run.status === "awaiting_confirmation") {
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
  const finished = results.filter((r) => ["succeeded", "failed"].includes(r.status)).length;
  const pct = Math.round((finished / results.length) * 100);
  const runningNames = results.filter((r) => r.status === "running").map((r) => r.candidate_name);
  $("train-progress-text").innerHTML = `
    <span>Training ${results.length} candidate model(s)${runningNames.length ? ` — now: ${escapeHtml(runningNames.join(", "))}` : ""}</span>
    <span class="mono">${finished} of ${results.length} finished · ${pct}%</span>`;
  $("train-progress-fill").style.width = `${Math.max(pct, 4)}%`;
}

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
    for (const col of run.profile_columns || []) {
      const opt = document.createElement("option");
      opt.value = col.name;
      opt.textContent = `${col.name} (${col.dtype}${col.is_pii ? ", PII" : ""})`;
      if (col.name === spec.target_column) opt.selected = true;
      targetSelect.appendChild(opt);
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

$("confirm-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = {
    target_column: $("target-select").value,
    task_type: $("tasktype-select").value,
    metric: $("metric-select").value,
    constraints: [],
  };
  const res = await fetch(`/api/runs/${currentRunId}/confirm`, {
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

/* ================= results table ================= */

function renderResults(run) {
  const results = run.training_results || [];
  const card = $("results-card");
  if (!results.length) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  const metric = (run.task_spec || {}).metric;
  $("results-sub").textContent = metric ? `ranked by ${metric}` : "";

  const metricNames = [...new Set(results.flatMap((r) => Object.keys(r.metrics || {})))];
  const bestId = (run.best_model || {}).run_id;
  const zebra = results.length > 15;

  let html = `<tr><th>Candidate</th><th>Status</th>${metricNames.map((m) => `<th>${m}</th>`).join("")}</tr>`;
  for (const r of results) {
    const isBest = r.run_id === bestId;
    html += `<tr class="${isBest ? "best" : ""} ${zebra ? "zebra" : ""}">
      <td>${escapeHtml(r.candidate_name)}${isBest ? '<span class="winner-tag">★ BEST</span>' : ""}</td>
      <td>${r.status}${r.error ? errorDisclosure(r.error) : ""}</td>
      ${metricNames.map((m) => `<td class="num">${r.metrics && m in r.metrics ? Number(r.metrics[m]).toFixed(4) : "—"}</td>`).join("")}
    </tr>`;
  }
  $("results-table").innerHTML = html;
}

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

/* ================= report / test tabs ================= */

function renderReport(run) {
  const card = $("report-card");
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

  if ($("tab-test-btn").classList.contains("active")) loadPredictTab(run);
}

$("trace-toggle-btn").addEventListener("click", async () => {
  const details = $("trace-details");
  const wasHidden = details.classList.contains("hidden");
  details.classList.toggle("hidden");
  details.open = wasHidden;
  if (wasHidden && !$("trace-body").textContent) {
    const trace = await (await fetch(`/api/runs/${currentRunId}/trace`)).json();
    $("trace-body").textContent = JSON.stringify(trace, null, 2);
  }
});

function switchTab(name) {
  const isReport = name === "report";
  $("tab-report-btn").classList.toggle("active", isReport);
  $("tab-report-btn").setAttribute("aria-selected", String(isReport));
  $("tab-test-btn").classList.toggle("active", !isReport);
  $("tab-test-btn").setAttribute("aria-selected", String(!isReport));
  $("tab-report-panel").classList.toggle("hidden", !isReport);
  $("tab-test-panel").classList.toggle("hidden", isReport);
  if (!isReport && lastRun) loadPredictTab(lastRun);
}
$("tab-report-btn").addEventListener("click", () => switchTab("report"));
$("tab-test-btn").addEventListener("click", () => switchTab("test"));

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
    const schema = await (await fetch(`/api/runs/${run.run_id}/model/schema`)).json();
    form.innerHTML = schema.feature_columns
      .map(
        (col) => `
        <label class="field">
          <span class="mono small">${escapeHtml(col)}</span>
          <input type="number" step="any" name="${escapeHtml(col)}" placeholder="0" />
        </label>`
      )
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
    values[input.name] = input.value === "" ? 0 : Number(input.value);
  }
  const resultBox = $("predict-result");
  resultBox.classList.remove("hidden", "is-error");
  resultBox.innerHTML = `<p class="muted small">Scoring…</p>`;
  try {
    const res = await fetch(`/api/runs/${currentRunId}/predict`, {
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

loadRecentRuns();
