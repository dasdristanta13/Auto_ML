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

$("cancel-btn").addEventListener("click", async () => {
  if (!confirm("Cancel this run? Work completed so far is kept, but no further stages will run.")) return;
  try {
    await authFetch(`/api/runs/${currentRunId}/cancel`, { method: "POST" });
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
  chatPendingQuestion = null;
  $("chat-input").value = "";
  $("chat-error").classList.add("hidden");
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
    const res = await authFetch(`/api/runs/${currentRunId}`);
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

  $("cancel-btn").classList.toggle(
    "hidden",
    !["profiling", "running", "awaiting_confirmation", "awaiting_feature_approval"].includes(run.status)
  );

  renderStatCards(run);
  renderStageTracker(run);
  renderTrainProgress(run);
  renderConfirm(run);
  renderLeakage(run);
  renderFeatureApproval(run);
  renderDatasetSummary(run);
  renderClassDistribution(run);
  renderQuality(run);
  renderInsights(run);
  renderResults(run);
  renderTuningTrend(run);
  renderFeatureImportance(run);
  renderActivity(run);
  renderReport(run);
  renderChat(run);
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
  const finished = results.filter((r) => ["succeeded", "failed"].includes(r.status)).length;
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
  if (t.note) return t.note;
  if (!t.enabled) return result.status === "running" ? "training…" : result.status;
  const last = t.history[t.history.length - 1];
  const best = last ? `best ${t.metric} ${last.best_score.toFixed(3)}` : "starting…";
  const doneTraining = ["succeeded", "failed"].includes(result.status);
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
        <span class="tuning-track"><span class="tuning-fill ${r.status === "failed" ? "failed" : ""}" style="transform:scaleX(${Math.max(pct, 3) / 100})"></span></span>
        <span class="tuning-status mono small">${escapeHtml(tuningStatusText(r))}</span>
      </div>`;
    })
    .join("");
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

/* ================= results table ================= */

function renderResults(run) {
  const results = run.training_results || [];
  const card = $("results-card");
  if (!results.length) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  const metric = (run.task_spec || {}).metric;
  $("results-sub").textContent = metric ? `ranked by ${metric}` : "";

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

  const metricNames = [...new Set(results.flatMap((r) => Object.keys(r.metrics || {})))];
  const bestId = (run.best_model || {}).run_id;
  const zebra = results.length > 15;
  const hasCv = metric && results.some((r) => r.cv_metrics && metric in r.cv_metrics);

  let html = `<tr><th>Candidate</th><th>Status</th>${metricNames.map((m) => `<th>${m}</th>`).join("")}${hasCv ? `<th>CV ${escapeHtml(metric)}</th>` : ""}</tr>`;
  for (const r of results) {
    const isBest = r.run_id === bestId;
    html += `<tr class="${isBest ? "best" : ""} ${zebra ? "zebra" : ""}">
      <td>${escapeHtml(r.candidate_name)}${isBest ? '<span class="winner-tag">★ BEST</span>' : ""}</td>
      <td>${r.status}${r.error ? errorDisclosure(r.error) : ""}</td>
      ${metricNames.map((m) => `<td class="num">${r.metrics && m in r.metrics ? Number(r.metrics[m]).toFixed(4) : "—"}</td>`).join("")}
      ${hasCv ? `<td class="num">${cvCell(r, metric)}</td>` : ""}
    </tr>`;
  }
  $("results-table").innerHTML = html;
}

function cvCell(result, metric) {
  const cv = result.cv_metrics && result.cv_metrics[metric];
  if (!cv) {
    return result.cv_note ? `<span class="muted small" title="${escapeHtml(result.cv_note)}">n/a</span>` : "—";
  }
  return `<span title="${result.cv_folds}-fold cross-validation">${cv.mean.toFixed(4)} ± ${cv.std.toFixed(3)}</span>`;
}

/* ================= tuning trend chart ================= */

function renderTuningTrend(run) {
  const card = $("tuning-card");
  const best = run.best_model || {};
  const t = best.tuning || {};
  const history = t.history || [];
  if (!t.enabled || history.length < 2) {
    card.classList.add("hidden");
    return;
  }
  card.classList.remove("hidden");

  const direction = t.lower_is_better ? "lower is better" : "higher is better";
  $("tuning-sub").textContent = `${best.candidate_name} · ${history.length} trials · ${t.metric} (${direction})`;

  // geometry: one x slot per trial, y spans the observed score range with headroom
  const W = 520, H = 190, padL = 46, padR = 14, padT = 12, padB = 30;
  const scores = history.map((h) => h.score).concat(history.map((h) => h.best_score));
  let lo = Math.min(...scores), hi = Math.max(...scores);
  if (hi - lo < 1e-9) { hi += 0.001; lo -= 0.001; }
  const span = hi - lo;
  lo -= span * 0.08; hi += span * 0.08;
  const x = (i) => padL + (history.length === 1 ? 0 : (i / (history.length - 1)) * (W - padL - padR));
  const y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);

  const gridValues = [lo + (hi - lo) * 0.1, lo + (hi - lo) * 0.5, lo + (hi - lo) * 0.9];
  const grid = gridValues
    .map(
      (v) => `
      <line x1="${padL}" y1="${y(v)}" x2="${W - padR}" y2="${y(v)}" class="tt-grid"></line>
      <text x="${padL - 6}" y="${y(v) + 3}" class="tt-axis" text-anchor="end">${v.toFixed(3)}</text>`
    )
    .join("");

  const scoreLine = history.map((h, i) => `${x(i)},${y(h.score)}`).join(" ");
  const bestLine = history.map((h, i) => `${x(i)},${y(h.best_score)}`).join(" ");
  const dots = history
    .map(
      (h, i) => `
      <circle cx="${x(i)}" cy="${y(h.score)}" r="3.5" class="tt-dot">
        <title>${h.trial === 0 ? "Trial 0 (proposed baseline)" : `Trial ${h.trial}`} — ${t.metric}: ${h.score.toFixed(4)} (best so far ${h.best_score.toFixed(4)})</title>
      </circle>`
    )
    .join("");
  const last = history[history.length - 1];
  const finalLabel = `<text x="${x(history.length - 1) - 6}" y="${y(last.best_score) - 8}" class="tt-final" text-anchor="end">${last.best_score.toFixed(4)}</text>`;

  $("tuning-chart").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Hyperparameter tuning: ${escapeHtml(t.metric)} per trial for ${escapeHtml(best.candidate_name || "the best model")}" style="width:100%;height:auto">
      ${grid}
      <polyline points="${bestLine}" class="tt-best-line" fill="none"></polyline>
      <polyline points="${scoreLine}" class="tt-score-line" fill="none"></polyline>
      ${dots}
      ${finalLabel}
      <text x="${(padL + W - padR) / 2}" y="${H - 8}" class="tt-axis" text-anchor="middle">trial</text>
    </svg>`;

  $("tuning-legend").innerHTML = `
    <li><span class="tt-chip tt-chip-score"></span>per-trial score</li>
    <li><span class="tt-chip tt-chip-best"></span>best so far</li>`;
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
    const trace = await (await authFetch(`/api/runs/${currentRunId}/trace`)).json();
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

$("logout-btn").addEventListener("click", async () => {
  try {
    await window.fetch("/api/auth/logout", { method: "POST" });
  } finally {
    window.location.href = "/login.html";
  }
});

loadRecentRuns();
// independent of the per-run poll loop, so the sidebar stays eventually
// consistent regardless of which trigger points did or didn't fire
setInterval(loadRecentRuns, 4000);
