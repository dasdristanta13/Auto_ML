/* Agentic AutoML frontend — vanilla JS, no build step.
   Implements PRODUCT.md's intake -> live run (stage tracker + checkpoint) ->
   report flow, styled per DESIGN.md. Polls GET /api/runs/{id} for state. */

const STAGES = [
  { node: "profile", label: "Profiling" },
  { node: "understand_usecase", label: "Understanding" },
  { node: "confirm", label: "Your review" },
  { node: "leakage_check", label: "Leakage check" },
  { node: "feature_engineering", label: "Features" },
  { node: "apply_feature_plan", label: "Transform" },
  { node: "model_selection", label: "Model search" },
  { node: "dispatch_training", label: "Dispatch" },
  { node: "poll_training", label: "Training" },
  { node: "evaluate", label: "Evaluate" },
  { node: "report", label: "Report" },
];

const METRICS = {
  classification: ["f1", "accuracy", "roc_auc"],
  regression: ["rmse", "mae", "r2"],
  forecasting: ["rmse", "mae", "r2"],
};

const ICONS = {
  check: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6 9 17l-5-5"/></svg>',
  sparkle: '<svg class="icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2 14 9 21 12 14 15 12 22 10 15 3 12 10 9Z"/></svg>',
  warning: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>',
  error: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="m9.5 9.5 5 5m0-5-5 5"/></svg>',
};

const $ = (id) => document.getElementById(id);
let pollTimer = null;
let elapsedTimer = null;
let currentRunId = null;
let currentRunCreatedAt = null;
let currentRunStatus = null;
let selectedFile = null;

/* ---------- new run form ---------- */

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
  $("dropzone-label").innerHTML = `<strong>${file.name}</strong> (${(file.size / 1024).toFixed(1)} KB)`;
  estimateDataset(file);
  updateSubmit();
}

/* Never a silent spinner: give an honest, clearly-labeled client-side
   estimate before the file is even uploaded (PRODUCT.md 3.2 / 4.2). This
   reads only the first chunk — never the whole file — into memory. */
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
  $("submit-btn").innerHTML = "<span>Uploading…</span>";
  try {
    const res = await fetch("/api/runs", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const { run_id } = await res.json();
    openRun(run_id);
  } catch (err) {
    alert("Failed to start run: " + err.message);
  } finally {
    $("submit-btn").innerHTML = "<span>Run pipeline</span>";
    updateSubmit();
  }
});

$("new-run-btn").addEventListener("click", () => {
  stopPolling();
  currentRunId = null;
  $("run-view").classList.add("hidden");
  $("intake-view").classList.remove("hidden");
  selectedFile = null;
  dropzone.classList.remove("has-file");
  $("dropzone-label").innerHTML = "<strong>Drop a CSV here</strong> or click to browse";
  $("estimate-row").innerHTML = "";
  $("description").value = "";
  loadRecentRuns();
});

$("cancel-btn").addEventListener("click", async () => {
  if (!confirm("Cancel this run? Work completed so far is kept, but no further stages will run.")) return;
  try {
    await fetch(`/api/runs/${currentRunId}/cancel`, { method: "POST" });
  } catch { /* poll() will reflect the outcome regardless */ }
});

/* ---------- recent runs ---------- */

async function loadRecentRuns() {
  const box = $("recent-runs");
  try {
    const runs = await (await fetch("/api/runs")).json();
    if (!runs.length) {
      box.innerHTML = `<p class="recent-run-empty">No runs yet — upload a dataset above to get started.</p>`;
      return;
    }
    box.innerHTML = "";
    for (const run of runs.slice(0, 8)) {
      const el = document.createElement("div");
      el.className = "recent-run";
      el.innerHTML = `<span><strong>${escapeHtml(run.filename)}</strong> — ${escapeHtml(run.description || "")}</span>
                      <span class="status-badge ${run.status}"><span class="dot"></span>${run.status.replaceAll("_", " ")}</span>`;
      el.onclick = () => openRun(run.run_id);
      box.appendChild(el);
    }
  } catch {
    box.innerHTML = `<p class="recent-run-empty">No runs yet — upload a dataset above to get started.</p>`;
  }
}

/* ---------- run view + polling ---------- */

function openRun(runId) {
  currentRunId = runId;
  currentRunStatus = null;
  $("intake-view").classList.add("hidden");
  $("run-view").classList.remove("hidden");
  $("trace-details").classList.add("hidden");
  $("trace-body").textContent = "";
  stopPolling();
  poll();
  pollTimer = setInterval(poll, 1500);
  elapsedTimer = setInterval(tickElapsed, 1000);
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  if (elapsedTimer) clearInterval(elapsedTimer);
  pollTimer = null;
  elapsedTimer = null;
}

function tickElapsed() {
  if (!currentRunCreatedAt || ["completed", "failed", "cancelled"].includes(currentRunStatus)) return;
  const seconds = Math.floor(Date.now() / 1000 - currentRunCreatedAt);
  $("meta-elapsed").textContent = `Running for ${formatDuration(seconds)}`;
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
  if (["completed", "failed", "cancelled"].includes(run.status)) stopPolling();
}

function render(run) {
  currentRunCreatedAt = run.created_at;
  currentRunStatus = run.status;

  $("run-title").textContent = `${run.filename}`;
  $("run-desc").textContent = run.description || "";
  $("meta-elapsed").textContent = `${["completed", "failed", "cancelled"].includes(run.status) ? "Finished in" : "Running for"} ${formatDuration(run.elapsed_seconds)}`;
  $("meta-cost").textContent = run.llm_call_count ? `${run.llm_call_count} LLM call${run.llm_call_count === 1 ? "" : "s"} this run` : "";

  const badge = $("status-badge");
  badge.className = `status-badge ${run.status}`;
  $("status-badge-text").textContent = run.status.replaceAll("_", " ");

  $("cancel-btn").classList.toggle("hidden", !["profiling", "running", "awaiting_confirmation"].includes(run.status));

  renderStageTracker(run);
  renderDecisionLog(run);
  renderConfirm(run);
  renderLeakage(run);
  renderResults(run);
  renderFeatureImportance(run);
  renderReport(run);
  renderCaveats(run);
  renderErrors(run);
}

function renderStageTracker(run) {
  const done = new Set(run.stages_done);
  const confirmDone = ["running", "completed", "failed", "cancelled"].includes(run.status) && done.has("understand_usecase");
  const tracker = $("stage-tracker");
  tracker.innerHTML = "";

  const terminal = ["completed", "failed", "cancelled"].includes(run.status);
  let activeAssigned = false;

  for (const stage of STAGES) {
    const li = document.createElement("li");
    li.className = "stage";
    const isConfirmStage = stage.node === "confirm";
    const stageDone = isConfirmStage ? confirmDone : (stage.node === "poll_training" ? done.has("evaluate") : done.has(stage.node));

    let stateClass = "";
    let annotation = "";
    if (stageDone) {
      stateClass = "done";
    } else if (isConfirmStage && run.status === "awaiting_confirmation") {
      stateClass = "needs_input";
      annotation = "Action needed below";
      activeAssigned = true;
    } else if (!activeAssigned && !terminal) {
      stateClass = "active";
      activeAssigned = true;
      annotation = stageAnnotation(stage.node, run);
    } else if (!activeAssigned && run.status === "failed" && !stageDone) {
      stateClass = "failed";
      activeAssigned = true;
    }

    li.classList.add(stateClass || "pending");
    const retry = (run.retry_count || {}).feature_engineering;
    const showRetry = stateClass === "active" && ["feature_engineering", "apply_feature_plan"].includes(stage.node) && retry;

    li.innerHTML = `
      <span class="stage-dot">${stateClass === "done" ? ICONS.check : stateClass === "failed" ? "!" : ""}</span>
      <span class="stage-label">${stage.label}</span>
      ${annotation ? `<span class="stage-annotation">${annotation}</span>` : ""}
      ${showRetry ? `<span class="stage-retry">Attempt ${retry + 1} of 4</span>` : ""}
    `;
    tracker.appendChild(li);
  }
}

function stageAnnotation(node, run) {
  const candidateCount = (run.training_results || []).length || (run.candidate_models_count || 0);
  const map = {
    poll_training: candidateCount ? `Training ${candidateCount} candidate model(s)…` : "Training candidate models…",
    dispatch_training: "Starting async training jobs…",
    model_selection: "Choosing candidates for this data…",
    feature_engineering: "Planning transformations…",
    apply_feature_plan: "Applying transformations…",
    leakage_check: "Scanning for leakage signals…",
    understand_usecase: "Interpreting your goal…",
    profile: "Computing statistical profile…",
    evaluate: "Comparing candidates…",
    report: "Writing the plain-language report…",
  };
  return map[node] || "";
}

function renderDecisionLog(run) {
  const log = $("decision-log");
  const events = run.events || [];
  log.innerHTML = events
    .map((e) => `<li>${ICONS.check}<span>${escapeHtml(e.message)}</span></li>`)
    .join("");
}

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

function renderResults(run) {
  const results = run.training_results || [];
  const card = $("results-card");
  if (!results.length) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  const metric = (run.task_spec || {}).metric;
  $("results-sub").textContent = metric
    ? `All candidates tried, ranked by ${metric} — the search wasn't arbitrary.`
    : "All candidates tried.";

  const metricNames = [...new Set(results.flatMap((r) => Object.keys(r.metrics || {})))];
  const bestId = (run.best_model || {}).run_id;
  const zebra = results.length > 15;

  let html = `<tr><th>Candidate</th><th>Status</th>${metricNames.map((m) => `<th>${m}</th>`).join("")}</tr>`;
  for (const r of results) {
    const isBest = r.run_id === bestId;
    html += `<tr class="${isBest ? "best" : ""} ${zebra ? "zebra" : ""}">
      <td>${escapeHtml(r.candidate_name)}${isBest ? '<span class="winner-tag">★ BEST</span>' : ""}</td>
      <td>${r.status}${r.error ? ` — ${escapeHtml(r.error)}` : ""}</td>
      ${metricNames.map((m) => `<td class="num">${r.metrics && m in r.metrics ? Number(r.metrics[m]).toFixed(4) : "—"}</td>`).join("")}
    </tr>`;
  }
  $("results-table").innerHTML = html;
}

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
}

$("trace-toggle-btn").addEventListener("click", async () => {
  const details = $("trace-details");
  const wasHidden = details.classList.contains("hidden");
  details.classList.toggle("hidden");
  details.open = !wasHidden ? false : true;
  if (wasHidden && !$("trace-body").textContent) {
    const trace = await (await fetch(`/api/runs/${currentRunId}/trace`)).json();
    $("trace-body").textContent = JSON.stringify(trace, null, 2);
  }
});

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

/* ---------- helpers ---------- */

function formatDuration(totalSeconds) {
  const seconds = Math.max(0, Math.round(totalSeconds || 0));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const rem = seconds % 60;
  return `${minutes}m ${rem}s`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

loadRecentRuns();
