/* Agentic AutoML frontend — vanilla JS, no build step.
   Polls GET /api/runs/{id} and renders the pipeline lifecycle:
   upload -> progress timeline -> human confirmation checkpoint -> results/report. */

const STAGES = [
  { node: "profile", label: "Profiling data" },
  { node: "understand_usecase", label: "Understanding use case" },
  { node: "confirm", label: "Your confirmation" },
  { node: "leakage_check", label: "Checking target leakage" },
  { node: "feature_engineering", label: "Planning features" },
  { node: "apply_feature_plan", label: "Applying transformations" },
  { node: "model_selection", label: "Selecting candidate models" },
  { node: "dispatch_training", label: "Dispatching training jobs" },
  { node: "poll_training", label: "Training models" },
  { node: "evaluate", label: "Evaluating candidates" },
  { node: "report", label: "Writing report" },
];

const METRICS = {
  classification: ["f1", "accuracy", "roc_auc"],
  regression: ["rmse", "mae", "r2"],
  forecasting: ["rmse", "mae", "r2"],
};

const $ = (id) => document.getElementById(id);
let pollTimer = null;
let currentRunId = null;
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
  updateSubmit();
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

$("new-run-btn").addEventListener("click", () => {
  stopPolling();
  currentRunId = null;
  $("run-view").classList.add("hidden");
  $("new-run-card").classList.remove("hidden");
  loadRecentRuns();
});

/* ---------- recent runs ---------- */

async function loadRecentRuns() {
  try {
    const runs = await (await fetch("/api/runs")).json();
    const box = $("recent-runs");
    box.innerHTML = runs.length ? "<h3 class='muted small'>Recent runs</h3>" : "";
    for (const run of runs.slice(0, 8)) {
      const el = document.createElement("div");
      el.className = "recent-run";
      el.innerHTML = `<span><strong>${run.filename}</strong> — ${run.description || ""}</span>
                      <span class="badge ${run.status}">${run.status.replaceAll("_", " ")}</span>`;
      el.onclick = () => openRun(run.run_id);
      box.appendChild(el);
    }
  } catch { /* server not ready yet; ignore */ }
}

/* ---------- run view + polling ---------- */

function openRun(runId) {
  currentRunId = runId;
  $("new-run-card").classList.add("hidden");
  $("run-view").classList.remove("hidden");
  stopPolling();
  poll();
  pollTimer = setInterval(poll, 1500);
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
  if (run.status === "completed" || run.status === "failed") stopPolling();
}

function render(run) {
  $("run-title").textContent = `Run ${run.run_id} — ${run.filename}`;
  $("run-desc").textContent = run.description || "";

  const badge = $("status-badge");
  badge.textContent = run.status.replaceAll("_", " ");
  badge.className = `badge ${run.status}`;

  renderTimeline(run);
  renderConfirm(run);
  renderLeakage(run);
  renderResults(run);
  renderReport(run);
  renderErrors(run);
}

function renderTimeline(run) {
  const done = new Set(run.stages_done);
  const confirmDone = ["running", "completed", "failed"].includes(run.status) && done.has("understand_usecase");
  const timeline = $("timeline");
  timeline.innerHTML = "";

  let activeSet = false;
  for (const stage of STAGES) {
    const li = document.createElement("li");
    const isDone = stage.node === "confirm" ? confirmDone : done.has(stage.node);
    // poll_training keeps emitting events while jobs run; treat as done only when evaluate ran
    const reallyDone = stage.node === "poll_training" ? done.has("evaluate") : isDone;

    if (reallyDone) {
      li.className = "done";
      li.innerHTML = `<span class="dot">✓</span> ${stage.label}`;
    } else if (!activeSet && run.status !== "completed" && run.status !== "failed") {
      li.className = "active";
      li.innerHTML = `<span class="dot">●</span> ${stage.label}${stage.node === "confirm" && run.status === "awaiting_confirmation" ? " — action needed below" : ""}`;
      activeSet = true;
    } else {
      li.innerHTML = `<span class="dot"></span> ${stage.label}`;
    }
    timeline.appendChild(li);
  }
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
    ? `The platform needs your input: ${spec.ambiguity_reason}`
    : "Review the inferred task before compute-heavy work begins.";

  const summary = run.profile_summary || {};
  $("dataset-summary").innerHTML = `
    <span class="chip">${summary.row_count ?? "?"} rows</span>
    <span class="chip">${summary.column_count ?? "?"} columns</span>
    ${summary.pii_columns_detected ? `<span class="chip pii">${summary.pii_columns_detected} PII column(s) redacted</span>` : ""}`;

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
    .map((f) => `<li><strong>${f.column}</strong> — ${f.reason} <em>(${f.severity})</em></li>`)
    .join("");
}

function renderResults(run) {
  const results = run.training_results || [];
  const card = $("results-card");
  if (!results.length) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  const metricNames = [...new Set(results.flatMap((r) => Object.keys(r.metrics || {})))];
  const bestId = (run.best_model || {}).run_id;

  let html = `<tr><th>Candidate</th><th>Status</th>${metricNames.map((m) => `<th>${m}</th>`).join("")}</tr>`;
  for (const r of results) {
    const isBest = r.run_id === bestId;
    html += `<tr class="${isBest ? "best" : ""}">
      <td>${r.candidate_name}${isBest ? '<span class="tag">★ BEST</span>' : ""}</td>
      <td>${r.status}${r.error ? ` — ${r.error}` : ""}</td>
      ${metricNames.map((m) => `<td>${r.metrics && m in r.metrics ? Number(r.metrics[m]).toFixed(4) : "—"}</td>`).join("")}
    </tr>`;
  }
  $("results-table").innerHTML = html;

  const hasModel = bestId && (run.best_model || {}).model_path;
  $("download-btn").style.display = hasModel ? "" : "none";
  $("download-btn").href = `/api/runs/${run.run_id}/model`;
  $("trace-btn").href = `/api/runs/${run.run_id}/trace`;
}

function renderReport(run) {
  const card = $("report-card");
  if (!run.report) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  $("report-body").textContent = run.report;
}

function renderErrors(run) {
  const errors = run.errors || [];
  const card = $("error-card");
  if (!errors.length) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  $("error-list").innerHTML = errors.map((e) => `<li>${e}</li>`).join("");
}

loadRecentRuns();
