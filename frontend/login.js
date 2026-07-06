/* Standalone login page script — loads independently of app.js, since this
   page must work with no session at all. Theme toggle mirrors app.js's
   exactly (same localStorage key) so the choice carries over once the user
   reaches the home view. */

const $ = (id) => document.getElementById(id);

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("automl-theme", theme);
  const isDark = theme === "dark";
  $("theme-label").textContent = isDark ? "Light mode" : "Dark mode";
  document.querySelector(".icon-moon").classList.toggle("hidden", isDark);
  document.querySelector(".icon-sun").classList.toggle("hidden", !isDark);
  $("theme-toggle").setAttribute("aria-pressed", String(isDark));
}
$("theme-toggle").addEventListener("click", () => {
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});
applyTheme(localStorage.getItem("automl-theme") || "light");

async function loadDemoHint() {
  const hint = $("login-demo-hint");
  try {
    const res = await fetch("/api/auth/demo-credentials");
    const creds = await res.json();
    hint.textContent = `Demo credentials: ${creds.email} / ${creds.password}`;
  } catch {
    hint.textContent = "";
  }
}
loadDemoHint();

$("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errorBox = $("login-error");
  errorBox.classList.add("hidden");
  const submitBtn = $("login-submit");
  submitBtn.disabled = true;
  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: $("login-email").value.trim(), password: $("login-password").value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "invalid email or password");
    window.location.href = "/";
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove("hidden");
    submitBtn.disabled = false;
  }
});
