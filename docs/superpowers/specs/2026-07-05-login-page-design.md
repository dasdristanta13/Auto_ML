# Login page + session gate

Date: 2026-07-05
Status: approved

## Context

A reference marketing-landing-page mockup (top nav with Product/Features/How it
Works/Pricing/Docs/About links, logo, theme toggle, "Log in"/"Get Started Free"
buttons, purple gradient branding) was the basis for the home view already built on
`feature/ai-assistant-chat`. This spec adds a **login page** styled after the same
image, gated in front of that home view, with a minimal (non-graph) backend session
mechanism. Branched from `feature/ai-assistant-chat` so "the landing page the user
reaches after login" is the current home view/dashboard, not main's older sparse
intake screen.

## Non-negotiable scope constraint

Zero changes to `src/graph/`, `src/agents/`, `src/training/`, `src/profiling/`,
`src/insights/`, `src/llm/`, `src/sandbox/`, or `src/pii/` — this is a new,
self-contained auth layer alongside the pipeline, not a change to it, per the user's
explicit instruction.

## Goal

A branded login page matching the reference image's chrome (nav bar, logo, gradient
background, button style) with a centered email/password login card in place of the
hero/dashboard content. On success, the user is redirected to the existing home view.
The rest of the app (frontend pages and `/api/runs*` endpoints) requires a valid
session; without one, requests are redirected/rejected back to the login page.

## Out of scope

Real user accounts, a user table, password hashing/reset flows, OAuth, multi-user
support. This app remains single-local-user; the "login" is a demo credential gate
consistent with the app's existing "local stand-in" architecture (in-memory `_runs`
registry, no Postgres) — not a step toward production multi-tenant auth.

## Backend

### Session store

New `src/auth/session.py`:

```python
def create_session(email: str) -> str          # returns opaque token, stores {token: {"email": ..., "expires_at": ...}}
def get_session(token: str | None) -> dict | None   # None if missing/expired; expired entries are evicted lazily
def destroy_session(token: str | None) -> None
```

In-memory `dict[str, dict]` module-level store, same pattern as `src/api/server.py`'s
existing `_runs` registry — no new persistence layer. TTL read from
`config/runtime.yaml`'s new `auth.session_ttl_hours` (default 24).

### Config additions

`config/runtime.yaml`:
```yaml
auth:
  session_ttl_hours: 24
```

`.env.example` additions:
```
# Demo login credential for this single-user local build (see src/auth/).
# Defaults below are used if unset — change them if you don't want the
# defaults exposed via GET /api/auth/demo-credentials.
AUTOML_DEMO_EMAIL=demo@automl.local
AUTOML_DEMO_PASSWORD=demo123
```

### New endpoints (`src/api/server.py`)

- `POST /api/auth/login` — body `{email, password}`. Case-insensitive email compare,
  exact password compare, against `os.environ.get("AUTOML_DEMO_EMAIL", "demo@automl.local")`
  / `os.environ.get("AUTOML_DEMO_PASSWORD", "demo123")`. On match: create a session,
  set it as an HttpOnly, `SameSite=Lax`, `path=/` cookie named `automl_session` with
  `max_age = session_ttl_hours * 3600`; return `{"ok": true}`. On mismatch: 401
  `{"detail": "invalid email or password"}`. `secure=False` on the cookie
  (deliberately — `run_server.py` serves plain HTTP on localhost; a `Secure` cookie
  would silently never be sent and break login).
- `POST /api/auth/logout` — destroys the session for the request's cookie (if any),
  clears the cookie, returns `{"ok": true}`. Always 200, even with no session.
- `GET /api/auth/session` — `{"authenticated": bool, "email": str | None}` from the
  request's cookie. Never raises; frontend polls this on load to decide whether to
  redirect.
- `GET /api/auth/demo-credentials` — unauthenticated, returns
  `{"email": <configured>, "password": <configured>}` so the login page can render a
  live, accurate hint instead of a string that can drift from `.env`. This
  intentionally exposes the demo password — acceptable because it's a demo credential
  for a single-user local tool, not a real secret.

### Route protection

- New dependency `require_session(request: Request) -> dict` — reads the
  `automl_session` cookie, calls `get_session()`, raises `HTTPException(401, "not
  authenticated")` if missing/expired, otherwise returns the session dict. Added via
  `Depends(require_session)` to all 12 existing `/api/runs*` route functions
  (`create_run`, `list_runs`, `get_run`, `confirm_run`, `approve_features`,
  `cancel_run`, `download_model`, `download_script`, `get_model_schema`, `predict`,
  `get_trace`, `chat`).
- `GET /` becomes an explicit route (defined before the static-files mount) that
  returns `FileResponse("frontend/index.html")` if `require_session` would succeed for
  the request, else `RedirectResponse("/login.html")`. `login.html` itself stays
  reachable unauthenticated via the existing static mount (no explicit route needed).

## Frontend

### `frontend/login.html` + `frontend/login.js` (new, standalone — not part of the SPA bundle)

Reuses `frontend/styles.css`'s existing design tokens (`--accent-primary`,
`--bg-surface`, `--radius`, etc.) and the existing dark/light theme toggle logic
(same `localStorage` key, `applyTheme()` pattern copied into `login.js` since this
page loads independently of `app.js`) — no new color families.

Layout, matching the reference image's chrome:
- Top nav bar: brand mark + "AutoML Agentic" wordmark (left), inert nav links
  "Product · Features · How it Works · Pricing · Docs · About" (center-left, rendered
  as non-interactive `<span>` elements wrapped in a `div` with `aria-hidden="true"`,
  since this app has no marketing subpages), and the light/dark theme toggle (right).
- Background: the same soft gradient/blob treatment used behind the image's hero.
- Centered card: "Welcome back" heading, one-line subtext, email field, password
  field, "Log in" button (primary, full-width), an inline error message region
  (hidden until a failed attempt — no `alert()`), and a small muted "Demo credentials"
  box populated from `GET /api/auth/demo-credentials` on page load.

Submit handler: `POST /api/auth/login`; on success `window.location.href = "/"`; on
401, show the inline error with the response's `detail` text and re-enable the button.

### `frontend/app.js` changes

- On script load (before any other initialization), `await fetch("/api/auth/session")`;
  if `!authenticated`, `window.location.href = "/login.html"` and stop.
- A thin wrapper `authFetch(url, options)` (used by every existing `fetch()` call site
  that hits `/api/runs*`) that redirects to `/login.html` on a 401 response instead of
  letting the caller's existing error handling show a confusing "Failed to..." alert
  for what's actually an expired session.
- Sidebar footer (`frontend/index.html`, next to the existing theme toggle / "Local
  build" chip) gains a "Log out" button: `POST /api/auth/logout` then redirect to
  `/login.html`.

## Data flow summary

```
GET /login.html          -> static, unauthenticated, always reachable
GET /                     -> 200 index.html (session valid) | 307 /login.html (no session)
POST /api/auth/login      -> sets automl_session cookie -> frontend redirects to /
GET  /api/auth/session    -> {authenticated, email} -> used by app.js's load-time guard
POST /api/auth/logout     -> clears cookie -> frontend redirects to /login.html
/api/runs*                -> 401 without automl_session -> authFetch() redirects to /login.html
```

## Testing

- `tests/test_auth.py` (new): login with correct credentials sets a cookie and
  `GET /api/auth/session` reflects it; wrong email/wrong password both 401 without
  setting a cookie; logout clears the session (subsequent `/api/auth/session` shows
  unauthenticated); a protected endpoint (`GET /api/runs`) 401s without a cookie and
  200s with one obtained via `/api/auth/login`; `GET /api/auth/demo-credentials`
  reflects `AUTOML_DEMO_EMAIL`/`AUTOML_DEMO_PASSWORD` env overrides via monkeypatch.
- `tests/conftest.py` (new): a session-scoped-safe, function-scoped **autouse**
  fixture that sets `server.app.dependency_overrides[server.require_session] = lambda:
  {"email": "test@local"}` before each test and clears it after — so all ~15 existing
  API test modules (`test_api_chat.py`, `test_api_run_listing.py`,
  `test_target_cardinality_guard.py`, `test_pipeline_smoke.py` if it hits the API,
  etc.) keep passing unauthenticated with zero per-file edits.
  `tests/test_auth.py` explicitly does
  `server.app.dependency_overrides.pop(server.require_session, None)` at the top of
  each test that needs the real dependency enforced, restoring it via a local
  fixture/teardown.
- `node --check frontend/login.js frontend/app.js`.
- Manual browser pass: load `/` with no session → redirected to `/login.html`; wrong
  credentials → inline error, no redirect; correct (demo-hint) credentials → redirect
  to home view; refresh mid-session → stays logged in; "Log out" → redirected back to
  `/login.html` and `/` now redirects again; both themes render correctly on the login
  page.

## Non-negotiables carried over

- No raw data / no new LLM-facing surface — this feature has no LLM involvement at
  all.
- No new color families in `login.js`/`styles.css` additions; reuse existing tokens.
- Session cookie is HttpOnly (not readable by JS) — `login.js` never reads/writes the
  cookie directly, only calls the three auth endpoints.
