# Login Page + Session Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a branded login page (styled after the reference marketing-image chrome) gated in front of the existing home view/dashboard, backed by a minimal in-memory session mechanism — no user table, no graph/pipeline changes.

**Architecture:** A new `src/auth/session.py` module (same in-memory "local stand-in" pattern as `src/api/server.py`'s existing `_runs` registry) backs four new FastAPI endpoints (`/api/auth/login`, `/logout`, `/session`, `/demo-credentials`) and a `require_session` dependency applied to all 12 existing `/api/runs*` endpoints plus an explicit `GET /` route. A new standalone `frontend/login.html`/`login.js` (independent of `app.js`, since it must work with no session) handles the login form; `app.js` gains a load-time session check, an `authFetch` wrapper that redirects to `/login.html` on any 401, and a sidebar "Log out" button.

**Tech Stack:** FastAPI (existing `src/api/server.py`), Starlette cookie-based sessions (no new dependency), vanilla JS/CSS frontend (no build step), pytest + FastAPI `TestClient`.

**Spec:** `docs/superpowers/specs/2026-07-05-login-page-design.md`

## Global Constraints

- Zero changes to `src/graph/`, `src/agents/`, `src/training/`, `src/profiling/`, `src/insights/`, `src/llm/`, `src/sandbox/`, `src/pii/` — this is a self-contained auth layer, not a pipeline change.
- No real user accounts/password hashing/OAuth — a single demo credential from env vars (`AUTOML_DEMO_EMAIL`/`AUTOML_DEMO_PASSWORD`, defaulting to `demo@automl.local`/`demo123`), matching the app's existing single-local-user, in-memory-registry architecture.
- Session cookie: name `automl_session`, `HttpOnly`, `SameSite=Lax`, `secure=False` (deliberate — `run_server.py` serves plain HTTP on localhost), `path=/`, `max_age` from `config/runtime.yaml`'s new `auth.session_ttl_hours` (default 24).
- `GET /api/auth/demo-credentials` is intentionally unauthenticated and returns the real configured demo password — acceptable because it is a demo credential for a single-user local tool, not a secret.
- No new color families in any CSS addition — reuse existing tokens (`--accent-primary`, `--bg-surface`, `--border-subtle`, etc.).
- Existing test suites must keep passing unauthenticated with zero per-file edits (via an autouse dependency-override fixture in a new `tests/conftest.py`).
- Login page nav mirrors the reference image's chrome (logo, inert nav links, theme toggle) per the user's explicit "maintain the design of the reference image" instruction — nav links are non-interactive placeholders (no destinations exist in this app).

---

### Task 1: Session store

**Files:**
- Create: `src/auth/__init__.py` (empty)
- Create: `src/auth/session.py`
- Modify: `config/runtime.yaml` (append `auth` block)
- Test: `tests/test_session_store.py` (create)

**Interfaces:**
- Produces: `create_session(email: str, ttl_hours: float) -> str`, `get_session(token: str | None) -> dict[str, Any] | None`, `destroy_session(token: str | None) -> None` — consumed by Task 2's endpoints.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_session_store.py`:

```python
"""Pure unit tests for the in-memory session store (src/auth/session.py) —
same "local stand-in" pattern as src/api/server.py's _runs registry, no
Postgres/user table (see docs/superpowers/specs/2026-07-05-login-page-design.md).
No FastAPI involved here; src/api/server.py's auth endpoints are tested in
tests/test_auth.py."""

from __future__ import annotations

from src.auth import session as session_module


def test_create_and_get_session_round_trips_email():
    token = session_module.create_session("demo@automl.local", ttl_hours=1)

    result = session_module.get_session(token)

    assert result is not None
    assert result["email"] == "demo@automl.local"


def test_get_session_returns_none_for_unknown_token():
    assert session_module.get_session("not-a-real-token") is None


def test_get_session_returns_none_for_missing_token():
    assert session_module.get_session(None) is None


def test_session_expires_after_ttl(monkeypatch):
    fake_now = [1_000_000.0]
    monkeypatch.setattr(session_module.time, "time", lambda: fake_now[0])

    token = session_module.create_session("demo@automl.local", ttl_hours=1)
    assert session_module.get_session(token) is not None

    fake_now[0] += 3601  # just past the 1-hour TTL
    assert session_module.get_session(token) is None


def test_destroy_session_removes_it():
    token = session_module.create_session("demo@automl.local", ttl_hours=1)
    assert session_module.get_session(token) is not None

    session_module.destroy_session(token)

    assert session_module.get_session(token) is None


def test_destroy_session_is_a_no_op_for_missing_token():
    session_module.destroy_session(None)  # must not raise
    session_module.destroy_session("not-a-real-token")  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_session_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.auth'`

- [ ] **Step 3: Create the package and implement the store**

Create `src/auth/__init__.py` (empty file, 0 bytes — matches every other `src/` subpackage's convention).

Create `src/auth/session.py`:

```python
"""In-memory session store backing the demo login gate (src/api/server.py's
auth endpoints). Same "local stand-in" pattern as src/api/server.py's _runs
registry — a single-demo-user local build, not production multi-tenant auth.
See docs/superpowers/specs/2026-07-05-login-page-design.md."""

from __future__ import annotations

import time
import uuid
from typing import Any

_sessions: dict[str, dict[str, Any]] = {}


def create_session(email: str, ttl_hours: float) -> str:
    token = uuid.uuid4().hex
    _sessions[token] = {"email": email, "expires_at": time.time() + ttl_hours * 3600}
    return token


def get_session(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    entry = _sessions.get(token)
    if entry is None:
        return None
    if entry["expires_at"] < time.time():
        del _sessions[token]
        return None
    return entry


def destroy_session(token: str | None) -> None:
    if token:
        _sessions.pop(token, None)
```

- [ ] **Step 4: Add the runtime config block**

Open `config/runtime.yaml`. Find:

```yaml
budgets:
  max_llm_calls_per_run: 40
  max_tokens_per_run: 200000
```

Replace with:

```yaml
budgets:
  max_llm_calls_per_run: 40
  max_tokens_per_run: 200000

auth:
  session_ttl_hours: 24 # demo login session lifetime (src/auth/session.py)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_session_store.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add src/auth/__init__.py src/auth/session.py config/runtime.yaml tests/test_session_store.py
git commit -m "feat: add in-memory session store for the demo login gate"
```

---

### Task 2: Auth API endpoints

**Files:**
- Modify: `src/api/server.py`
- Modify: `.env.example`
- Test: `tests/test_auth.py` (create)

**Interfaces:**
- Consumes: `create_session`, `get_session`, `destroy_session` from Task 1's `src/auth/session.py`.
- Produces: `require_session(request: Request) -> dict[str, Any]` FastAPI dependency and `_get_session_from_request(request: Request) -> dict[str, Any] | None` helper — both consumed by Task 3 (route protection) and Task 4 (the `/` redirect route). `SESSION_COOKIE_NAME` constant — consumed by Task 3's tests.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_auth.py`:

```python
"""API-level tests for the login gate (src/auth/session.py's pure logic is
covered separately in tests/test_session_store.py). Uses the real
require_session dependency — tests/conftest.py's autouse override is
disabled here via the `real_auth` fixture so these hit actual 401s."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import server


@pytest.fixture()
def real_auth():
    """Every other test file gets require_session bypassed by
    tests/conftest.py's autouse fixture (added in Task 3); this file needs
    the real dependency enforced, so it removes that override for its
    duration and restores it afterward."""
    override = server.app.dependency_overrides.pop(server.require_session, None)
    yield
    if override is not None:
        server.app.dependency_overrides[server.require_session] = override


@pytest.fixture()
def client(real_auth, monkeypatch):
    monkeypatch.setenv("AUTOML_DEMO_EMAIL", "demo@automl.local")
    monkeypatch.setenv("AUTOML_DEMO_PASSWORD", "demo123")
    return TestClient(server.app)


def test_login_with_correct_credentials_sets_cookie_and_returns_ok(client):
    res = client.post("/api/auth/login", json={"email": "demo@automl.local", "password": "demo123"})

    assert res.status_code == 200
    assert res.json() == {"ok": True}
    assert server.SESSION_COOKIE_NAME in res.cookies


def test_login_with_wrong_password_is_401_and_sets_no_cookie(client):
    res = client.post("/api/auth/login", json={"email": "demo@automl.local", "password": "wrong"})

    assert res.status_code == 401
    assert server.SESSION_COOKIE_NAME not in res.cookies


def test_login_with_wrong_email_is_401(client):
    res = client.post("/api/auth/login", json={"email": "someone@else.com", "password": "demo123"})

    assert res.status_code == 401


def test_login_email_match_is_case_insensitive(client):
    res = client.post("/api/auth/login", json={"email": "DEMO@AUTOML.LOCAL", "password": "demo123"})

    assert res.status_code == 200


def test_session_endpoint_reflects_authentication_state(client):
    before = client.get("/api/auth/session").json()
    assert before == {"authenticated": False, "email": None}

    client.post("/api/auth/login", json={"email": "demo@automl.local", "password": "demo123"})

    after = client.get("/api/auth/session").json()
    assert after == {"authenticated": True, "email": "demo@automl.local"}


def test_logout_clears_the_session(client):
    client.post("/api/auth/login", json={"email": "demo@automl.local", "password": "demo123"})
    assert client.get("/api/auth/session").json()["authenticated"] is True

    logout_res = client.post("/api/auth/logout")

    assert logout_res.status_code == 200
    assert client.get("/api/auth/session").json()["authenticated"] is False


def test_demo_credentials_endpoint_reflects_env_overrides(client, monkeypatch):
    monkeypatch.setenv("AUTOML_DEMO_EMAIL", "custom@example.com")
    monkeypatch.setenv("AUTOML_DEMO_PASSWORD", "customPass1")

    res = client.get("/api/auth/demo-credentials")

    assert res.json() == {"email": "custom@example.com", "password": "customPass1"}


def test_demo_credentials_endpoint_has_defaults_when_env_unset(client, monkeypatch):
    monkeypatch.delenv("AUTOML_DEMO_EMAIL", raising=False)
    monkeypatch.delenv("AUTOML_DEMO_PASSWORD", raising=False)

    res = client.get("/api/auth/demo-credentials")

    assert res.json() == {"email": "demo@automl.local", "password": "demo123"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_auth.py -v`
Expected: FAIL — `AttributeError: module 'src.api.server' has no attribute 'SESSION_COOKIE_NAME'` (or a 404 on the new routes).

- [ ] **Step 3: Add imports and the auth block to `src/api/server.py`**

Find:

```python
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
```

Replace with:

```python
import os

import yaml
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
```

Find:

```python
from src.export.script_export import generate_training_script
```

Replace with:

```python
from src.auth.session import create_session, destroy_session, get_session
from src.export.script_export import generate_training_script
```

Find:

```python
_intake_graph = build_intake_graph()
_prep_graph = build_prep_graph()
_train_graph = build_train_graph()
```

Replace with:

```python
_intake_graph = build_intake_graph()
_prep_graph = build_prep_graph()
_train_graph = build_train_graph()

SESSION_COOKIE_NAME = "automl_session"


def _auth_config() -> dict[str, Any]:
    with open("config/runtime.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["auth"]


def _get_session_from_request(request: Request) -> dict[str, Any] | None:
    return get_session(request.cookies.get(SESSION_COOKIE_NAME))


def require_session(request: Request) -> dict[str, Any]:
    """FastAPI dependency: 401s any request without a valid, unexpired
    session cookie. Applied to every /api/runs* endpoint (Task 3) — this is
    a demo-credential gate for a single-user local tool, not production
    multi-tenant auth (see docs/superpowers/specs/2026-07-05-login-page-design.md)."""
    session = _get_session_from_request(request)
    if session is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return session


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/auth/login")
def login(body: LoginRequest, response: Response) -> dict[str, Any]:
    demo_email = os.environ.get("AUTOML_DEMO_EMAIL", "demo@automl.local")
    demo_password = os.environ.get("AUTOML_DEMO_PASSWORD", "demo123")
    if body.email.strip().lower() != demo_email.strip().lower() or body.password != demo_password:
        raise HTTPException(status_code=401, detail="invalid email or password")

    ttl_hours = _auth_config()["session_ttl_hours"]
    token = create_session(demo_email, ttl_hours)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=int(ttl_hours * 3600),
        httponly=True,
        samesite="lax",
        secure=False,  # local http dev (run_server.py); see design spec
        path="/",
    )
    return {"ok": True}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response) -> dict[str, Any]:
    destroy_session(request.cookies.get(SESSION_COOKIE_NAME))
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/auth/session")
def auth_session(request: Request) -> dict[str, Any]:
    session = _get_session_from_request(request)
    return {"authenticated": session is not None, "email": session["email"] if session else None}


@app.get("/api/auth/demo-credentials")
def demo_credentials() -> dict[str, Any]:
    """Unauthenticated by design — this is a demo credential for a
    single-user local tool, not a secret (see design spec)."""
    return {
        "email": os.environ.get("AUTOML_DEMO_EMAIL", "demo@automl.local"),
        "password": os.environ.get("AUTOML_DEMO_PASSWORD", "demo123"),
    }
```

- [ ] **Step 4: Add demo credential env vars to `.env.example`**

Find:

```
# Set to 1 to run the entire pipeline + web UI with canned LLM responses —
# no API keys or network needed. Great for local testing/demo.
AUTOML_MOCK_LLM=
```

Replace with:

```
# Set to 1 to run the entire pipeline + web UI with canned LLM responses —
# no API keys or network needed. Great for local testing/demo.
AUTOML_MOCK_LLM=

# Demo login credential for this single-user local build (see src/auth/).
# Defaults below are used if unset — change them if you don't want the
# defaults exposed via GET /api/auth/demo-credentials.
AUTOML_DEMO_EMAIL=demo@automl.local
AUTOML_DEMO_PASSWORD=demo123
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_auth.py -v`
Expected: PASS (8 passed)

- [ ] **Step 6: Commit**

```bash
git add src/api/server.py .env.example tests/test_auth.py
git commit -m "feat: add login/logout/session/demo-credentials auth endpoints"
```

---

### Task 3: Protect existing API endpoints + keep the existing test suite green

**Files:**
- Modify: `src/api/server.py` (12 endpoint signatures)
- Create: `tests/conftest.py`
- Test: append to `tests/test_auth.py`

**Interfaces:**
- Consumes: `require_session` from Task 2.
- Produces: none new — this task's contract is behavioral (401 without a session, unchanged behavior with one).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_auth.py`:

```python


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/api/runs"),
        ("GET", "/api/runs/nonexistent"),
        ("POST", "/api/runs/nonexistent/confirm"),
        ("POST", "/api/runs/nonexistent/approve-features"),
        ("POST", "/api/runs/nonexistent/cancel"),
        ("GET", "/api/runs/nonexistent/model"),
        ("GET", "/api/runs/nonexistent/script"),
        ("GET", "/api/runs/nonexistent/model/schema"),
        ("POST", "/api/runs/nonexistent/predict"),
        ("GET", "/api/runs/nonexistent/trace"),
        ("POST", "/api/runs/nonexistent/chat"),
    ],
)
def test_protected_endpoints_401_without_a_session(client, method, path):
    res = client.request(method, path, json={} if method == "POST" else None)

    assert res.status_code == 401


def test_protected_endpoint_200s_with_a_valid_session(client):
    client.post("/api/auth/login", json={"email": "demo@automl.local", "password": "demo123"})

    res = client.get("/api/runs")

    assert res.status_code == 200


def test_create_run_401s_without_a_session(client):
    import io

    res = client.post(
        "/api/runs",
        files={"file": ("data.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")},
        data={"description": "predict a"},
    )

    assert res.status_code == 401
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_auth.py -v`
Expected: FAIL — every parametrized case gets 404 (unknown run) instead of 401, since `require_session` isn't wired into these endpoints yet.

- [ ] **Step 3: Add `Depends(require_session)` to all 12 protected endpoints**

Open `src/api/server.py`. Make each of the following 12 find/replace edits.

Find:
```python
async def create_run(file: UploadFile = File(...), description: str = Form(...)) -> dict[str, Any]:
```
Replace:
```python
async def create_run(
    file: UploadFile = File(...), description: str = Form(...), _session: dict[str, Any] = Depends(require_session)
) -> dict[str, Any]:
```

Find:
```python
def list_runs() -> list[dict[str, Any]]:
```
Replace:
```python
def list_runs(_session: dict[str, Any] = Depends(require_session)) -> list[dict[str, Any]]:
```

Find:
```python
def get_run(run_id: str) -> dict[str, Any]:
```
Replace:
```python
def get_run(run_id: str, _session: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
```

Find:
```python
def confirm_run(run_id: str, body: ConfirmRequest) -> dict[str, Any]:
```
Replace:
```python
def confirm_run(
    run_id: str, body: ConfirmRequest, _session: dict[str, Any] = Depends(require_session)
) -> dict[str, Any]:
```

Find:
```python
def approve_features(run_id: str, body: FeatureApprovalRequest) -> dict[str, Any]:
```
Replace:
```python
def approve_features(
    run_id: str, body: FeatureApprovalRequest, _session: dict[str, Any] = Depends(require_session)
) -> dict[str, Any]:
```

Find:
```python
def cancel_run(run_id: str) -> dict[str, Any]:
```
Replace:
```python
def cancel_run(run_id: str, _session: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
```

Find:
```python
def download_model(run_id: str):
```
Replace:
```python
def download_model(run_id: str, _session: dict[str, Any] = Depends(require_session)):
```

Find:
```python
def download_script(run_id: str) -> Response:
```
Replace:
```python
def download_script(run_id: str, _session: dict[str, Any] = Depends(require_session)) -> Response:
```

Find:
```python
def get_model_schema(run_id: str) -> dict[str, Any]:
```
Replace:
```python
def get_model_schema(run_id: str, _session: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
```

Find:
```python
def predict(run_id: str, body: PredictRequest) -> dict[str, Any]:
```
Replace:
```python
def predict(
    run_id: str, body: PredictRequest, _session: dict[str, Any] = Depends(require_session)
) -> dict[str, Any]:
```

Find:
```python
def get_trace(run_id: str) -> list[dict[str, Any]]:
```
Replace:
```python
def get_trace(run_id: str, _session: dict[str, Any] = Depends(require_session)) -> list[dict[str, Any]]:
```

Find:
```python
def chat(run_id: str, body: ChatRequest) -> dict[str, Any]:
```
Replace:
```python
def chat(run_id: str, body: ChatRequest, _session: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
```

- [ ] **Step 4: Add the autouse override fixture so every OTHER existing test file stays unauthenticated**

Create `tests/conftest.py`:

```python
"""Repo-wide pytest fixtures.

Adding require_session (src/api/server.py) protected every /api/runs*
endpoint (see docs/superpowers/specs/2026-07-05-login-page-design.md). This
autouse fixture disables that dependency for every test EXCEPT
tests/test_auth.py, which explicitly restores the real dependency via its
own `real_auth` fixture — so the ~15 pre-existing API test modules keep
passing unauthenticated with zero per-file edits.
"""

from __future__ import annotations

import pytest

from src.api import server


@pytest.fixture(autouse=True)
def _bypass_auth_by_default():
    server.app.dependency_overrides[server.require_session] = lambda: {"email": "test@local"}
    yield
    server.app.dependency_overrides.pop(server.require_session, None)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_auth.py -v`
Expected: PASS (13 passed)

- [ ] **Step 6: Run the full backend suite to confirm zero regressions in existing API tests**

Run: `.venv/Scripts/python -m pytest tests/test_api_chat.py tests/test_api_run_listing.py tests/test_target_cardinality_guard.py -v`
Expected: all pass unchanged (the autouse fixture bypasses auth for these files).

- [ ] **Step 7: Commit**

```bash
git add src/api/server.py tests/conftest.py tests/test_auth.py
git commit -m "feat: protect all /api/runs endpoints behind the session gate"
```

---

### Task 4: Redirect guard on the app shell (`GET /`)

**Files:**
- Modify: `src/api/server.py`
- Test: append to `tests/test_auth.py`

**Interfaces:**
- Consumes: `_get_session_from_request` from Task 2.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_auth.py`:

```python


def test_root_redirects_to_login_without_a_session(client):
    res = client.get("/", follow_redirects=False)

    assert res.status_code == 307
    assert res.headers["location"] == "/login.html"


def test_root_serves_the_app_with_a_valid_session(client):
    client.post("/api/auth/login", json={"email": "demo@automl.local", "password": "demo123"})

    res = client.get("/", follow_redirects=False)

    assert res.status_code == 200
    assert "Agentic AutoML" in res.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_auth.py -k test_root -v`
Expected: FAIL — without an explicit route, `GET /` currently falls through to the static mount and serves `index.html` unconditionally (200 in both cases), so `test_root_redirects_to_login_without_a_session` fails.

- [ ] **Step 3: Add the explicit `/` route before the static mount**

Open `src/api/server.py`. Find:

```python
# Serve the frontend last so /api/* wins routing.
app.mount("/", NoCacheStaticFiles(directory="frontend", html=True), name="frontend")
```

Replace with:

```python
@app.get("/")
def serve_index(request: Request):
    """The one static path that needs server-side auth enforcement — every
    other static asset (styles.css, app.js, login.html/login.js) stays
    reachable unauthenticated via the mount below, but the app shell itself
    should redirect to the login page rather than flash stale/empty UI."""
    if _get_session_from_request(request) is None:
        return RedirectResponse("/login.html")
    return FileResponse("frontend/index.html")


# Serve the frontend last so /api/* wins routing.
app.mount("/", NoCacheStaticFiles(directory="frontend", html=True), name="frontend")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_auth.py -v`
Expected: PASS (15 passed)

- [ ] **Step 5: Commit**

```bash
git add src/api/server.py tests/test_auth.py
git commit -m "feat: redirect GET / to the login page without a valid session"
```

---

### Task 5: Login page frontend (HTML/JS/CSS)

**Files:**
- Create: `frontend/login.html`
- Create: `frontend/login.js`
- Modify: `frontend/styles.css`

**Interfaces:**
- Consumes: `POST /api/auth/login`, `GET /api/auth/demo-credentials` (Task 2).

- [ ] **Step 1: Create `frontend/login.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Log in — Agentic AutoML</title>
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Ctext y='18' font-size='20'%3E%E2%AC%A2%3C/text%3E%3C/svg%3E" />
  <link rel="stylesheet" href="/styles.css" />
</head>
<body>
  <div class="login-page">
    <nav class="login-nav">
      <div class="login-nav-brand">
        <span class="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2 21 7.2v9.6L12 22 3 16.8V7.2L12 2Z"/></svg>
        </span>
        <span class="login-nav-name">AutoML Agentic</span>
      </div>
      <div class="login-nav-links" aria-hidden="true">
        <span class="login-nav-link">Product</span>
        <span class="login-nav-link">Features</span>
        <span class="login-nav-link">How it Works</span>
        <span class="login-nav-link">Pricing</span>
        <span class="login-nav-link">Docs</span>
        <span class="login-nav-link">About</span>
      </div>
      <button class="login-theme-toggle" id="theme-toggle" type="button" aria-pressed="false">
        <svg class="icon icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8Z"/></svg>
        <svg class="icon icon-sun hidden" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
      </button>
    </nav>

    <main class="login-hero">
      <section class="login-card">
        <div class="hero-eyebrow" aria-hidden="true">
          <span>AI-Powered</span><span class="hero-eyebrow-dot"></span>
          <span>Agentic</span><span class="hero-eyebrow-dot"></span>
          <span>Automated</span>
        </div>
        <h1 class="login-title">Welcome back.</h1>
        <p class="muted login-sub">Log in to continue building better models, automatically.</p>

        <form id="login-form" class="login-form">
          <label class="field">
            <span>Email</span>
            <input type="email" id="login-email" required autocomplete="username" placeholder="you@company.com" />
          </label>
          <label class="field">
            <span>Password</span>
            <input type="password" id="login-password" required autocomplete="current-password" placeholder="••••••••" />
          </label>
          <p class="login-error hidden" id="login-error" role="alert"></p>
          <div class="btn-row">
            <button type="submit" class="btn primary login-submit" id="login-submit">Log in</button>
          </div>
        </form>

        <p class="login-demo-hint muted small" id="login-demo-hint">Loading demo credentials…</p>
      </section>
    </main>
  </div>

<script src="/login.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `frontend/login.js`**

```javascript
/* Standalone login page script — loads independently of app.js, since this
   page must work with no session at all. Theme toggle mirrors app.js's
   exactly (same localStorage key) so the choice carries over once the user
   reaches the home view. */

const $ = (id) => document.getElementById(id);

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("automl-theme", theme);
  const isDark = theme === "dark";
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
```

- [ ] **Step 3: Add login page styles to `frontend/styles.css`**

Append at the end of the file:

```css

/* ================= login page ================= */

.login-page { min-height: 100vh; display: flex; flex-direction: column; }

.login-nav {
  display: flex; align-items: center; justify-content: space-between; gap: var(--sp-3);
  padding: var(--sp-3) var(--sp-5);
}
.login-nav-brand { display: flex; align-items: center; gap: 10px; color: var(--text-primary); }
.login-nav-name { font-family: var(--font-display); font-weight: 700; font-size: var(--text-base); }
.login-nav-links { display: flex; align-items: center; gap: var(--sp-4); }
.login-nav-link { font-size: var(--text-sm); font-weight: 550; color: var(--text-secondary); cursor: default; }
@media (max-width: 780px) { .login-nav-links { display: none; } }

.login-theme-toggle {
  width: 36px; height: 36px; border-radius: 999px; flex-shrink: 0;
  display: inline-flex; align-items: center; justify-content: center;
  background: var(--bg-surface); border: 1px solid var(--border-subtle);
  color: var(--text-secondary); cursor: pointer;
}
.login-theme-toggle:hover { color: var(--accent-primary); border-color: var(--accent-primary); }

.login-hero {
  flex: 1; display: flex; align-items: center; justify-content: center;
  padding: var(--sp-5) var(--sp-3);
  background:
    radial-gradient(600px circle at 15% 10%, var(--accent-primary-soft), transparent 60%),
    radial-gradient(600px circle at 85% 90%, var(--accent-primary-soft), transparent 60%),
    var(--bg-base);
}

.login-card {
  width: 100%; max-width: 420px;
  background: var(--bg-surface); border: 1px solid var(--border-subtle);
  border-radius: var(--radius); box-shadow: var(--shadow);
  padding: var(--sp-5); text-align: center;
}
.login-title { font-size: var(--text-xl); margin-top: var(--sp-3); }
.login-sub { font-size: var(--text-sm); margin: 4px 0 var(--sp-4); }
.login-form { display: grid; gap: var(--sp-3); text-align: left; }
.login-submit { width: 100%; justify-content: center; }
.login-error { color: var(--accent-danger); font-size: var(--text-xs); margin: 0; }
.login-demo-hint {
  margin-top: var(--sp-4); padding: var(--sp-2) var(--sp-3);
  background: var(--bg-surface-raised); border: 1px dashed var(--border-subtle);
  border-radius: var(--radius-sm);
}
```

- [ ] **Step 4: Syntax-check the JS**

Run: `node --check frontend/login.js`
Expected: no output (success)

- [ ] **Step 5: Commit**

```bash
git add frontend/login.html frontend/login.js frontend/styles.css
git commit -m "feat: add branded login page"
```

---

### Task 6: Frontend session guard + logout

**Files:**
- Modify: `frontend/app.js`
- Modify: `frontend/index.html`

**Interfaces:**
- Consumes: `GET /api/auth/session`, `POST /api/auth/logout` (Task 2); every existing `/api/runs*` fetch call site in `app.js`.

- [ ] **Step 1: Replace every existing `fetch(` call with `authFetch(` (11 call sites, all to `/api/runs*`)**

Do this now, before Step 2 adds the `authFetch` function itself — at this point in the file, the literal substring `fetch(` appears in exactly 11 places (the calls to `/api/runs`, `/api/runs/{id}`, `/api/runs/{id}/cancel`, `/api/runs/{id}/confirm`, `/api/runs/{id}/approve-features`, `/api/runs/{id}/chat`, `/api/runs/{id}/trace`, `/api/runs/{id}/model/schema`, `/api/runs/{id}/predict`, plus the upload and re-fetch-current-run calls) and nowhere else — the comment `// re-fetch: chat_history...` contains "re-fetch:" with no open-paren immediately after, so it does not match. Doing this replacement *before* Step 2 also sidesteps any self-reference: once Step 2 adds a function literally named `authFetch`, its own definition line and its internal use of the real browser fetch (written as `window.fetch(...)`, never bare `fetch(...)`) will not contain the substring `fetch(` in lowercase-preceded-by-nothing-else form... to keep this unambiguous, perform the substring replacement as a single edit with these exact parameters:

- `old_string`: `fetch(`
- `new_string`: `authFetch(`
- `replace_all`: `true`

Applied to `frontend/app.js` in its current state (before Step 2's insertion). Verify afterward that grepping the file for the literal string `fetch(` (case-sensitive) returns 11 matches, all reading `authFetch(`.

- [ ] **Step 2: Add the `authFetch` wrapper and load-time session guard**

Find:

```javascript
const $ = (id) => document.getElementById(id);
let pollTimer = null;
```

Replace with:

```javascript
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
```

- [ ] **Step 3: Add the "Log out" button to the sidebar**

In `frontend/index.html`, find:

```html
    <div class="sidebar-footer">
      <button class="nav-item" id="theme-toggle" type="button" aria-pressed="false">
        <svg class="icon icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8Z"/></svg>
        <svg class="icon icon-sun hidden" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
        <span id="theme-label">Dark mode</span>
      </button>
      <div class="mode-chip" title="Everything runs on this machine. Swap providers in config/models.yaml">Local build</div>
    </div>
```

Replace with:

```html
    <div class="sidebar-footer">
      <button class="nav-item" id="theme-toggle" type="button" aria-pressed="false">
        <svg class="icon icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8Z"/></svg>
        <svg class="icon icon-sun hidden" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
        <span id="theme-label">Dark mode</span>
      </button>
      <button class="nav-item" id="logout-btn" type="button">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="M16 17l5-5-5-5"/><path d="M21 12H9"/></svg>
        Log out
      </button>
      <div class="mode-chip" title="Everything runs on this machine. Swap providers in config/models.yaml">Local build</div>
    </div>
```

- [ ] **Step 4: Wire the logout button in `app.js`**

Find (the last two lines of the file):

```javascript
loadRecentRuns();
// independent of the per-run poll loop, so the sidebar stays eventually
// consistent regardless of which trigger points did or didn't fire
setInterval(loadRecentRuns, 4000);
```

Replace with:

```javascript
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
```

- [ ] **Step 5: Syntax-check the JS**

Run: `node --check frontend/app.js`
Expected: no output (success)

- [ ] **Step 6: Commit**

```bash
git add frontend/app.js frontend/index.html
git commit -m "feat: guard the app shell with a session check and add logout"
```

---

### Task 7: Full regression + manual browser verification

**Files:** none (verification only; spec update if reality diverged)

- [ ] **Step 1: Run the full backend test suite**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: all tests pass, no regressions in any pre-existing module.

- [ ] **Step 2: Manual browser pass**

Start the server (PowerShell): `$env:AUTOML_MOCK_LLM = "1"; .venv/Scripts/python run_server.py` (background), then:

1. Open `http://127.0.0.1:8000/` with no prior session (clear cookies first) — confirm it redirects to `/login.html`, showing the nav (logo, inert links, theme toggle) and centered card with a working demo-credential hint.
2. Submit wrong credentials — confirm an inline error appears, no redirect, button re-enables.
3. Submit the displayed demo credentials — confirm redirect to `/` and the home view loads normally.
4. Refresh the page — confirm you stay logged in (session persists).
5. Click "Log out" in the sidebar — confirm redirect back to `/login.html`, and that navigating to `/` again redirects back to login (not the app).
6. Toggle dark mode on the login page — confirm it persists into the home view after logging in (shared `localStorage` key).
7. With dev tools open, delete the `automl_session` cookie while the dashboard is open, then trigger any action that calls the API (e.g. open a run) — confirm it redirects to `/login.html` rather than showing a broken/blank panel.

- [ ] **Step 3: Update the design spec if anything diverged, then commit**

If any implementation detail diverged from `docs/superpowers/specs/2026-07-05-login-page-design.md`, update the spec to match reality.

```bash
git add -A
git commit -m "chore: final cleanup for login page feature"
```

(Skip this commit if there is nothing to add.)
