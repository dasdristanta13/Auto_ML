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


def test_root_redirects_to_login_without_a_session(client):
    res = client.get("/", follow_redirects=False)

    assert res.status_code == 307
    assert res.headers["location"] == "/login.html"


def test_root_serves_the_app_with_a_valid_session(client):
    client.post("/api/auth/login", json={"email": "demo@automl.local", "password": "demo123"})

    res = client.get("/", follow_redirects=False)

    assert res.status_code == 200
    assert "Agentic AutoML" in res.text
