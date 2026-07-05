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
