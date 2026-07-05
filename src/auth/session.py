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
