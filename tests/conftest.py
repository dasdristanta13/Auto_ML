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
