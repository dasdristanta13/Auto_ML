"""LLM-backed, on-demand Q&A about ONE completed run's already-computed
results. NOT a StateGraph node — see
docs/superpowers/specs/2026-07-05-ai-assistant-chat-design.md. Invoked
directly by src/api/server.py's chat endpoint, only after the run's report
is ready. Only ever sees the same already-redacted, already-computed data
the report/frontend already show; no raw dataset access, no tools."""

from __future__ import annotations

from typing import Any

from src.agents.prompt_utils import render_prompt
from src.llm.client import get_llm_client

_MAX_HISTORY_MESSAGES = 6  # last 3 exchanges, so prompt size stays bounded


def answer_chat_question(
    run_id: str,
    context: dict[str, Any],
    history: list[dict[str, str]],
    question: str,
) -> str:
    system_prompt = render_prompt(
        "chat.md",
        RUN_CONTEXT_JSON=context,
        CHAT_HISTORY_JSON=history[-_MAX_HISTORY_MESSAGES:],
    )
    return get_llm_client().generate(
        run_id=run_id,
        node="chat",
        system_prompt=system_prompt,
        user_prompt=question,
        json_schema=None,
    )
