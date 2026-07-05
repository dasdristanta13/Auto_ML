"""answer_chat_question is NOT a StateGraph node (see
docs/superpowers/specs/2026-07-05-ai-assistant-chat-design.md) — it's called
directly by the chat API endpoint. These tests monkeypatch LLMClient.generate
the same way tests/test_pipeline_smoke.py does, so no API keys/network are
needed."""

from __future__ import annotations

from src.agents.chat_node import answer_chat_question
from src.llm.client import LLMClient


def test_answer_chat_question_calls_llm_with_question_as_user_prompt(monkeypatch):
    captured = {}

    def _fake_generate(self, run_id, node, system_prompt, user_prompt, json_schema=None, retries=1):
        captured.update(
            run_id=run_id, node=node, system_prompt=system_prompt,
            user_prompt=user_prompt, json_schema=json_schema,
        )
        return "the answer"

    monkeypatch.setattr(LLMClient, "generate", _fake_generate)

    result = answer_chat_question(
        run_id="r1",
        context={"task_spec": {"metric": "f1"}},
        history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        question="why is tenure important?",
    )

    assert result == "the answer"
    assert captured["run_id"] == "r1"
    assert captured["node"] == "chat"
    assert captured["user_prompt"] == "why is tenure important?"
    assert captured["json_schema"] is None
    assert "f1" in captured["system_prompt"]


def test_answer_chat_question_trims_history_to_last_three_exchanges(monkeypatch):
    captured = {}

    def _fake_generate(self, run_id, node, system_prompt, user_prompt, json_schema=None, retries=1):
        captured["system_prompt"] = system_prompt
        return "ok"

    monkeypatch.setattr(LLMClient, "generate", _fake_generate)

    long_history = [{"role": "user", "content": f"question {i}"} for i in range(10)]
    answer_chat_question(run_id="r1", context={}, history=long_history, question="latest?")

    # last 6 messages of a 10-message list are indices 4..9
    assert "question 9" in captured["system_prompt"]
    assert "question 4" in captured["system_prompt"]
    assert "question 3" not in captured["system_prompt"]
    assert "question 0" not in captured["system_prompt"]
