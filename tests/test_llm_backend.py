"""Native vs litellm backend switch in src/llm/client.py
(docs/superpowers/specs/2026-07-07-litellm-backend-design.md)."""

from __future__ import annotations

import pytest

from src.llm import client as llm_client


BACKEND_CFG = {
    "backend": "native",
    "active_profile": "openai",
    "profiles": {
        "openai": {"provider": "openai", "model": "gpt-5-nano"},
        "bedrock_claude": {
            "provider": "bedrock",
            "model": "anthropic.claude-3-haiku-20240307-v1:0",
            "fallback_models": ["openai/gpt-5-nano"],
        },
    },
    "default": {"temperature": 0.0, "max_tokens": 4096},
    "nodes": {},
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("AUTOML_LLM_PROFILE", "AUTOML_LLM_MODEL", "AUTOML_LLM_PROVIDER", "AUTOML_LLM_BACKEND"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def backend_config(monkeypatch):
    monkeypatch.setattr(llm_client, "_models_config", lambda: BACKEND_CFG)


def test_backend_defaults_to_native(backend_config):
    cfg = llm_client.node_model_config("chat")
    assert cfg["backend"] == "native"
    assert cfg["fallback_models"] == []


def test_backend_env_var_overrides_yaml(backend_config, monkeypatch):
    monkeypatch.setenv("AUTOML_LLM_BACKEND", "litellm")
    cfg = llm_client.node_model_config("chat")
    assert cfg["backend"] == "litellm"


def test_unknown_backend_raises(backend_config, monkeypatch):
    monkeypatch.setenv("AUTOML_LLM_BACKEND", "sagemaker")
    with pytest.raises(ValueError, match="sagemaker"):
        llm_client.node_model_config("chat")


def test_profile_fallback_models_surfaced(backend_config, monkeypatch):
    monkeypatch.setenv("AUTOML_LLM_PROFILE", "bedrock_claude")
    cfg = llm_client.node_model_config("chat")
    assert cfg["provider"] == "bedrock"
    assert cfg["model"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert cfg["fallback_models"] == ["openai/gpt-5-nano"]


def test_legacy_schema_without_profiles_has_empty_fallback_models(monkeypatch):
    legacy_cfg = {
        "default": {"provider": "openai", "model": "gpt-5-nano", "temperature": 0.0, "max_tokens": 4096},
        "nodes": {},
    }
    monkeypatch.setattr(llm_client, "_models_config", lambda: legacy_cfg)
    cfg = llm_client.node_model_config("chat")
    assert cfg["backend"] == "native"
    assert cfg["fallback_models"] == []


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletionResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeLiteLLM:
    def __init__(self, content="hello from litellm", cost=0.0012, cost_error=False):
        self.content = content
        self.cost = cost
        self.cost_error = cost_error
        self.last_completion_kwargs = None

    def completion(self, **kwargs):
        self.last_completion_kwargs = kwargs
        return _FakeCompletionResponse(self.content)

    def completion_cost(self, completion_response):
        if self.cost_error:
            raise ValueError("no cost data for this model")
        return self.cost


@pytest.fixture()
def fake_litellm(monkeypatch):
    fake = _FakeLiteLLM()
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    return fake


def test_call_litellm_builds_provider_model_string(fake_litellm):
    content, cost = llm_client._call_litellm(
        "sys", "user", "bedrock", "anthropic.claude-3-haiku-20240307-v1:0",
        0.0, 2048, json_mode=False, fallback_models=[],
    )
    assert content == "hello from litellm"
    assert cost == 0.0012
    assert fake_litellm.last_completion_kwargs["model"] == "bedrock/anthropic.claude-3-haiku-20240307-v1:0"


def test_call_litellm_passes_fallbacks(fake_litellm):
    llm_client._call_litellm(
        "sys", "user", "openai", "gpt-5-nano",
        0.0, 2048, json_mode=False, fallback_models=["anthropic/claude-haiku-4-5"],
    )
    assert fake_litellm.last_completion_kwargs["fallbacks"] == ["anthropic/claude-haiku-4-5"]


def test_call_litellm_json_mode_for_openai(fake_litellm):
    llm_client._call_litellm(
        "sys", "user", "openai", "gpt-5-nano",
        0.0, 2048, json_mode=True, fallback_models=[],
    )
    assert fake_litellm.last_completion_kwargs["response_format"] == {"type": "json_object"}


def test_call_litellm_no_json_mode_for_anthropic(fake_litellm):
    llm_client._call_litellm(
        "sys", "user", "anthropic", "claude-haiku-4-5",
        0.0, 2048, json_mode=True, fallback_models=[],
    )
    assert "response_format" not in fake_litellm.last_completion_kwargs


def test_call_litellm_cost_none_when_completion_cost_fails(monkeypatch):
    fake = _FakeLiteLLM(cost_error=True)
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    content, cost = llm_client._call_litellm(
        "sys", "user", "openai", "gpt-5-nano",
        0.0, 2048, json_mode=False, fallback_models=[],
    )
    assert content == "hello from litellm"
    assert cost is None


@pytest.fixture()
def generate_cfg(monkeypatch):
    cfg = {
        "backend": "native",
        "active_profile": "openai",
        "profiles": {"openai": {"provider": "openai", "model": "gpt-5-nano"}},
        "default": {"temperature": 0.0, "max_tokens": 4096},
        "nodes": {},
    }
    monkeypatch.setattr(llm_client, "_models_config", lambda: cfg)
    monkeypatch.setattr(
        llm_client, "_runtime_config", lambda: {"budgets": {"max_llm_calls_per_run": 100}}
    )
    return cfg


def test_generate_native_backend_calls_call_openai(generate_cfg, monkeypatch):
    calls = {}

    def fake_call_openai(system, user, model, temperature, max_tokens, json_mode):
        calls["args"] = (system, user, model, temperature, max_tokens, json_mode)
        return "plain text response"

    monkeypatch.setattr(llm_client, "_call_openai", fake_call_openai)
    client = llm_client.LLMClient()
    result = client.generate("run-1", "chat", "system prompt", "user prompt")

    assert result == "plain text response"
    assert calls["args"][2] == "gpt-5-nano"


def test_generate_litellm_backend_calls_call_litellm(generate_cfg, monkeypatch):
    generate_cfg["backend"] = "litellm"
    calls = {}

    def fake_call_litellm(system, user, provider, model, temperature, max_tokens, json_mode, fallback_models):
        calls["args"] = (provider, model, fallback_models)
        return "litellm response", 0.002

    monkeypatch.setattr(llm_client, "_call_litellm", fake_call_litellm)
    client = llm_client.LLMClient()
    result = client.generate("run-2", "chat", "system prompt", "user prompt")

    assert result == "litellm response"
    assert calls["args"] == ("openai", "gpt-5-nano", [])


def test_generate_litellm_cost_reaches_trace_log(generate_cfg, monkeypatch):
    monkeypatch.setattr(
        llm_client, "_call_litellm",
        lambda *a, **k: ("litellm response", 0.0037),
    )
    generate_cfg["backend"] = "litellm"

    logged = {}

    def fake_log_llm_call(run_id, node, provider, model, system_prompt, user_prompt, response, error=None, **extra):
        logged["extra"] = extra

    monkeypatch.setattr(llm_client, "log_llm_call", fake_log_llm_call)
    client = llm_client.LLMClient()
    client.generate("run-3", "chat", "system prompt", "user prompt")

    assert logged["extra"]["cost_usd"] == 0.0037
