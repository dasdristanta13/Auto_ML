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
