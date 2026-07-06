"""Provider-profile resolution in src/llm/client.node_model_config
(docs/superpowers/specs/2026-07-06-llm-provider-profiles-design.md):
one-line/env switching between anthropic, openai, and gemini profiles,
per-node model tiers, env escape hatches, and legacy-schema compatibility."""

from __future__ import annotations

import pytest

from src.llm import client as llm_client


PROFILE_CFG = {
    "active_profile": "openai",
    "profiles": {
        "openai": {"provider": "openai", "model": "gpt-5-nano"},
        "anthropic": {
            "provider": "anthropic",
            "model": "claude-haiku-4-5",
            "nodes": {"report": {"model": "claude-opus-4-8"}},
        },
        "gemini": {"provider": "gemini", "model": "gemini-2.5-flash"},
    },
    "default": {"temperature": 0.0, "max_tokens": 4096},
    "nodes": {"report": {"temperature": 0.2}, "chat": {"max_tokens": 1024}},
}

LEGACY_CFG = {
    "default": {"provider": "openai", "model": "gpt-5-nano", "temperature": 0.0, "max_tokens": 4096},
    "nodes": {"report": {"provider": "gemini", "model": "gemini-2.5-flash", "temperature": 0.2}},
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("AUTOML_LLM_PROFILE", "AUTOML_LLM_MODEL", "AUTOML_LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def profile_config(monkeypatch):
    monkeypatch.setattr(llm_client, "_models_config", lambda: PROFILE_CFG)


def test_active_profile_from_yaml(profile_config):
    cfg = llm_client.node_model_config("chat")
    assert cfg["provider"] == "openai"
    assert cfg["model"] == "gpt-5-nano"
    assert cfg["max_tokens"] == 1024  # node generation params still apply


def test_env_profile_overrides_yaml(profile_config, monkeypatch):
    monkeypatch.setenv("AUTOML_LLM_PROFILE", "anthropic")
    cfg = llm_client.node_model_config("chat")
    assert cfg["provider"] == "anthropic"
    assert cfg["model"] == "claude-haiku-4-5"


def test_profile_per_node_model_tier(profile_config, monkeypatch):
    monkeypatch.setenv("AUTOML_LLM_PROFILE", "anthropic")
    cfg = llm_client.node_model_config("report")
    assert cfg["model"] == "claude-opus-4-8"  # stronger tier for the report node
    assert cfg["temperature"] == 0.2  # generation params unaffected by profile


def test_gemini_profile(profile_config, monkeypatch):
    monkeypatch.setenv("AUTOML_LLM_PROFILE", "gemini")
    cfg = llm_client.node_model_config("report")
    assert (cfg["provider"], cfg["model"]) == ("gemini", "gemini-2.5-flash")


def test_unknown_profile_raises_with_available_names(profile_config, monkeypatch):
    monkeypatch.setenv("AUTOML_LLM_PROFILE", "nope")
    with pytest.raises(ValueError, match="anthropic"):
        llm_client.node_model_config("chat")


def test_model_and_provider_env_escape_hatches(profile_config, monkeypatch):
    monkeypatch.setenv("AUTOML_LLM_MODEL", "gpt-5-mini")
    cfg = llm_client.node_model_config("report")
    assert cfg["model"] == "gpt-5-mini"
    assert cfg["provider"] == "openai"  # profile provider untouched

    monkeypatch.setenv("AUTOML_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("AUTOML_LLM_MODEL", "claude-opus-4-8")
    cfg = llm_client.node_model_config("report")
    assert (cfg["provider"], cfg["model"]) == ("anthropic", "claude-opus-4-8")


def test_legacy_schema_without_profiles_still_resolves(monkeypatch):
    monkeypatch.setattr(llm_client, "_models_config", lambda: LEGACY_CFG)
    assert llm_client.node_model_config("chat")["provider"] == "openai"
    report = llm_client.node_model_config("report")
    assert (report["provider"], report["model"]) == ("gemini", "gemini-2.5-flash")


def test_shipped_models_yaml_resolves_every_node(monkeypatch):
    """The real config file must resolve for every known node under every
    profile — catches yaml typos before they surface mid-pipeline."""
    llm_client._models_config.cache_clear()
    for profile in ("openai", "anthropic", "gemini"):
        monkeypatch.setenv("AUTOML_LLM_PROFILE", profile)
        for node in ("understand_usecase", "feature_engineering", "model_selection", "report", "chat"):
            cfg = llm_client.node_model_config(node)
            assert cfg["provider"] and cfg["model"]
            assert "max_tokens" in cfg and "temperature" in cfg
