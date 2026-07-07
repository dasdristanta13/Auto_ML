# litellm Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `src/llm/client.py` route every LLM call through either its existing native per-provider adapters or through litellm, switchable with one config key / env var, with litellm calls getting cost tracking and per-profile fallback lists.

**Architecture:** A new `backend: native | litellm` key in `config/models.yaml` (env override `AUTOML_LLM_BACKEND`) is resolved into `node_model_config()`'s output alongside the existing provider/model/temperature/max_tokens. `LLMClient.generate()` branches once per attempt on that resolved backend: native keeps calling `_call_anthropic`/`_call_openai`/`_call_gemini` exactly as today; litellm calls a new `_call_litellm()` that builds a `"<provider>/<model>"` string for `litellm.completion()`, which is what lets any litellm-supported provider work with zero new adapter code.

**Tech Stack:** Python 3.11+, litellm (new optional dependency), pytest with `monkeypatch`.

## Global Constraints

- Do not hardcode model strings — read from `config/models.yaml` (CLAUDE.md).
- Never block an LLM call on a training job — not touched by this change, but don't introduce new synchronous blocking calls.
- Log full agent reasoning traces for every LLM call, including provider/model (CLAUDE.md rule #7) — the new litellm path must keep calling `log_llm_call` for every attempt, success or failure.
- `backend` default is `native` — existing deployments with no `backend` key in `models.yaml` must behave exactly as before.
- litellm is a lazy import (like `anthropic`/`openai`/`google.generativeai` today) — `native`-only users must not need it installed.

---

### Task 1: Resolve `backend` and `fallback_models` in `node_model_config`

**Files:**
- Modify: `src/llm/client.py:51-97` (`_active_profile`, `node_model_config`)
- Test: `tests/test_llm_backend.py` (new file)

**Interfaces:**
- Produces: `_effective_backend(cfg: dict[str, Any]) -> str` — raises `ValueError` on anything other than `"native"`/`"litellm"`.
- Produces: `node_model_config(node)` now also returns `"backend"` (`str`) and `"fallback_models"` (`list[str]`, possibly empty) keys, alongside the existing `"provider"`/`"model"`/`"temperature"`/`"max_tokens"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_backend.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_backend.py -v`
Expected: FAIL — `KeyError: 'backend'` (and similar) since `node_model_config` doesn't produce these keys yet.

- [ ] **Step 3: Implement `_effective_backend` and wire it into `node_model_config`**

In `src/llm/client.py`, add this function directly above `def node_model_config(node: str)`:

```python
def _effective_backend(cfg: dict[str, Any]) -> str:
    """Resolve native vs litellm execution backend: AUTOML_LLM_BACKEND env
    var wins, then models.yaml's top-level `backend` key (default
    "native"). Read per call, like the other env escape hatches, so
    switching needs no yaml edit."""
    backend = os.environ.get("AUTOML_LLM_BACKEND") or cfg.get("backend", "native")
    if backend not in ("native", "litellm"):
        raise ValueError(f"unknown LLM backend '{backend}' (expected 'native' or 'litellm')")
    return backend
```

Then, inside `node_model_config`, replace:

```python
    if os.environ.get("AUTOML_LLM_PROVIDER"):
        merged["provider"] = os.environ["AUTOML_LLM_PROVIDER"]
    if os.environ.get("AUTOML_LLM_MODEL"):
        merged["model"] = os.environ["AUTOML_LLM_MODEL"]

    if "provider" not in merged or "model" not in merged:
        raise ValueError(f"No provider/model configured for node '{node}' in config/models.yaml")
    return merged
```

with:

```python
    if os.environ.get("AUTOML_LLM_PROVIDER"):
        merged["provider"] = os.environ["AUTOML_LLM_PROVIDER"]
    if os.environ.get("AUTOML_LLM_MODEL"):
        merged["model"] = os.environ["AUTOML_LLM_MODEL"]

    if "provider" not in merged or "model" not in merged:
        raise ValueError(f"No provider/model configured for node '{node}' in config/models.yaml")

    merged["backend"] = _effective_backend(cfg)
    merged["fallback_models"] = list((profile or {}).get("fallback_models", []))
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_backend.py tests/test_model_profiles.py -v`
Expected: PASS (all of `test_llm_backend.py`, and `test_model_profiles.py` still green — it never inspects `backend`/`fallback_models` so the new keys don't break its assertions).

- [ ] **Step 5: Commit**

```bash
git add src/llm/client.py tests/test_llm_backend.py
git commit -m "feat: resolve native/litellm backend and fallback_models in node_model_config"
```

---

### Task 2: `_call_litellm` adapter

**Files:**
- Modify: `src/llm/client.py` (add after `_call_gemini`, currently ending at line 276)
- Test: `tests/test_llm_backend.py` (append)

**Interfaces:**
- Consumes: nothing new from Task 1 directly (it's a standalone adapter function, parallel to `_call_anthropic`/`_call_openai`/`_call_gemini`).
- Produces: `_call_litellm(system: str, user: str, provider: str, model: str, temperature: float, max_tokens: int, json_mode: bool, fallback_models: list[str]) -> tuple[str, Optional[float]]` — returns `(content, cost_or_none)`. Task 3 calls this exactly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm_backend.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_backend.py -v -k call_litellm`
Expected: FAIL with `AttributeError: module 'src.llm.client' has no attribute '_call_litellm'`.

- [ ] **Step 3: Implement `_call_litellm`**

In `src/llm/client.py`, add directly after `_call_gemini` (after its closing `return resp.text` line):

```python
def _call_litellm(
    system: str,
    user: str,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    fallback_models: list[str],
) -> tuple[str, Optional[float]]:
    """Route a call through litellm instead of a hand-rolled per-provider
    adapter. The model string is built as `<provider>/<model>` — litellm
    dispatches on this prefix, which is what lets any of its 100+ supported
    providers (Bedrock, Azure, Ollama, Mistral, Cohere, ...) work here with
    no new adapter code, as long as models.yaml points at them. Provider
    quirks (e.g. OpenAI reasoning-model token params) are left to litellm's
    own normalization rather than re-implemented here."""
    import litellm

    kwargs: dict[str, Any] = {
        "model": f"{provider}/{model}",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode and provider in ("openai", "azure", "gemini"):
        kwargs["response_format"] = {"type": "json_object"}
    if fallback_models:
        kwargs["fallbacks"] = list(fallback_models)

    resp = litellm.completion(**kwargs)
    content = resp.choices[0].message.content

    try:
        cost = litellm.completion_cost(completion_response=resp)
    except Exception:  # noqa: BLE001 - cost is best-effort, never fatal
        cost = None

    return content, cost
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_backend.py -v`
Expected: PASS (all tests from Task 1 and Task 2).

- [ ] **Step 5: Commit**

```bash
git add src/llm/client.py tests/test_llm_backend.py
git commit -m "feat: add _call_litellm adapter with cost tracking and fallbacks"
```

---

### Task 3: Wire the backend branch and cost logging into `LLMClient.generate()`

**Files:**
- Modify: `src/llm/client.py:291-370` (`LLMClient.generate`)
- Test: `tests/test_llm_backend.py` (append)

**Interfaces:**
- Consumes: `node_model_config(node)["backend"]` / `["fallback_models"]` (Task 1), `_call_litellm(...)` (Task 2).
- Produces: `LLMClient.generate(...)` behavior — unchanged return contract (parsed dict or raw string), now backend-aware.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm_backend.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_backend.py -v -k generate`
Expected: FAIL — with `backend: litellm` in config, `generate()` currently still dispatches on the native `if provider == ...` chain and never calls `_call_litellm`, so `test_generate_litellm_backend_calls_call_litellm` fails (native `_call_openai` gets called for real, raising since it isn't mocked in that test — surfacing as the assertion failure or an unmocked-network error).

- [ ] **Step 3: Implement the branch and cost logging**

In `src/llm/client.py`, inside `LLMClient.generate`, replace this block:

```python
        last_error: Optional[str] = None
        attempt_user_prompt = user_prompt
        for attempt in range(retries + 1):
            self._calls_per_run[run_id] = self._calls_per_run.get(run_id, 0) + 1
            try:
                if provider == "anthropic":
                    raw = _call_anthropic(effective_system, attempt_user_prompt, model, temperature, max_tokens)
                elif provider == "openai":
                    raw = _call_openai(
                        effective_system, attempt_user_prompt, model, temperature, max_tokens,
                        json_mode=json_schema is not None,
                    )
                elif provider == "gemini":
                    raw = _call_gemini(
                        effective_system, attempt_user_prompt, model, temperature, max_tokens,
                        json_mode=json_schema is not None,
                    )
                else:
                    raise ValueError(f"Unknown provider '{provider}' for node '{node}'")
            except Exception as exc:  # noqa: BLE001 - surfaced to caller after logging
                log_llm_call(run_id, node, provider, model, effective_system, attempt_user_prompt, "", error=str(exc))
                last_error = str(exc)
                continue

            if json_schema is None:
                log_llm_call(run_id, node, provider, model, effective_system, attempt_user_prompt, raw)
                return raw

            try:
                parsed = _extract_json(raw)
                log_llm_call(run_id, node, provider, model, effective_system, attempt_user_prompt, raw)
                return parsed
            except json.JSONDecodeError as exc:
                log_llm_call(run_id, node, provider, model, effective_system, attempt_user_prompt, raw, error=str(exc))
                last_error = f"invalid JSON: {exc}"
                attempt_user_prompt = (
                    user_prompt
                    + f"\n\nYour previous response was not valid JSON ({exc}). "
                    "Return ONLY the corrected JSON object."
                )
                continue
```

with:

```python
        backend = cfg["backend"]
        fallback_models = cfg["fallback_models"]

        last_error: Optional[str] = None
        attempt_user_prompt = user_prompt
        for attempt in range(retries + 1):
            self._calls_per_run[run_id] = self._calls_per_run.get(run_id, 0) + 1
            cost: Optional[float] = None
            try:
                if backend == "litellm":
                    raw, cost = _call_litellm(
                        effective_system, attempt_user_prompt, provider, model, temperature, max_tokens,
                        json_mode=json_schema is not None, fallback_models=fallback_models,
                    )
                elif provider == "anthropic":
                    raw = _call_anthropic(effective_system, attempt_user_prompt, model, temperature, max_tokens)
                elif provider == "openai":
                    raw = _call_openai(
                        effective_system, attempt_user_prompt, model, temperature, max_tokens,
                        json_mode=json_schema is not None,
                    )
                elif provider == "gemini":
                    raw = _call_gemini(
                        effective_system, attempt_user_prompt, model, temperature, max_tokens,
                        json_mode=json_schema is not None,
                    )
                else:
                    raise ValueError(f"Unknown provider '{provider}' for node '{node}'")
            except Exception as exc:  # noqa: BLE001 - surfaced to caller after logging
                log_llm_call(
                    run_id, node, provider, model, effective_system, attempt_user_prompt, "",
                    error=str(exc), cost_usd=cost,
                )
                last_error = str(exc)
                continue

            if json_schema is None:
                log_llm_call(
                    run_id, node, provider, model, effective_system, attempt_user_prompt, raw, cost_usd=cost,
                )
                return raw

            try:
                parsed = _extract_json(raw)
                log_llm_call(
                    run_id, node, provider, model, effective_system, attempt_user_prompt, raw, cost_usd=cost,
                )
                return parsed
            except json.JSONDecodeError as exc:
                log_llm_call(
                    run_id, node, provider, model, effective_system, attempt_user_prompt, raw,
                    error=str(exc), cost_usd=cost,
                )
                last_error = f"invalid JSON: {exc}"
                attempt_user_prompt = (
                    user_prompt
                    + f"\n\nYour previous response was not valid JSON ({exc}). "
                    "Return ONLY the corrected JSON object."
                )
                continue
```

This requires `cfg` (the `node_model_config(node)` result) to already be in scope above this block — confirm the existing line `cfg = node_model_config(node)` (just above `provider, model = cfg["provider"], cfg["model"]`) is unchanged and still runs before this block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_backend.py -v`
Expected: PASS (all tests across Tasks 1-3).

Then run the full LLM-related suite to check nothing else regressed:

Run: `pytest tests/test_llm_backend.py tests/test_model_profiles.py tests/test_chat_node.py tests/test_pipeline_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/client.py tests/test_llm_backend.py
git commit -m "feat: dispatch LLMClient.generate() through litellm when backend=litellm"
```

---

### Task 4: Ship config, docs, and dependency changes

**Files:**
- Modify: `config/models.yaml`
- Modify: `.env.example`
- Modify: `requirements.txt`
- Test: `tests/test_llm_backend.py` (append one test against the real shipped file)

**Interfaces:**
- Consumes: `_effective_backend`, `node_model_config` (Task 1) — this task only edits data files plus one integration test.
- Produces: nothing new consumed by later tasks (this is the last task).

- [ ] **Step 1: Write the verification test**

Append to `tests/test_llm_backend.py`. This is a regression guard rather than a red-green TDD step: Task 1 already makes `backend`/`fallback_models` resolve for any config, including the real shipped file, so it should pass right away — its job is to catch a future edit to `config/models.yaml` (e.g. someone flipping `backend` to `litellm` by mistake) rather than to drive new code in this task.

```python
def test_shipped_models_yaml_defaults_to_native_backend(monkeypatch):
    llm_client._models_config.cache_clear()
    cfg = llm_client.node_model_config("chat")
    assert cfg["backend"] == "native"
    assert cfg["fallback_models"] == []
```

- [ ] **Step 2: Run test to confirm it already passes**

Run: `pytest tests/test_llm_backend.py -v -k shipped_models_yaml_defaults_to_native_backend`
Expected: PASS — confirms the current shipped `config/models.yaml` (no `backend` key yet) resolves to `"native"` before Step 3's edits touch the file.

- [ ] **Step 3: Update `config/models.yaml`**

Add a `backend` key at the very top of the file, replacing:

```yaml
# Which LLM provider+model powers each LangGraph node.
#
# SWITCHING PROVIDERS — one line: set `active_profile` to anthropic | openai |
# gemini. Or, without touching this file, set the AUTOML_LLM_PROFILE
# environment variable (takes precedence over active_profile).
# Escape hatches: AUTOML_LLM_MODEL forces one model id for every node;
# AUTOML_LLM_PROVIDER forces the provider (use together for a custom combo).
#
# API keys come from environment variables (see .env.example):
# ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY — never hardcoded here.

active_profile: openai
```

with:

```yaml
# Which LLM provider+model powers each LangGraph node.
#
# SWITCHING PROVIDERS — one line: set `active_profile` to anthropic | openai |
# gemini. Or, without touching this file, set the AUTOML_LLM_PROFILE
# environment variable (takes precedence over active_profile).
# Escape hatches: AUTOML_LLM_MODEL forces one model id for every node;
# AUTOML_LLM_PROVIDER forces the provider (use together for a custom combo).
#
# EXECUTION BACKEND — native (default) uses the hand-rolled adapters in
# src/llm/client.py for anthropic/openai/gemini. litellm routes every call
# through the litellm library instead, which unlocks any of its 100+
# supported providers (Bedrock, Azure, Ollama, Mistral, Cohere, ...) just by
# setting `provider`/`model` on a profile to that provider's litellm name —
# no new adapter code needed. Switch with `backend:` below, or the
# AUTOML_LLM_BACKEND environment variable (takes precedence over this key).
# litellm must be installed (`pip install litellm`) to use backend: litellm.
#
# API keys come from environment variables (see .env.example):
# ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY — never hardcoded here.

backend: native

active_profile: openai
```

Then, in the `profiles:` section, add one commented-out example showing a litellm-only provider plus `fallback_models`, directly after the existing `gemini:` profile block:

```yaml
  gemini:
    provider: gemini
    model: gemini-2.5-flash

  # Example: any litellm-supported provider works here once backend:
  # litellm is set above, with zero new code in src/llm/client.py.
  # bedrock_claude:
  #   provider: bedrock
  #   model: anthropic.claude-3-haiku-20240307-v1:0
  #   fallback_models:        # only used when backend: litellm
  #     - openai/gpt-5-nano
```

- [ ] **Step 4: Update `.env.example`**

Replace:

```
# Switch every node's LLM provider without editing config/models.yaml:
# one of the profiles defined there (anthropic | openai | gemini).
# Leave empty to use models.yaml's active_profile.
AUTOML_LLM_PROFILE=

# Escape hatches: force one model id (and/or provider) for every node,
# overriding whatever the profile says. Usually leave empty.
AUTOML_LLM_MODEL=
AUTOML_LLM_PROVIDER=
```

with:

```
# Switch every node's LLM provider without editing config/models.yaml:
# one of the profiles defined there (anthropic | openai | gemini).
# Leave empty to use models.yaml's active_profile.
AUTOML_LLM_PROFILE=

# Escape hatches: force one model id (and/or provider) for every node,
# overriding whatever the profile says. Usually leave empty.
AUTOML_LLM_MODEL=
AUTOML_LLM_PROVIDER=

# Execution backend: native (default) or litellm. litellm unlocks any of
# its 100+ supported providers via config/models.yaml profiles, and adds
# cost tracking + per-profile fallback_models. Leave empty to use
# models.yaml's `backend` key. Requires `pip install litellm`.
AUTOML_LLM_BACKEND=
```

- [ ] **Step 5: Update `requirements.txt`**

Replace:

```
# LLM providers (install only the ones you plan to use)
anthropic>=0.34.0
openai>=1.40.0
google-generativeai>=0.7.0
```

with:

```
# LLM providers (install only the ones you plan to use)
anthropic>=0.34.0
openai>=1.40.0
google-generativeai>=0.7.0
litellm>=1.50.0   # only needed when config/models.yaml sets backend: litellm
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_llm_backend.py -v`
Expected: PASS (all tests from Tasks 1-4).

Then run the full test suite to confirm no regressions:

Run: `pytest -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add config/models.yaml .env.example requirements.txt tests/test_llm_backend.py
git commit -m "docs: document litellm backend switch in models.yaml, .env.example, requirements"
```
