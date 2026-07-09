# Azure OpenAI Provider Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Azure OpenAI as a fourth native LLM provider in `src/llm/client.py` (alongside anthropic/openai/gemini), configured with an explicit endpoint URL (yaml) and API key (env var).

**Architecture:** A new `_call_azure_openai()` adapter using `openai.AzureOpenAI`, wired into `node_model_config()` (to pass through `azure_endpoint`/`api_version` from the active profile) and `LLMClient.generate()` (new `provider == "azure"` branch, with a fail-fast `ValueError` if `azure_endpoint` is missing).

**Tech Stack:** Python 3.11+, `openai` SDK (`AzureOpenAI` client — already a dependency for `_call_openai`), pytest + `monkeypatch` for tests.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-09-azure-openai-provider-design.md`
- No secrets in `config/models.yaml` — API key comes only from `AZURE_OPENAI_API_KEY` env var (CLAUDE.md: "Do not hardcode credentials, API keys, or model version strings — use `config/`").
- Model string is never hardcoded in `src/llm/client.py` — the shipped model id (`gpt-5.4-nano`) and api version (`2025-04-01-preview`) live only in `config/models.yaml`.
- Default `api_version` fallback constant in code must equal `"2025-04-01-preview"` (matches the shipped profile value).
- No reasoning-model special-casing for Azure (out of scope per spec).
- Every LLM call must remain trace-logged via `log_llm_call` (CLAUDE.md rule #7) — this is already handled generically by `LLMClient.generate()`; new adapters must not bypass it.

---

### Task 1: Pass `azure_endpoint`/`api_version` through `node_model_config()`

**Files:**
- Modify: `src/llm/client.py:95-99` (inside `node_model_config`)
- Test: `tests/test_llm_backend.py`

**Interfaces:**
- Consumes: existing `node_model_config(node: str) -> dict[str, Any]`, existing `BACKEND_CFG` fixture pattern in `tests/test_llm_backend.py`.
- Produces: `node_model_config()` return dict now optionally contains `"azure_endpoint": str` and `"api_version": str` keys when the active profile defines them. Task 3 relies on `cfg.get("azure_endpoint")` and `cfg.get("api_version", ...)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_llm_backend.py` (near the other `node_model_config` tests, after `test_profile_fallback_models_surfaced`):

```python
def test_azure_profile_surfaces_endpoint_and_api_version(monkeypatch):
    cfg = {
        "backend": "native",
        "active_profile": "azure",
        "profiles": {
            "azure": {
                "provider": "azure",
                "model": "gpt-5.4-nano",
                "azure_endpoint": "https://my-resource.openai.azure.com/",
                "api_version": "2025-04-01-preview",
            },
        },
        "default": {"temperature": 0.0, "max_tokens": 4096},
        "nodes": {},
    }
    monkeypatch.setattr(llm_client, "_models_config", lambda: cfg)
    resolved = llm_client.node_model_config("chat")
    assert resolved["provider"] == "azure"
    assert resolved["model"] == "gpt-5.4-nano"
    assert resolved["azure_endpoint"] == "https://my-resource.openai.azure.com/"
    assert resolved["api_version"] == "2025-04-01-preview"


def test_azure_profile_without_azure_endpoint_still_resolves(monkeypatch):
    cfg = {
        "backend": "native",
        "active_profile": "azure",
        "profiles": {"azure": {"provider": "azure", "model": "gpt-5.4-nano"}},
        "default": {"temperature": 0.0, "max_tokens": 4096},
        "nodes": {},
    }
    monkeypatch.setattr(llm_client, "_models_config", lambda: cfg)
    resolved = llm_client.node_model_config("chat")
    assert resolved["provider"] == "azure"
    assert "azure_endpoint" not in resolved
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_backend.py -k azure_profile -v`
Expected: FAIL — `resolved["azure_endpoint"]` raises `KeyError` (the key doesn't exist yet).

- [ ] **Step 3: Implement the passthrough**

In `src/llm/client.py`, inside `node_model_config`, the current profile block reads:

```python
    profile = _active_profile(cfg)
    if profile is not None:
        merged["provider"] = profile["provider"]
        node_override = (profile.get("nodes") or {}).get(node) or {}
        merged["model"] = node_override.get("model", profile["model"])
```

Change it to:

```python
    profile = _active_profile(cfg)
    if profile is not None:
        merged["provider"] = profile["provider"]
        node_override = (profile.get("nodes") or {}).get(node) or {}
        merged["model"] = node_override.get("model", profile["model"])
        if "azure_endpoint" in profile:
            merged["azure_endpoint"] = profile["azure_endpoint"]
        if "api_version" in profile:
            merged["api_version"] = profile["api_version"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_backend.py -k azure_profile -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm/client.py tests/test_llm_backend.py
git commit -m "feat: pass azure_endpoint/api_version through node_model_config"
```

---

### Task 2: `_call_azure_openai` adapter

**Files:**
- Modify: `src/llm/client.py` (add new function after `_call_gemini`, which currently ends at line 308, before `_call_litellm` at line 311)
- Test: `tests/test_llm_backend.py`

**Interfaces:**
- Consumes: nothing new from earlier tasks (standalone adapter function, same shape as `_call_openai`/`_call_gemini`).
- Produces: `_call_azure_openai(system: str, user: str, model: str, temperature: float, max_tokens: int, json_mode: bool, azure_endpoint: str, api_version: str) -> str`. Task 3 calls this exact signature.
- Also produces module-level constant `_DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"`, which Task 3 uses as the fallback when a profile omits `api_version`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_llm_backend.py`, near the other fake-client fixtures (after the `_FakeLiteLLM` class / before `fake_litellm` fixture is fine, or anywhere below the imports):

```python
class _FakeAzureMessage:
    def __init__(self, content):
        self.content = content


class _FakeAzureChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = _FakeAzureMessage(content)
        self.finish_reason = finish_reason


class _FakeAzureCompletionResponse:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [_FakeAzureChoice(content, finish_reason)]


class _FakeAzureChatCompletions:
    def __init__(self, content, finish_reason):
        self._content = content
        self._finish_reason = finish_reason
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeAzureCompletionResponse(self._content, self._finish_reason)


class _FakeAzureChat:
    def __init__(self, content, finish_reason):
        self.completions = _FakeAzureChatCompletions(content, finish_reason)


class _FakeAzureOpenAI:
    """Stand-in for openai.AzureOpenAI — records constructor args and the
    chat.completions.create kwargs so tests can assert on both."""

    last_instance = None

    def __init__(self, content="hello from azure", finish_reason="stop", azure_endpoint=None, api_version=None):
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.chat = _FakeAzureChat(content, finish_reason)
        _FakeAzureOpenAI.last_instance = self


def test_call_azure_openai_passes_endpoint_and_deployment(monkeypatch):
    fake_module = type("FakeOpenAIModule", (), {"AzureOpenAI": _FakeAzureOpenAI})
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)

    result = llm_client._call_azure_openai(
        "sys", "user", "gpt-5.4-nano", 0.0, 2048,
        json_mode=False,
        azure_endpoint="https://my-resource.openai.azure.com/",
        api_version="2025-04-01-preview",
    )

    assert result == "hello from azure"
    instance = _FakeAzureOpenAI.last_instance
    assert instance.azure_endpoint == "https://my-resource.openai.azure.com/"
    assert instance.api_version == "2025-04-01-preview"
    assert instance.chat.completions.last_kwargs["model"] == "gpt-5.4-nano"
    assert instance.chat.completions.last_kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]


def test_call_azure_openai_json_mode_sets_response_format(monkeypatch):
    fake_module = type("FakeOpenAIModule", (), {"AzureOpenAI": _FakeAzureOpenAI})
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)

    llm_client._call_azure_openai(
        "sys", "user", "gpt-5.4-nano", 0.0, 2048,
        json_mode=True,
        azure_endpoint="https://my-resource.openai.azure.com/",
        api_version="2025-04-01-preview",
    )

    kwargs = _FakeAzureOpenAI.last_instance.chat.completions.last_kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


def test_call_azure_openai_raises_on_empty_content(monkeypatch):
    fake_module = type(
        "FakeOpenAIModule", (),
        {"AzureOpenAI": lambda **kw: _FakeAzureOpenAI(content=None, finish_reason="length", **kw)},
    )
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)

    with pytest.raises(RuntimeError, match="empty content"):
        llm_client._call_azure_openai(
            "sys", "user", "gpt-5.4-nano", 0.0, 2048,
            json_mode=False,
            azure_endpoint="https://my-resource.openai.azure.com/",
            api_version="2025-04-01-preview",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_backend.py -k call_azure_openai -v`
Expected: FAIL — `AttributeError: module 'src.llm.client' has no attribute '_call_azure_openai'`

- [ ] **Step 3: Implement the adapter**

In `src/llm/client.py`, add this after `_call_gemini` (which ends right before `def _call_litellm(`):

```python
_DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"


def _call_azure_openai(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    azure_endpoint: str,
    api_version: str,
) -> str:
    """`model` is the Azure *deployment name* (set via config/models.yaml's
    profile `model` field), not an OpenAI model id. The API key is read from
    AZURE_OPENAI_API_KEY by the SDK's default env lookup — never passed or
    logged here (CLAUDE.md: no hardcoded credentials)."""
    from openai import AzureOpenAI

    client = AzureOpenAI(azure_endpoint=azure_endpoint, api_version=api_version)
    kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )
    content = resp.choices[0].message.content
    if not content:
        finish_reason = resp.choices[0].finish_reason
        raise RuntimeError(
            f"Azure OpenAI returned empty content (finish_reason={finish_reason}). "
            "This usually means the token budget was exhausted before any visible "
            "output was written — increase max_tokens for this node in "
            "config/models.yaml."
        )
    return content
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_backend.py -k call_azure_openai -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm/client.py tests/test_llm_backend.py
git commit -m "feat: add native Azure OpenAI adapter (_call_azure_openai)"
```

---

### Task 3: Route `provider == "azure"` in `LLMClient.generate()`

**Files:**
- Modify: `src/llm/client.py:399-440` (inside `LLMClient.generate`)
- Test: `tests/test_llm_backend.py`

**Interfaces:**
- Consumes: `_call_azure_openai(...)` and `_DEFAULT_AZURE_API_VERSION` from Task 2; `cfg["azure_endpoint"]` / `cfg.get("api_version", ...)` from Task 1.
- Produces: `LLMClient.generate()` now supports `provider == "azure"` end-to-end. Nothing downstream depends on new names from this task.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_llm_backend.py`, near `test_generate_native_backend_calls_call_openai`:

```python
@pytest.fixture()
def azure_generate_cfg(monkeypatch):
    cfg = {
        "backend": "native",
        "active_profile": "azure",
        "profiles": {
            "azure": {
                "provider": "azure",
                "model": "gpt-5.4-nano",
                "azure_endpoint": "https://my-resource.openai.azure.com/",
                "api_version": "2025-04-01-preview",
            },
        },
        "default": {"temperature": 0.0, "max_tokens": 4096},
        "nodes": {},
    }
    monkeypatch.setattr(llm_client, "_models_config", lambda: cfg)
    monkeypatch.setattr(
        llm_client, "_runtime_config", lambda: {"budgets": {"max_llm_calls_per_run": 100}}
    )
    return cfg


def test_generate_native_backend_calls_call_azure_openai(azure_generate_cfg, monkeypatch):
    calls = {}

    def fake_call_azure_openai(system, user, model, temperature, max_tokens, json_mode, azure_endpoint, api_version):
        calls["args"] = (model, azure_endpoint, api_version, json_mode)
        return "azure response"

    monkeypatch.setattr(llm_client, "_call_azure_openai", fake_call_azure_openai)
    client = llm_client.LLMClient()
    result = client.generate("run-azure-1", "chat", "system prompt", "user prompt")

    assert result == "azure response"
    assert calls["args"] == (
        "gpt-5.4-nano", "https://my-resource.openai.azure.com/", "2025-04-01-preview", False,
    )


def test_generate_azure_missing_endpoint_raises_value_error(azure_generate_cfg, monkeypatch):
    azure_generate_cfg["profiles"]["azure"].pop("azure_endpoint")
    called = {"count": 0}

    def fail_if_called(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("_call_azure_openai should not be invoked")

    monkeypatch.setattr(llm_client, "_call_azure_openai", fail_if_called)
    client = llm_client.LLMClient()

    with pytest.raises(ValueError, match="azure_endpoint"):
        client.generate("run-azure-2", "chat", "system prompt", "user prompt")

    assert called["count"] == 0


def test_generate_azure_falls_back_to_default_api_version(azure_generate_cfg, monkeypatch):
    azure_generate_cfg["profiles"]["azure"].pop("api_version")
    calls = {}

    def fake_call_azure_openai(system, user, model, temperature, max_tokens, json_mode, azure_endpoint, api_version):
        calls["api_version"] = api_version
        return "azure response"

    monkeypatch.setattr(llm_client, "_call_azure_openai", fake_call_azure_openai)
    client = llm_client.LLMClient()
    client.generate("run-azure-3", "chat", "system prompt", "user prompt")

    assert calls["api_version"] == llm_client._DEFAULT_AZURE_API_VERSION
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_backend.py -k "generate_native_backend_calls_call_azure_openai or generate_azure" -v`
Expected: FAIL — `test_generate_native_backend_calls_call_azure_openai` fails because `generate()` doesn't call `_call_azure_openai` (falls through to `else: raise ValueError(f"Unknown provider 'azure'...")`); `test_generate_azure_missing_endpoint_raises_value_error` fails because the error message doesn't mention `azure_endpoint`.

- [ ] **Step 3: Implement the routing**

In `src/llm/client.py`, `LLMClient.generate` currently has:

```python
        cfg = node_model_config(node)
        provider, model = cfg["provider"], cfg["model"]
        temperature = cfg.get("temperature", 0.0)
        max_tokens = cfg.get("max_tokens", 2048)
```

Change to add the fail-fast guard right after:

```python
        cfg = node_model_config(node)
        provider, model = cfg["provider"], cfg["model"]
        temperature = cfg.get("temperature", 0.0)
        max_tokens = cfg.get("max_tokens", 2048)

        if provider == "azure" and not cfg.get("azure_endpoint"):
            raise ValueError(f"provider 'azure' requires azure_endpoint in the profile for node '{node}'")
```

Then, inside the retry loop, the current branch chain is:

```python
                elif provider == "gemini":
                    raw = _call_gemini(
                        effective_system, attempt_user_prompt, model, temperature, max_tokens,
                        json_mode=json_schema is not None,
                    )
                else:
                    raise ValueError(f"Unknown provider '{provider}' for node '{node}'")
```

Add an `azure` branch before the `else`:

```python
                elif provider == "gemini":
                    raw = _call_gemini(
                        effective_system, attempt_user_prompt, model, temperature, max_tokens,
                        json_mode=json_schema is not None,
                    )
                elif provider == "azure":
                    raw = _call_azure_openai(
                        effective_system, attempt_user_prompt, model, temperature, max_tokens,
                        json_mode=json_schema is not None,
                        azure_endpoint=cfg["azure_endpoint"],
                        api_version=cfg.get("api_version", _DEFAULT_AZURE_API_VERSION),
                    )
                else:
                    raise ValueError(f"Unknown provider '{provider}' for node '{node}'")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_backend.py -v`
Expected: PASS (all tests in the file, including the 3 new ones and everything from Tasks 1-2)

- [ ] **Step 5: Commit**

```bash
git add src/llm/client.py tests/test_llm_backend.py
git commit -m "feat: route provider 'azure' to _call_azure_openai in LLMClient.generate"
```

---

### Task 4: Ship the `azure` profile in config + document the env var

**Files:**
- Modify: `config/models.yaml:18-51`
- Modify: `.env.example:1-6`
- Test: `tests/test_llm_backend.py`

**Interfaces:**
- Consumes: nothing new — this task only adds shipped config data and a docs-verifying test using the real `config/models.yaml`/`node_model_config` (same pattern as the existing `test_shipped_models_yaml_defaults_to_native_backend`).
- Produces: nothing consumed by later tasks (this is the last task).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_llm_backend.py`, near `test_shipped_models_yaml_defaults_to_native_backend`:

```python
def test_shipped_models_yaml_has_azure_profile(monkeypatch):
    llm_client._models_config.cache_clear()
    monkeypatch.setenv("AUTOML_LLM_PROFILE", "azure")
    cfg = llm_client.node_model_config("chat")
    assert cfg["provider"] == "azure"
    assert cfg["model"] == "gpt-5.4-nano"
    assert cfg["azure_endpoint"]
    assert cfg["api_version"] == "2025-04-01-preview"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_backend.py -k shipped_models_yaml_has_azure_profile -v`
Expected: FAIL — `ValueError: unknown LLM profile 'azure'` (no `azure` profile exists in `config/models.yaml` yet).

- [ ] **Step 3: Add the `azure` profile to `config/models.yaml`**

Open `config/models.yaml`. The header comment block (lines 1-19) currently ends with:

```yaml
# API keys come from environment variables (see .env.example):
# ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY — never hardcoded here.

backend: native
```

Change the comment line to also mention Azure:

```yaml
# API keys come from environment variables (see .env.example):
# ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, AZURE_OPENAI_API_KEY —
# never hardcoded here. Azure additionally needs azure_endpoint (and
# optionally api_version) set directly on its profile below, since those
# are non-secret.

backend: native
```

Then, in the `profiles:` section, the `gemini` profile currently ends right before the litellm-example comment block:

```yaml
  gemini:
    provider: gemini
    model: gemini-2.5-flash

  # Example: any litellm-supported provider works here once backend:
```

Add the `azure` profile between them:

```yaml
  gemini:
    provider: gemini
    model: gemini-2.5-flash

  azure:
    provider: azure
    model: gpt-5.4-nano   # Azure *deployment name*, not an OpenAI model id
    azure_endpoint: https://my-resource.openai.azure.com/
    api_version: "2025-04-01-preview"

  # Example: any litellm-supported provider works here once backend:
```

Note: `azure_endpoint` above is a placeholder resource URL — whoever deploys this must replace it with their actual Azure OpenAI resource endpoint. This is expected: it's non-secret configuration, not a working credential.

- [ ] **Step 4: Add `AZURE_OPENAI_API_KEY` to `.env.example`**

Open `.env.example`. Current top of file:

```
# Copy to .env and fill in whichever provider(s) you configured in config/models.yaml.
# Only the keys for providers actually referenced in models.yaml are required.

ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
```

Change to:

```
# Copy to .env and fill in whichever provider(s) you configured in config/models.yaml.
# Only the keys for providers actually referenced in models.yaml are required.

ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
AZURE_OPENAI_API_KEY=
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_llm_backend.py -k shipped_models_yaml_has_azure_profile -v`
Expected: PASS

- [ ] **Step 6: Run the full test file to confirm no regressions**

Run: `pytest tests/test_llm_backend.py -v`
Expected: PASS (all tests)

- [ ] **Step 7: Commit**

```bash
git add config/models.yaml .env.example tests/test_llm_backend.py
git commit -m "feat: ship azure profile (gpt-5.4-nano, api-version 2025-04-01-preview) and AZURE_OPENAI_API_KEY"
```

---

## Self-Review Notes

- **Spec coverage:** config surface (Task 1 + 4), adapter (Task 2), routing + error handling (Task 3), docs/.env.example (Task 4), testing (all 4 tasks) — all spec sections have a corresponding task. Azure AD/managed-identity auth and reasoning-model handling are explicitly out of scope per the spec and are not implemented here.
- **Placeholder scan:** no TODOs; the one intentional placeholder value (`https://my-resource.openai.azure.com/` as a to-be-replaced resource URL in shipped yaml) is flagged explicitly as expected, not left ambiguous.
- **Type/signature consistency:** `_call_azure_openai(system, user, model, temperature, max_tokens, json_mode, azure_endpoint, api_version) -> str` is defined once in Task 2 and called with matching keyword args in Task 3; `_DEFAULT_AZURE_API_VERSION` is defined in Task 2 and referenced by exact name in Task 3.
