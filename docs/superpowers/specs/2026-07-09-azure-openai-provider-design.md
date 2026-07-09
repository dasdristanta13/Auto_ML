# Azure OpenAI provider support

## Problem

`src/llm/client.py` supports Anthropic, OpenAI, and Gemini as native providers
(`backend: native`). Azure OpenAI is only reachable today via `backend:
litellm`, which relies on litellm's own env-var conventions
(`AZURE_API_KEY`/`AZURE_API_BASE`/`AZURE_API_VERSION`) rather than this
project's config pattern. We want Azure OpenAI as a first-class native
provider, configured with an explicit endpoint URL and API key.

## Design

### Config surface (`config/models.yaml`)

A new profile, using the shipped defaults for this project:

```yaml
azure:
  provider: azure
  model: gpt-5.4-nano          # Azure *deployment name*, passed as `model`
  azure_endpoint: https://my-resource.openai.azure.com/
  api_version: "2025-04-01-preview"   # optional; falls back to this default if omitted
```

`azure_endpoint` and `api_version` are non-secret and live in yaml, following
the same pass-through mechanism `node_model_config()` already uses for
`provider`/`model`. The API key is a secret and stays out of yaml.

### Secrets (`.env.example`)

Add `AZURE_OPENAI_API_KEY=`, read from env inside the new adapter — matching
how `_call_anthropic`/`_call_openai` implicitly read `ANTHROPIC_API_KEY`/
`OPENAI_API_KEY` via the SDK's default env lookup.

### Adapter (`src/llm/client.py`)

New function `_call_azure_openai(system, user, model, temperature, max_tokens,
json_mode, azure_endpoint, api_version)`, using `openai.AzureOpenAI` (the
`openai` package is already a dependency for `_call_openai`). Mirrors
`_call_openai`'s shape: builds `messages` (system + user), sets
`temperature`/`max_tokens`, adds `response_format={"type": "json_object"}`
when `json_mode` is true, and raises a descriptive `RuntimeError` if the
response content comes back empty (same guard `_call_openai` has for
exhausted reasoning-token budgets).

No reasoning-model special-casing (`_is_openai_reasoning_model`) — out of
scope per YAGNI; can be added later if an o1/o3-class deployment is used via
Azure.

### Routing (`LLMClient.generate`)

New branch alongside the existing `elif provider == "openai"` /
`elif provider == "gemini"`:

```python
elif provider == "azure":
    azure_endpoint = cfg.get("azure_endpoint")
    if not azure_endpoint:
        raise ValueError(f"provider 'azure' requires azure_endpoint in the profile for node '{node}'")
    raw = _call_azure_openai(
        effective_system, attempt_user_prompt, model, temperature, max_tokens,
        json_mode=json_schema is not None,
        azure_endpoint=azure_endpoint,
        api_version=cfg.get("api_version", _DEFAULT_AZURE_API_VERSION),
    )
```

`_DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"` (matches the value used in
the shipped `azure` profile, so an operator who copies that profile and later
deletes the `api_version` line gets the same behavior).

`node_model_config()` needs one addition: pass `azure_endpoint`/`api_version`
through from the active profile into `merged`, the same way `provider`/
`model` are already copied over.

### Error handling

- Missing `azure_endpoint` for provider `azure` → `ValueError` raised before
  any network call (mirrors the existing "No provider/model configured for
  node" guard), not a network-level SDK error.
- Missing `AZURE_OPENAI_API_KEY` env var → surfaces as whatever error
  `openai.AzureOpenAI()` raises natively (consistent with how the other
  adapters handle missing keys today — no special-casing).
- Empty response content → `RuntimeError` with a descriptive message, same
  pattern as `_call_openai`.

### Testing (`tests/test_llm_backend.py`)

1. `node_model_config` test: a profile with `azure_endpoint`/`api_version`
   set is surfaced unchanged in the merged config.
2. `node_model_config` test: `provider: azure` without `azure_endpoint`
   still resolves the config (the `ValueError` guard lives in `generate`,
   not `node_model_config`, since a profile might legitimately be inspected
   without ever being called).
3. `_call_azure_openai` unit test using a fake `AzureOpenAI` client (same
   fake-injection pattern already used for the litellm tests) — asserts the
   deployment name, endpoint, api_version, and messages are passed through
   correctly, and that `json_mode=True` sets `response_format`.
4. `_call_azure_openai` empty-content test, mirroring
   `test_call_litellm_raises_on_empty_content`.
5. `LLMClient.generate` routing test: `provider == "azure"` calls
   `_call_azure_openai` with the resolved endpoint/version.
6. `LLMClient.generate` test: `provider == "azure"` with no `azure_endpoint`
   in the profile raises `ValueError` before any adapter call.

### Docs

- `.env.example`: add `AZURE_OPENAI_API_KEY=`.
- `config/models.yaml`: add the `azure` profile (shown above, using
  `gpt-5.4-nano` / `2025-04-01-preview`) alongside the existing
  `openai`/`anthropic`/`gemini` profiles, plus a one-line comment on the
  `AZURE_OPENAI_API_KEY` env var next to the existing key-source comment
  block at the top of the file.

## Out of scope

- Azure AD / managed-identity auth (key-based auth only, per the request).
- Reasoning-model token-budget handling for Azure-hosted o1/o3-class
  deployments.
- Changes to the `litellm` backend's existing (already-working) Azure path.
