# litellm Backend — Design

Date: 2026-07-07
Status: approved

## Problem

`src/llm/client.py` has one hand-rolled adapter per provider
(`_call_anthropic`/`_call_openai`/`_call_gemini`), each reimplementing
provider-specific quirks (e.g. OpenAI reasoning-model token params). Adding
a new provider today means writing a new `_call_x` function. litellm offers
a single `completion()` call that already speaks 100+ providers (Bedrock,
Azure, Ollama, Mistral, Cohere, ...) and normalizes their quirks, plus
built-in cost tracking and fallback lists — but the existing native path
should remain available and be the default, since it has no extra
dependency and is already proven in this codebase.

## Decision (from brainstorming)

A single global `backend: native | litellm` switch (config + env override),
matching the existing `AUTOML_LLM_PROFILE` override pattern. When
`litellm`, provider+model strings are passed straight to
`litellm.completion()` as `f"{provider}/{model}"` — this is what unlocks
arbitrary litellm-supported providers with zero new adapter code, since a
profile can set `provider: bedrock` / `model: anthropic.claude-3-haiku-...`
today without any change to `client.py`. Cost tracking and per-profile
fallback lists are turned on for the litellm path; caching is out of scope.

## Design

### config/models.yaml

```yaml
backend: native   # native | litellm — one-line switch, overridable by
                   # AUTOML_LLM_BACKEND env var

active_profile: openai

profiles:
  openai:
    provider: openai
    model: gpt-5-nano

  anthropic:
    provider: anthropic
    model: claude-haiku-4-5
    nodes:
      feature_engineering: { model: claude-opus-4-8 }
      model_selection: { model: claude-opus-4-8 }
      report: { model: claude-opus-4-8 }

  gemini:
    provider: gemini
    model: gemini-2.5-flash

  # Example custom profile: any litellm-supported provider works here with
  # no code change, as long as backend: litellm is active.
  bedrock_claude:
    provider: bedrock
    model: anthropic.claude-3-haiku-20240307-v1:0
    fallback_models:            # only used when backend: litellm
      - openai/gpt-5-nano
```

`fallback_models` is an optional list on a profile: `provider/model`
strings passed to litellm's `fallbacks=` kwarg. Ignored entirely under
`backend: native`.

### src/llm/client.py

- `_effective_backend(cfg) -> str`: `AUTOML_LLM_BACKEND` env var, else
  `cfg.get("backend", "native")`. Read per call, like the other env
  escape hatches (no yaml re-read needed to switch).
- `node_model_config()` gains `fallback_models` in its merged output,
  sourced from the active profile only (empty list if absent). No change
  to the existing provider/model/temperature/max_tokens resolution order.
- New `_call_litellm(system, user, provider, model, temperature, max_tokens, json_mode, fallback_models) -> tuple[str, Optional[float]]`:
  - lazy `import litellm`
  - `model=f"{provider}/{model}"`; same construction for entries in
    `fallback_models`
  - passes `response_format={"type": "json_object"}` only when
    `json_mode` and `provider in {"openai", "azure", "gemini"}` — mirrors
    which providers the native path already treats as JSON-mode-capable
    (native `_call_anthropic` never receives a `json_mode` flag at all)
  - does **not** reimplement the OpenAI reasoning-model token-param
    special-casing that `_call_openai` has — that's exactly the class of
    quirk litellm normalizes internally
  - calls `litellm.completion(**kwargs)`, extracts
    `resp.choices[0].message.content`
  - computes `cost = litellm.completion_cost(completion_response=resp)`
    in a `try/except` (best-effort; `None` if the model isn't in
    litellm's cost table)
  - returns `(content, cost)`
- `LLMClient.generate()`: per attempt, branches once on
  `_effective_backend()` — `"litellm"` → `_call_litellm`, anything else →
  today's native `if provider == "anthropic"/"openai"/"gemini"` dispatch.
  Budget check, the JSON-repair retry loop, and `log_llm_call` calls are
  unchanged and shared by both paths. When a cost is available, it's
  passed to `log_llm_call(..., cost_usd=cost)` — an existing `**extra`
  sink, no tracing schema change.
- Unknown `backend` value → `ValueError`, same style as the existing
  unknown-provider error.

### requirements.txt

Add `litellm>=1.50.0` under the LLM providers section, commented as
optional (only needed when `backend: litellm`), consistent with how
`anthropic`/`openai`/`google-generativeai` are already listed as
install-only-what-you-use.

### .env.example

Document `AUTOML_LLM_BACKEND` next to `AUTOML_LLM_PROFILE`.

## Testing (tests/test_llm_backend.py)

- `backend` unset/`native` → dispatch is unchanged (existing
  `_call_anthropic`/`_call_openai`/`_call_gemini` behavior, verified via
  the same mocking approach as `tests/test_model_profiles.py`).
- `backend: litellm` → `litellm.completion` is called with
  `model="<provider>/<model>"` and `fallbacks=` built from
  `fallback_models`.
- `AUTOML_LLM_BACKEND` env var overrides `models.yaml`'s `backend` key.
- Cost is attached to the trace log when `completion_cost` succeeds, and
  the call still succeeds (cost logged as `None`) when it raises.
- `json_mode` → `response_format` is passed for openai/gemini providers
  and omitted for anthropic, under the litellm backend.

## Out of scope

- Response caching (in-memory or Redis).
- litellm proxy server / spend-tracking dashboard.
- Per-node backend switching (only the global switch from brainstorming).
- Streaming.
