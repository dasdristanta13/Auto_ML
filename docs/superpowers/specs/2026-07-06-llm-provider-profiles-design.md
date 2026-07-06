# LLM Provider Profiles — Design

Date: 2026-07-06
Status: approved

## Problem

The LLM client already has anthropic/openai/gemini adapters and models.yaml
maps provider+model per node — but switching providers means editing every
node block (all currently hardcoded to openai/gpt-5-nano), and there is no
way to switch without editing files. "Easy switching from the backend"
between OpenAI, Gemini, and Anthropic models needs one switch.

## Decision (from brainstorming)

Named profiles in models.yaml + environment-variable override. No runtime
UI switcher (deferred).

## Design

### config/models.yaml

```yaml
active_profile: openai          # one-line switch

profiles:
  openai:
    provider: openai
    model: gpt-5-nano           # profile-wide default model
  anthropic:
    provider: anthropic
    model: claude-haiku-4-5     # cheap tier for simple nodes
    nodes:                      # per-node model tiers within the profile
      feature_engineering: {model: claude-opus-4-8}
      model_selection: {model: claude-opus-4-8}
      report: {model: claude-opus-4-8}
  gemini:
    provider: gemini
    model: gemini-2.5-flash

default:                        # generation params shared by all profiles
  temperature: 0.0
  max_tokens: 4096
nodes:                          # per-node generation params (provider-free)
  understand_usecase: {max_tokens: 2048}
  ...
```

### Resolution order (src/llm/client.py, node_model_config)

1. Merge `default` + `nodes.<node>` (temperature/max_tokens, and legacy
   provider/model if a config still carries them — full backward compat
   with the old schema).
2. If `profiles` exist: the active profile (env `AUTOML_LLM_PROFILE`, else
   yaml `active_profile`) supplies `provider` and `model` (profile default,
   overridden by `profiles.<name>.nodes.<node>.model`). Unknown profile
   name → ValueError listing available profiles.
3. Env escape hatches applied last: `AUTOML_LLM_PROVIDER` forces provider,
   `AUTOML_LLM_MODEL` forces one model id for every node.

Env vars are read per call (not cached), so switching via .env + restart or
process env works without editing yaml; the yaml file itself stays cached.

### .env.example

Document `AUTOML_LLM_PROFILE`, `AUTOML_LLM_MODEL`, `AUTOML_LLM_PROVIDER`
next to the provider API keys (ANTHROPIC_API_KEY / OPENAI_API_KEY /
GOOGLE_API_KEY).

### Model choices

- openai profile: gpt-5-nano everywhere (current behavior, unchanged).
- anthropic profile: claude-haiku-4-5 for cheap/simple nodes
  (understand_usecase, chat), claude-opus-4-8 for planning-quality nodes
  (feature_engineering, model_selection, report).
- gemini profile: gemini-2.5-flash everywhere.

## Testing (tests/test_model_profiles.py)

- Active profile from yaml resolves provider+model per node, honoring the
  profile's per-node tiers.
- AUTOML_LLM_PROFILE env var overrides yaml active_profile.
- Unknown profile raises ValueError naming the available profiles.
- AUTOML_LLM_MODEL / AUTOML_LLM_PROVIDER force overrides.
- Legacy schema (default+nodes with provider/model, no profiles key) still
  resolves exactly as before.
- Generation params (temperature/max_tokens) come from default/nodes and
  are unaffected by profile switching.

## Out of scope

- Runtime/UI profile switching endpoint.
- Per-run provider selection.
