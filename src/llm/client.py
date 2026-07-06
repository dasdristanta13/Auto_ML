"""Provider-agnostic LLM client.

Which node uses which provider/model is configured in config/models.yaml, never
hardcoded here (CLAUDE.md: "do not hardcode model strings; read from config/models.yaml").
Swapping a node from Claude to GPT-4o to Gemini is a one-line YAML edit.

Each provider adapter is imported lazily so a user only needs the SDK installed
for the provider(s) they actually configured.

Mock mode: set AUTOML_MOCK_LLM=1 (env or .env) to replace every provider call
with deterministic canned responses — lets the full pipeline + web UI run
locally with no API keys and no network. Mock responses are still trace-logged.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from src.llm.tracing import log_llm_call

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


class LLMBudgetExceeded(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _models_config() -> dict[str, Any]:
    with open(CONFIG_DIR / "models.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _runtime_config() -> dict[str, Any]:
    with open(CONFIG_DIR / "runtime.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _active_profile(cfg: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Resolve the active provider profile: AUTOML_LLM_PROFILE env var wins,
    then models.yaml's `active_profile`. Returns None for legacy configs
    without a `profiles` section (per-node provider/model then applies)."""
    profiles = cfg.get("profiles") or {}
    if not profiles:
        return None
    name = os.environ.get("AUTOML_LLM_PROFILE") or cfg.get("active_profile")
    if not name:
        return None
    if name not in profiles:
        raise ValueError(f"unknown LLM profile '{name}' — available profiles: {sorted(profiles)}")
    return profiles[name]


def node_model_config(node: str) -> dict[str, Any]:
    """Effective provider/model/generation-params for a node.

    Resolution order (see docs/superpowers/specs/2026-07-06-llm-provider-
    profiles-design.md):
      1. `default` + `nodes.<node>` merge — generation params, plus legacy
         per-node provider/model for configs predating profiles.
      2. The active profile (AUTOML_LLM_PROFILE env or `active_profile`)
         supplies provider + model, with per-node model tiers from
         `profiles.<name>.nodes.<node>.model`.
      3. AUTOML_LLM_PROVIDER / AUTOML_LLM_MODEL env vars force a provider /
         one model id across every node (read per call — switching needs no
         yaml edit, just a process restart).
    """
    cfg = _models_config()
    merged = dict(cfg.get("default", {}))
    merged.update(cfg.get("nodes", {}).get(node, {}))

    profile = _active_profile(cfg)
    if profile is not None:
        merged["provider"] = profile["provider"]
        node_override = (profile.get("nodes") or {}).get(node) or {}
        merged["model"] = node_override.get("model", profile["model"])

    if os.environ.get("AUTOML_LLM_PROVIDER"):
        merged["provider"] = os.environ["AUTOML_LLM_PROVIDER"]
    if os.environ.get("AUTOML_LLM_MODEL"):
        merged["model"] = os.environ["AUTOML_LLM_MODEL"]

    if "provider" not in merged or "model" not in merged:
        raise ValueError(f"No provider/model configured for node '{node}' in config/models.yaml")
    return merged


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from an LLM response."""
    text = text.strip()
    # strip markdown code fences if present
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fall back to grabbing the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _mock_response(node: str, system_prompt: str) -> Any:
    """Deterministic stand-in responses for AUTOML_MOCK_LLM=1. The use-case
    node reports itself ambiguous so the human confirmation step always runs —
    the user supplies the target/task/metric through the UI, which is exactly
    the FR-9 checkpoint path."""
    if node == "understand_usecase":
        return {
            "task_type": None,
            "target_column": None,
            "metric": None,
            "time_column": None,
            "constraints": [],
            "is_ambiguous": True,
            "ambiguity_reason": "Mock LLM mode is active (AUTOML_MOCK_LLM=1); please confirm the task details.",
        }
    if node == "feature_engineering":
        return {
            "steps": [],
            "plan_rationale": "Mock mode: no transformations planned; training uses numeric columns as-is.",
        }
    if node == "model_selection":
        is_regression = '"task_type": "regression"' in system_prompt or '"task_type": "forecasting"' in system_prompt
        if is_regression:
            return {
                "candidates": [
                    {
                        "name": "Random Forest",
                        "library": "sklearn",
                        "estimator": "RandomForestRegressor",
                        "hyperparams": {"n_estimators": 100, "max_depth": 8, "random_state": 0},
                        "rationale": "Mock mode default: robust nonlinear baseline.",
                    },
                    {
                        "name": "Ridge Regression",
                        "library": "sklearn",
                        "estimator": "Ridge",
                        "hyperparams": {"alpha": 1.0},
                        "rationale": "Mock mode default: fast linear baseline for comparison.",
                    },
                ]
            }
        return {
            "candidates": [
                {
                    "name": "Random Forest",
                    "library": "sklearn",
                    "estimator": "RandomForestClassifier",
                    "hyperparams": {"n_estimators": 100, "max_depth": 8, "random_state": 0},
                    "rationale": "Mock mode default: robust nonlinear baseline.",
                },
                {
                    "name": "Logistic Regression",
                    "library": "sklearn",
                    "estimator": "LogisticRegression",
                    "hyperparams": {"max_iter": 1000},
                    "rationale": "Mock mode default: fast interpretable baseline for comparison.",
                },
            ]
        }
    if node == "report":
        return (
            "MOCK-MODE REPORT (AUTOML_MOCK_LLM=1 — no real LLM was called).\n\n"
            "The pipeline profiled your data, applied the confirmed task specification, "
            "checked for target leakage, trained the candidate models listed in the results "
            "table (with hyperparameter tuning where enabled), and selected the best one by "
            "your chosen metric.\n\n"
            "Caveats: target-leakage detection is heuristic and may miss cases. "
            "Unset AUTOML_MOCK_LLM and configure a provider in config/models.yaml "
            "to get a real plain-language rationale here."
        )
    if node == "chat":
        return (
            "MOCK-MODE ANSWER (AUTOML_MOCK_LLM=1 — no real LLM was called). "
            "In a real run, this would answer your question using only this "
            "run's already-computed profile, insights, feature plan, "
            "training results, and report."
        )
    raise ValueError(f"mock mode has no canned response for node '{node}'")


def _call_anthropic(system: str, user: str, model: str, temperature: float, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


_OPENAI_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_openai_reasoning_model(model: str) -> bool:
    """Reasoning-family models (o1/o3/o4, gpt-5*) use max_completion_tokens
    instead of max_tokens, and reject any non-default temperature."""
    return model.startswith(_OPENAI_REASONING_MODEL_PREFIXES)


_REASONING_MODEL_MIN_COMPLETION_TOKENS = 2048


def _call_openai(system: str, user: str, model: str, temperature: float, max_tokens: int, json_mode: bool) -> str:
    from openai import OpenAI

    client = OpenAI()
    kwargs: dict[str, Any] = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    if _is_openai_reasoning_model(model):
        # Reasoning models spend part of max_completion_tokens on hidden
        # reasoning tokens before writing any visible output. A low budget
        # (as configured for cheap/fast nodes like understand_usecase) can be
        # fully consumed by reasoning, leaving an empty message.content. Floor
        # the budget and ask for minimal reasoning effort to leave room for
        # the actual answer.
        kwargs["max_completion_tokens"] = max(max_tokens, _REASONING_MODEL_MIN_COMPLETION_TOKENS)
        kwargs["reasoning_effort"] = "minimal"
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature

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
            f"OpenAI returned empty content (finish_reason={finish_reason}). "
            "This usually means the reasoning-token budget was exhausted before any "
            "visible output was written — increase max_tokens for this node in "
            "config/models.yaml or use a non-reasoning model."
        )
    return content


def _call_gemini(system: str, user: str, model: str, temperature: float, max_tokens: int, json_mode: bool) -> str:
    import google.generativeai as genai

    gen_config: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if json_mode:
        gen_config["response_mime_type"] = "application/json"

    gm = genai.GenerativeModel(model_name=model, system_instruction=system)
    resp = gm.generate_content(user, generation_config=gen_config)
    return resp.text


class LLMClient:
    """Call-count/token budget is tracked per run_id per CLAUDE.md's cost-control NFR."""

    def __init__(self) -> None:
        self._calls_per_run: dict[str, int] = {}

    def _check_budget(self, run_id: str) -> None:
        max_calls = _runtime_config()["budgets"]["max_llm_calls_per_run"]
        used = self._calls_per_run.get(run_id, 0)
        if used >= max_calls:
            raise LLMBudgetExceeded(f"run {run_id} exceeded max_llm_calls_per_run={max_calls}")

    def generate(
        self,
        run_id: str,
        node: str,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict[str, Any]] = None,
        retries: int = 1,
    ) -> Any:
        """Generate a completion for `node`. Returns a parsed dict if json_schema is
        given, else the raw text string. Every call is logged with the run_id
        (CLAUDE.md rule #7)."""
        self._check_budget(run_id)

        if os.environ.get("AUTOML_MOCK_LLM") == "1":
            self._calls_per_run[run_id] = self._calls_per_run.get(run_id, 0) + 1
            mocked = _mock_response(node, system_prompt)
            log_llm_call(
                run_id, node, "mock", "mock", system_prompt, user_prompt,
                mocked if isinstance(mocked, str) else json.dumps(mocked),
            )
            return mocked

        cfg = node_model_config(node)
        provider, model = cfg["provider"], cfg["model"]
        temperature = cfg.get("temperature", 0.0)
        max_tokens = cfg.get("max_tokens", 2048)

        effective_system = system_prompt
        if json_schema is not None:
            effective_system = (
                system_prompt
                + "\n\nRespond with ONLY a single JSON object matching this schema, "
                "no markdown fences, no commentary:\n"
                + json.dumps(json_schema)
            )

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

        raise LLMResponseError(f"node '{node}' failed after {retries + 1} attempts: {last_error}")


_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
