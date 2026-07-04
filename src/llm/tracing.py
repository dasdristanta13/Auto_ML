"""Full agent reasoning trace logging (CLAUDE.md rule #7).

Every LLM call — prompt, provider/model, response — is appended to a JSONL
file keyed by run_id so any run can be replayed/audited later. This is a
local stand-in for LangSmith; swap `log_llm_call` for a LangSmith client call
in production without touching call sites.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

LOG_DIR = Path(os.environ.get("AUTOML_LOG_DIR", "logs/traces"))


def log_llm_call(
    run_id: str,
    node: str,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response: str,
    error: Optional[str] = None,
    **extra: Any,
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "run_id": run_id,
        "node": node,
        "provider": provider,
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": response,
        "error": error,
        **extra,
    }
    path = LOG_DIR / f"{run_id}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_trace(run_id: str) -> list[dict[str, Any]]:
    path = LOG_DIR / f"{run_id}.jsonl"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
