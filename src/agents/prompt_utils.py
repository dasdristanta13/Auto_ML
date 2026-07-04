"""Loads prompt templates from src/agents/prompts/*.md (kept out of code per
CLAUDE.md conventions) and fills {{TOKEN}} placeholders with JSON-serialized
state fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).parent / "prompts"


def render_prompt(template_name: str, **tokens: Any) -> str:
    path = PROMPTS_DIR / template_name
    text = path.read_text(encoding="utf-8")
    for key, value in tokens.items():
        placeholder = "{{" + key + "}}"
        rendered = value if isinstance(value, str) else json.dumps(value, indent=2, default=str)
        text = text.replace(placeholder, rendered)
    return text
