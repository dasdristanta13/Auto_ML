"""Local web entrypoint — serves the API + frontend at http://127.0.0.1:8000.

    python run_server.py

No API keys? Set AUTOML_MOCK_LLM=1 in .env to exercise the full flow with
canned LLM responses. With keys, configure providers per-node in
config/models.yaml (anthropic / openai / gemini).
"""

from __future__ import annotations

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
