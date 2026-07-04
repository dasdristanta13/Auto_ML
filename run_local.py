"""Local CLI entrypoint — runs the full agentic AutoML pipeline against a
local CSV without any external infra (Celery/Redis/S3/Postgres/Docker are all
swapped for local stand-ins, see CLAUDE.md-referenced modules for details).

Usage:
    python run_local.py --file tests/fixtures/imbalanced_classification.csv \\
        --description "predict which customers will churn" --target churned

Switch LLM providers per-node by editing config/models.yaml — no code changes
needed. Set the corresponding API key in .env (copy .env.example).
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv

from src.graph.build_graph import build_graph
from src.state import new_state


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the agentic AutoML pipeline locally.")
    parser.add_argument("--file", required=True, help="Path to a local CSV dataset")
    parser.add_argument("--description", required=True, help="Natural-language use case description")
    parser.add_argument("--run-id", default=None, help="Optional run id (default: random)")
    args = parser.parse_args()

    run_id = args.run_id or str(uuid.uuid4())[:8]
    state = new_state(run_id=run_id, dataset_path=args.file, use_case_description=args.description)

    print(f"Starting pipeline run {run_id} on {args.file}")
    graph = build_graph()
    final_state = graph.invoke(state, config={"recursion_limit": 100})

    print("\n=== RUN COMPLETE ===")
    print(f"status: {final_state.get('status')}")
    if final_state.get("errors"):
        print("errors encountered:")
        for err in final_state["errors"]:
            print(f"  - {err}")

    report = final_state.get("report", {})
    print("\n--- Report ---")
    print(report.get("narrative", "(no report generated)"))
    print(f"\nFull agent trace log: logs/traces/{run_id}.jsonl")

    Path("artifacts").mkdir(exist_ok=True)
    with open(f"artifacts/run_{run_id}_state.json", "w", encoding="utf-8") as f:
        json.dump(final_state, f, indent=2, default=str)
    print(f"Full final state written to artifacts/run_{run_id}_state.json")


if __name__ == "__main__":
    main()
