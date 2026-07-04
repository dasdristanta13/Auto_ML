"""StateGraph definitions — declarative only; routing logic lives in
src/graph/routing.py, node logic lives in src/graph/nodes.py (deterministic)
and src/agents/ (LLM-backed), per CLAUDE.md conventions.

Three builders:
- build_graph():        full CLI pipeline with a stdin human checkpoint.
- build_intake_graph(): profile -> understand_usecase, then stops. Used by the
                        API so the user confirms/corrects the task spec in the
                        browser (PRD FR-9/FR-10) instead of stdin.
- build_main_graph():   everything after confirmation (leakage check onward).
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.feature_engineering_node import feature_engineering_node
from src.agents.model_selection_node import model_selection_node
from src.agents.report_node import report_node
from src.agents.understand_usecase_node import understand_usecase_node
from src.graph import nodes, routing
from src.state import PipelineState


def build_intake_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("profile", nodes.profile_node)
    graph.add_node("understand_usecase", understand_usecase_node)
    graph.set_entry_point("profile")
    graph.add_edge("profile", "understand_usecase")
    graph.add_edge("understand_usecase", END)
    return graph.compile()


def _add_main_stages(graph: StateGraph, entry: str) -> None:
    graph.add_node("leakage_check", nodes.leakage_check_node)
    graph.add_node("feature_engineering", feature_engineering_node)
    graph.add_node("apply_feature_plan", nodes.apply_feature_plan_node)
    graph.add_node("model_selection", model_selection_node)
    graph.add_node("dispatch_training", nodes.dispatch_training_node)
    graph.add_node("poll_training", nodes.poll_training_node)
    graph.add_node("evaluate", nodes.evaluate_node)
    graph.add_node("report", report_node)

    graph.add_edge(entry, "leakage_check")
    graph.add_edge("leakage_check", "feature_engineering")

    graph.add_conditional_edges(
        "feature_engineering",
        routing.route_after_feature_engineering,
        {
            "apply_feature_plan": "apply_feature_plan",
            "feature_engineering": "feature_engineering",
            "report": "report",
        },
    )
    graph.add_conditional_edges(
        "apply_feature_plan",
        routing.route_after_apply_feature_plan,
        {
            "model_selection": "model_selection",
            "feature_engineering": "feature_engineering",
            "report": "report",
        },
    )

    graph.add_edge("model_selection", "dispatch_training")
    graph.add_edge("dispatch_training", "poll_training")
    graph.add_conditional_edges(
        "poll_training",
        routing.route_after_poll_training,
        {
            "evaluate": "evaluate",
            "poll_training": "poll_training",
        },
    )
    graph.add_edge("evaluate", "report")
    graph.add_edge("report", END)


def build_main_graph():
    """Post-confirmation pipeline. The API confirm endpoint has already set
    the task spec (human_confirmed=True) before this graph is invoked."""
    graph = StateGraph(PipelineState)
    _add_main_stages(graph, entry="__start__")
    return graph.compile()


def build_graph():
    """Full pipeline for run_local.py, with the stdin-based human checkpoint."""
    graph = StateGraph(PipelineState)
    graph.add_node("profile", nodes.profile_node)
    graph.add_node("understand_usecase", understand_usecase_node)
    graph.add_node("human_checkpoint", nodes.human_checkpoint_node)

    graph.set_entry_point("profile")
    graph.add_edge("profile", "understand_usecase")
    graph.add_edge("understand_usecase", "human_checkpoint")
    _add_main_stages(graph, entry="human_checkpoint")
    return graph.compile()
