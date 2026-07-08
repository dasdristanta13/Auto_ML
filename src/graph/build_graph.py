"""StateGraph definitions — declarative only; routing logic lives in
src/graph/routing.py, node logic lives in src/graph/nodes.py (deterministic)
and src/agents/ (LLM-backed), per CLAUDE.md conventions.

Four builders:
- build_graph():        full CLI pipeline, with stdin checkpoints for both
                        the task spec (if ambiguous) and the feature plan.
- build_intake_graph(): profile -> understand_usecase, then stops. Used by the
                        API so the user confirms/corrects the task spec in the
                        browser (PRD FR-9/FR-10) instead of stdin.
- build_prep_graph():   leakage_check -> eda -> feature_engineering, then
                        stops. Used by the API so the user reviews/approves
                        the EDA-informed feature plan (and any resampling
                        suggestion) in the browser before anything is applied.
- build_train_graph():  everything after feature-plan approval (apply the
                        approved plan onward). The API's approve-features
                        endpoint has already filtered feature_plan to the
                        approved steps and set resampling_plan before this
                        graph runs.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.explainability_node import explainability_node
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


def build_prep_graph():
    """API pre-approval pipeline: leakage check -> EDA -> LLM feature plan
    (EDA-informed, with a deterministic completeness floor). Stops here —
    the API's confirm_run starts this graph and the approve_features endpoint
    starts build_train_graph() once the user has reviewed the result."""
    graph = StateGraph(PipelineState)
    graph.add_node("leakage_check", nodes.leakage_check_node)
    graph.add_node("eda", nodes.eda_node)
    graph.add_node("feature_engineering", feature_engineering_node)

    graph.set_entry_point("leakage_check")
    graph.add_edge("leakage_check", "eda")
    graph.add_edge("eda", "feature_engineering")
    graph.add_conditional_edges(
        "feature_engineering",
        routing.route_after_feature_engineering_prep,
        {"done": END, "feature_engineering": "feature_engineering", "failed": END},
    )
    return graph.compile()


def build_train_graph():
    """API post-approval pipeline: apply the (human-approved) feature plan,
    then model search through the final report. If the approved plan fails
    to apply, this fails outright rather than silently regenerating an
    unreviewed plan (routing.route_after_apply_feature_plan_approved)."""
    graph = StateGraph(PipelineState)
    graph.add_node("apply_feature_plan", nodes.apply_feature_plan_node)
    graph.add_node("model_selection", model_selection_node)
    graph.add_node("dispatch_training", nodes.dispatch_training_node)
    graph.add_node("poll_training", nodes.poll_training_node)
    graph.add_node("evaluate", nodes.evaluate_node)
    graph.add_node("explainability", explainability_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("apply_feature_plan")
    graph.add_conditional_edges(
        "apply_feature_plan",
        routing.route_after_apply_feature_plan_approved,
        {"model_selection": "model_selection", "report": "report"},
    )
    graph.add_edge("model_selection", "dispatch_training")
    graph.add_edge("dispatch_training", "poll_training")
    graph.add_conditional_edges(
        "poll_training",
        routing.route_after_poll_training,
        {"evaluate": "evaluate", "poll_training": "poll_training"},
    )
    graph.add_edge("evaluate", "explainability")
    graph.add_edge("explainability", "report")
    graph.add_edge("report", END)
    return graph.compile()


def build_graph():
    """Full pipeline for run_local.py, with stdin-based human checkpoints for
    both the task spec (human_checkpoint, only if ambiguous) and the feature
    plan (feature_approval_checkpoint, always)."""
    graph = StateGraph(PipelineState)
    graph.add_node("profile", nodes.profile_node)
    graph.add_node("understand_usecase", understand_usecase_node)
    graph.add_node("human_checkpoint", nodes.human_checkpoint_node)
    graph.add_node("leakage_check", nodes.leakage_check_node)
    graph.add_node("eda", nodes.eda_node)
    graph.add_node("feature_engineering", feature_engineering_node)
    graph.add_node("feature_approval_checkpoint", nodes.feature_approval_checkpoint_node)
    graph.add_node("apply_feature_plan", nodes.apply_feature_plan_node)
    graph.add_node("model_selection", model_selection_node)
    graph.add_node("dispatch_training", nodes.dispatch_training_node)
    graph.add_node("poll_training", nodes.poll_training_node)
    graph.add_node("evaluate", nodes.evaluate_node)
    graph.add_node("explainability", explainability_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("profile")
    graph.add_edge("profile", "understand_usecase")
    graph.add_edge("understand_usecase", "human_checkpoint")
    graph.add_edge("human_checkpoint", "leakage_check")
    graph.add_edge("leakage_check", "eda")
    graph.add_edge("eda", "feature_engineering")

    graph.add_conditional_edges(
        "feature_engineering",
        routing.route_after_feature_engineering,
        {
            "apply_feature_plan": "feature_approval_checkpoint",
            "feature_engineering": "feature_engineering",
            "report": "report",
        },
    )
    graph.add_edge("feature_approval_checkpoint", "apply_feature_plan")
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
        {"evaluate": "evaluate", "poll_training": "poll_training"},
    )
    graph.add_edge("evaluate", "explainability")
    graph.add_edge("explainability", "report")
    graph.add_edge("report", END)
    return graph.compile()
