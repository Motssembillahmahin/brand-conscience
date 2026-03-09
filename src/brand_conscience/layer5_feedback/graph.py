"""LangGraph subgraph for Layer 5 — Feedback & Learning."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from brand_conscience.common.logging import get_logger

logger = get_logger(__name__)


class FeedbackState(TypedDict, total=False):
    """State for the feedback subgraph."""

    campaign_ids: list[str]
    collected_metrics: list[dict]
    drift_results: list[dict]
    retrain_decisions: list[dict]
    rewards: list[dict]


def collect_metrics(state: FeedbackState) -> dict[str, Any]:
    """Collect performance metrics for all active campaigns."""
    from brand_conscience.layer5_feedback.collector import MetricsCollector

    collector = MetricsCollector()
    metrics = []
    for cid in state.get("campaign_ids", []):
        # TODO: look up meta_campaign_id from database
        m = collector.collect(cid, "")
        metrics.append({"campaign_id": cid, **m})
    return {"collected_metrics": metrics}


def check_drift(state: FeedbackState) -> dict[str, Any]:
    """Run drift detection on model inputs."""
    # TODO: implement actual drift checking against stored distributions
    return {"drift_results": []}


def compute_rewards(state: FeedbackState) -> dict[str, Any]:
    """Compute RL rewards from collected metrics."""
    from brand_conscience.models.rl.reward import strategic_reward, tactical_reward

    rewards = []
    for m in state.get("collected_metrics", []):
        roas = m.get("roas", 0.0) or 0.0
        sr = strategic_reward(
            roas=roas,
            audience_quality_score=0.5,  # TODO: compute from actual data
            budget_efficiency=min(m.get("spend", 0) / 500.0, 1.0),
        )
        tr = tactical_reward(
            cpc_efficiency=0.5,  # TODO: compute from actual data
            delivery_pacing=0.8,
            spend_velocity_compliance=1.0,
        )
        rewards.append(
            {
                "campaign_id": m["campaign_id"],
                "strategic_reward": sr,
                "tactical_reward": tr,
            }
        )
    return {"rewards": rewards}


def handle_retrain(state: FeedbackState) -> dict[str, Any]:
    """Handle retraining decisions based on drift results."""
    from brand_conscience.layer5_feedback.model_updater import ModelUpdater

    updater = ModelUpdater()
    decisions = []
    for drift in state.get("drift_results", []):
        if drift.get("should_retrain", False):
            result = updater.trigger_retrain(
                model_name=drift["model_name"],
                reason=f"PSI={drift.get('psi', 0):.3f}",
            )
            decisions.append(result)
    return {"retrain_decisions": decisions}


def build_feedback_graph() -> StateGraph:
    """Build the Layer 5 feedback subgraph."""
    graph = StateGraph(FeedbackState)

    graph.add_node("collect_metrics", collect_metrics)
    graph.add_node("check_drift", check_drift)
    graph.add_node("compute_rewards", compute_rewards)
    graph.add_node("handle_retrain", handle_retrain)

    graph.set_entry_point("collect_metrics")
    graph.add_edge("collect_metrics", "check_drift")
    graph.add_edge("check_drift", "compute_rewards")
    graph.add_edge("compute_rewards", "handle_retrain")
    graph.add_edge("handle_retrain", END)

    return graph
