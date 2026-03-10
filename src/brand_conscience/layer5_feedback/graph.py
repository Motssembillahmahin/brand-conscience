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
    from brand_conscience.db.queries import get_campaign
    from brand_conscience.layer5_feedback.collector import MetricsCollector

    collector = MetricsCollector()
    metrics = []
    for cid in state.get("campaign_ids", []):
        # Look up meta_campaign_id from campaign record
        campaign = get_campaign(cid)
        meta_campaign_id = campaign.meta_campaign_id if campaign else ""
        m = collector.collect(cid, meta_campaign_id or "")
        metrics.append({"campaign_id": cid, **m})
    return {"collected_metrics": metrics}


def check_drift(state: FeedbackState) -> dict[str, Any]:
    """Run PSI drift detection on collected metric distributions."""
    import numpy as np

    from brand_conscience.layer5_feedback.drift_detector import DriftDetector

    detector = DriftDetector()
    results: list[dict] = []
    collected = state.get("collected_metrics", [])

    if len(collected) < 5:
        return {"drift_results": results}

    for metric_name in ("ctr", "cpc", "roas"):
        values = [m.get(metric_name, 0.0) for m in collected if m.get(metric_name) is not None]
        if len(values) < 4:
            continue

        arr = np.array(values, dtype=np.float64)
        midpoint = len(arr) // 2
        reference = arr[:midpoint]
        current = arr[midpoint:]

        if len(reference) < 2 or len(current) < 2:
            continue

        severity, psi = detector.check_drift(reference, current)
        results.append(
            {
                "metric": metric_name,
                "psi": psi,
                "severity": severity.value,
                "should_retrain": detector.should_retrain(psi),
                "model_name": f"performance_{metric_name}",
            }
        )

    return {"drift_results": results}


def compute_rewards(state: FeedbackState) -> dict[str, Any]:
    """Compute RL rewards from collected metrics."""
    from brand_conscience.models.rl.reward import strategic_reward, tactical_reward

    rewards = []
    for m in state.get("collected_metrics", []):
        roas = m.get("roas", 0.0) or 0.0
        spend = m.get("spend", 0.0) or 0.0
        cpc = m.get("cpc", 0.0) or 0.0

        # Audience quality score: proxy from ROAS (higher ROAS = better audience)
        audience_quality = min(roas / 2.0, 1.0)
        budget_efficiency = min(spend / 500.0, 1.0)

        sr = strategic_reward(
            roas=roas,
            audience_quality_score=audience_quality,
            budget_efficiency=budget_efficiency,
        )

        # CPC efficiency: lower CPC relative to $5 benchmark = better
        cpc_efficiency = max(0.0, 1.0 - cpc / 5.0)
        # Delivery pacing: fraction of budget spent
        delivery_pacing = budget_efficiency

        tr = tactical_reward(
            cpc_efficiency=cpc_efficiency,
            delivery_pacing=delivery_pacing,
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
