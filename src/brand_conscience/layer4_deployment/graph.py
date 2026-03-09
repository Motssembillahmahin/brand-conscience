"""LangGraph subgraph for Layer 4 — Deployment."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from brand_conscience.common.logging import get_logger

logger = get_logger(__name__)


class DeploymentState(TypedDict, total=False):
    """State for the deployment subgraph."""

    strategic_decision: dict
    approved_creative_ids: list[str]
    campaign_id: str
    meta_campaign_id: str
    ab_test_groups: list[str]
    deployment_status: str
    tactical_adjustments: dict


def create_campaign(state: DeploymentState) -> dict[str, Any]:
    """Create campaign structure on Meta."""
    from brand_conscience.layer4_deployment.campaign_manager import CampaignManager

    manager = CampaignManager()
    decision = state.get("strategic_decision", {})

    campaign_id = manager.create(
        name=f"BC-Auto-{decision.get('audience_segment', 'general')}",
        objective=decision.get("campaign_objective", "conversions"),
        daily_budget=decision.get("daily_budget", 500.0),
    )

    return {"campaign_id": campaign_id}


def setup_ab_test(state: DeploymentState) -> dict[str, Any]:
    """Set up A/B test groups for creative variants."""
    from brand_conscience.layer4_deployment.ab_testing import ThompsonSamplingMAB

    mab = ThompsonSamplingMAB()
    creative_ids = state.get("approved_creative_ids", [])

    if creative_ids:
        groups = mab.setup_test(
            campaign_id=state.get("campaign_id", ""),
            variant_ids=creative_ids,
        )
        return {"ab_test_groups": groups}

    return {"ab_test_groups": []}


def deploy_to_meta(state: DeploymentState) -> dict[str, Any]:
    """Deploy campaign to Meta Marketing API."""
    from brand_conscience.layer4_deployment.campaign_manager import CampaignManager

    manager = CampaignManager()
    campaign_id = state.get("campaign_id", "")

    try:
        status = manager.submit_for_deployment(campaign_id)
        return {"deployment_status": status.value}
    except Exception as exc:
        logger.error("deployment_failed", error=str(exc))
        return {"deployment_status": "failed"}


def build_deployment_graph() -> StateGraph:
    """Build the Layer 4 deployment subgraph."""
    graph = StateGraph(DeploymentState)

    graph.add_node("create_campaign", create_campaign)
    graph.add_node("setup_ab_test", setup_ab_test)
    graph.add_node("deploy_to_meta", deploy_to_meta)

    graph.set_entry_point("create_campaign")
    graph.add_edge("create_campaign", "setup_ab_test")
    graph.add_edge("setup_ab_test", "deploy_to_meta")
    graph.add_edge("deploy_to_meta", END)

    return graph
