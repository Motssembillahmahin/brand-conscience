"""Celery tasks for Layer 4 deployment."""

from __future__ import annotations

from brand_conscience.common.logging import bind_context, get_logger
from brand_conscience.common.tracing import traced

logger = get_logger(__name__)


@traced(name="run_tactical_cycle", tags=["layer4", "task"])
def run_tactical_cycle(campaign_id: str) -> dict:
    """Execute a tactical optimization cycle for a live campaign.

    Runs every few minutes to adjust bids and placements.
    """
    bind_context(layer="layer4_deployment", campaign_id=campaign_id)

    from brand_conscience.layer4_deployment.circuit_breaker import CircuitBreaker
    from brand_conscience.layer4_deployment.tactical_agent import TacticalAgent, TacticalState

    breaker = CircuitBreaker()

    # Check if circuit breaker is tripped
    if breaker.is_tripped:
        logger.warning("tactical_skipped_circuit_breaker", campaign_id=campaign_id)
        return {"status": "skipped", "reason": "circuit_breaker_tripped"}

    agent = TacticalAgent()

    # Fetch actual campaign metrics from database
    from brand_conscience.db.queries import get_aggregate_metrics, get_campaign

    campaign = get_campaign(campaign_id)
    if campaign is None:
        logger.warning("tactical_campaign_not_found", campaign_id=campaign_id)
        return {"status": "skipped", "reason": "campaign_not_found"}

    metrics = get_aggregate_metrics(campaign_id)
    state = TacticalState.encode(
        current_ctr=metrics.get("ctr", 0.0),
        current_cpc=metrics.get("cpc", 0.0),
        current_spend=metrics.get("spend", 0.0),
        daily_budget=campaign.daily_budget,
    )
    decision = agent.decide(state)

    logger.info(
        "tactical_cycle_complete",
        campaign_id=campaign_id,
        multipliers=decision["bid_multipliers"],
    )

    return {
        "status": "complete",
        "campaign_id": campaign_id,
        "adjustments": decision["bid_multipliers"],
    }


@traced(name="run_deployment_cycle", tags=["layer4", "task"])
def run_deployment_cycle(
    strategic_decision: dict,
    creative_ids: list[str],
) -> dict:
    """Execute a full deployment cycle — create campaign, set up A/B test, deploy.

    Called after Layer 3 produces approved creatives.
    """
    bind_context(layer="layer4_deployment")

    from brand_conscience.layer4_deployment.ab_testing import ThompsonSamplingMAB
    from brand_conscience.layer4_deployment.campaign_manager import CampaignManager

    manager = CampaignManager()
    mab = ThompsonSamplingMAB()

    # Create campaign
    campaign_id = manager.create(
        name=f"BC-Auto-{strategic_decision.get('audience_segment', 'general')}",
        objective=strategic_decision.get("campaign_objective", "conversions"),
        daily_budget=strategic_decision.get("daily_budget", 500.0),
    )

    # Set up A/B test
    groups = mab.setup_test(campaign_id=campaign_id, variant_ids=creative_ids)

    # Submit for deployment
    status = manager.submit_for_deployment(campaign_id)

    logger.info(
        "deployment_cycle_complete",
        campaign_id=campaign_id,
        status=status.value,
        n_variants=len(creative_ids),
    )

    return {
        "campaign_id": campaign_id,
        "status": status.value,
        "ab_groups": groups,
    }
