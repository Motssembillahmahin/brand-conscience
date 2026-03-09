"""Celery tasks for Layer 1 strategy."""

from __future__ import annotations

from brand_conscience.common.logging import bind_context, get_logger
from brand_conscience.common.tracing import traced

logger = get_logger(__name__)


@traced(name="run_strategic_cycle", tags=["layer1", "task"])
def run_strategic_cycle(moment_profile: dict) -> dict:
    """Execute a strategic decision cycle.

    Called after Layer 0 produces a MomentProfile with sufficient urgency.
    """
    bind_context(layer="layer1_strategy")

    from brand_conscience.layer0_awareness.signals import MomentProfile
    from brand_conscience.layer1_strategy.strategic_agent import StrategicAgent
    from brand_conscience.common.types import ActionType

    profile = MomentProfile(
        id=moment_profile.get("id", ""),
        urgency_score=moment_profile.get("urgency_score", 0.0),
        recommended_action=ActionType(
            moment_profile.get("recommended_action", "hold")
        ),
        affected_categories=moment_profile.get("affected_categories", []),
        affected_audiences=moment_profile.get("affected_audiences", []),
        context_summary=moment_profile.get("context_summary", ""),
    )

    agent = StrategicAgent()
    decision = agent.decide(moment_profile=profile)

    logger.info(
        "strategic_cycle_complete",
        audience=decision.audience_segment,
        budget=decision.daily_budget,
    )
    return decision.to_dict()
