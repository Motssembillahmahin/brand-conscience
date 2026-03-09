"""Reward computation for strategic and tactical RL agents."""

from __future__ import annotations

from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced

logger = get_logger(__name__)


@traced(name="compute_strategic_reward", tags=["rl", "reward"])
def strategic_reward(
    roas: float,
    audience_quality_score: float,
    budget_efficiency: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute reward for the strategic RL agent.

    Args:
        roas: Return on ad spend (revenue / spend).
        audience_quality_score: Quality metric for reached audience (0-1).
        budget_efficiency: Fraction of budget utilized effectively (0-1).
        weights: Optional weight overrides.

    Returns:
        Scalar reward value.
    """
    w = weights or {"roas": 0.5, "audience": 0.3, "efficiency": 0.2}

    # Normalize ROAS to 0-1 range (assume ROAS > 5 is excellent)
    normalized_roas = min(roas / 5.0, 1.0)

    reward = (
        w["roas"] * normalized_roas
        + w["audience"] * audience_quality_score
        + w["efficiency"] * budget_efficiency
    )

    logger.debug(
        "strategic_reward_computed",
        roas=roas,
        normalized_roas=normalized_roas,
        audience_quality=audience_quality_score,
        budget_efficiency=budget_efficiency,
        reward=reward,
    )
    return reward


@traced(name="compute_tactical_reward", tags=["rl", "reward"])
def tactical_reward(
    cpc_efficiency: float,
    delivery_pacing: float,
    spend_velocity_compliance: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute reward for the tactical RL agent.

    Args:
        cpc_efficiency: How close actual CPC is to target (1.0 = perfect, <1 = overpaying).
        delivery_pacing: How evenly budget is being spent (1.0 = perfect pacing).
        spend_velocity_compliance: Whether spend rate is within safety bounds (0-1).
        weights: Optional weight overrides.

    Returns:
        Scalar reward value.
    """
    w = weights or {"cpc": 0.4, "pacing": 0.3, "compliance": 0.3}

    reward = (
        w["cpc"] * cpc_efficiency
        + w["pacing"] * delivery_pacing
        + w["compliance"] * spend_velocity_compliance
    )

    logger.debug(
        "tactical_reward_computed",
        cpc_efficiency=cpc_efficiency,
        delivery_pacing=delivery_pacing,
        spend_velocity_compliance=spend_velocity_compliance,
        reward=reward,
    )
    return reward
