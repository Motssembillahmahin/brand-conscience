"""Budget allocation with constraints."""

from __future__ import annotations

from dataclasses import dataclass

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced

logger = get_logger(__name__)


@dataclass
class BudgetAllocation:
    """Budget allocation result."""

    daily_budget: float
    campaign_objective: str
    budget_fraction: float  # fraction of max budget used
    requires_approval: bool


class BudgetAllocator:
    """Allocate campaign budgets within constraints."""

    @traced(name="allocate_budget", tags=["layer1", "budget"])
    def allocate(
        self,
        budget_fraction: float,
        urgency_score: float,
        campaign_objective: str = "conversions",
    ) -> BudgetAllocation:
        """Allocate budget based on RL agent decision and urgency.

        Args:
            budget_fraction: Fraction of max budget (0-1) from RL agent.
            urgency_score: Current moment urgency (0-1).
            campaign_objective: Campaign objective type.

        Returns:
            BudgetAllocation with daily budget and approval requirement.
        """
        settings = get_settings()
        max_budget = settings.strategy.max_daily_budget
        approval_threshold = settings.deployment.spend_approval_threshold

        # Scale budget by urgency — higher urgency → closer to requested fraction
        urgency_multiplier = 0.5 + 0.5 * urgency_score
        effective_fraction = budget_fraction * urgency_multiplier

        daily_budget = max(
            settings.strategy.default_daily_budget * 0.1,  # minimum
            min(effective_fraction * max_budget, max_budget),
        )

        requires_approval = daily_budget > approval_threshold

        allocation = BudgetAllocation(
            daily_budget=round(daily_budget, 2),
            campaign_objective=campaign_objective,
            budget_fraction=effective_fraction,
            requires_approval=requires_approval,
        )

        logger.info(
            "budget_allocated",
            daily_budget=allocation.daily_budget,
            fraction=effective_fraction,
            urgency=urgency_score,
            requires_approval=requires_approval,
        )
        return allocation
