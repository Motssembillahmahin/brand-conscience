"""Strategic RL agent — hourly audience and budget decisions."""

from __future__ import annotations

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.layer0_awareness.signals import MomentProfile
from brand_conscience.layer1_strategy.audience_selector import AudienceSelector
from brand_conscience.layer1_strategy.budget_allocator import BudgetAllocator
from brand_conscience.layer1_strategy.state import StrategicState
from brand_conscience.models.rl.networks import StrategicNetwork
from brand_conscience.models.rl.ppo import PPOAgent

logger = get_logger(__name__)


class StrategicDecision:
    """Result of a strategic agent decision."""

    def __init__(
        self,
        audience_segment: str,
        daily_budget: float,
        campaign_objective: str,
        budget_fraction: float,
        requires_approval: bool,
        action_idx: int,
        log_prob: float,
        value: float,
    ) -> None:
        self.audience_segment = audience_segment
        self.daily_budget = daily_budget
        self.campaign_objective = campaign_objective
        self.budget_fraction = budget_fraction
        self.requires_approval = requires_approval
        self.action_idx = action_idx
        self.log_prob = log_prob
        self.value = value

    def to_dict(self) -> dict:
        return {
            "audience_segment": self.audience_segment,
            "daily_budget": self.daily_budget,
            "campaign_objective": self.campaign_objective,
            "budget_fraction": self.budget_fraction,
            "requires_approval": self.requires_approval,
        }


class StrategicAgent:
    """PPO-based strategic agent for audience and budget decisions.

    Runs hourly or on high-urgency triggers.
    """

    # 4 segments × 5 budget levels = 20 discrete actions
    ACTION_DIM = 20

    def __init__(self) -> None:
        settings = get_settings()
        rl_cfg = settings.models.strategic_rl  # type: ignore[attr-defined]

        self._network = StrategicNetwork(
            state_dim=StrategicState.STATE_DIM,
            action_dim=self.ACTION_DIM,
        )
        self._agent = PPOAgent(
            network=self._network,
            learning_rate=rl_cfg.learning_rate,
            gamma=rl_cfg.gamma,
            clip_epsilon=rl_cfg.clip_epsilon,
        )
        self._audience_selector = AudienceSelector()
        self._budget_allocator = BudgetAllocator()
        self._checkpoint_path = rl_cfg.checkpoint_path

    @traced(name="strategic_decide", tags=["layer1", "strategic"])
    def decide(
        self,
        moment_profile: MomentProfile,
        active_campaigns: int = 0,
        total_spend: float = 0.0,
        budget_remaining: float = 0.0,
    ) -> StrategicDecision:
        """Make a strategic decision based on the current moment.

        Args:
            moment_profile: Current MomentProfile from Layer 0.
            active_campaigns: Number of currently active campaigns.
            total_spend: Total spend so far today.
            budget_remaining: Remaining daily budget.

        Returns:
            StrategicDecision with audience and budget allocation.
        """
        from datetime import UTC, datetime

        now = datetime.now(UTC)

        state = StrategicState.encode(
            moment_profile=moment_profile,
            active_campaigns=active_campaigns,
            total_spend=total_spend,
            budget_remaining=budget_remaining,
            hour=now.hour,
            day_of_week=now.weekday(),
        )

        action, log_prob, value = self._agent.select_action(state)
        action_idx = int(action.item()) if action.dim() == 0 else int(action[0].item())

        decoded = StrategicState.decode_action(action_idx)
        audience = self._audience_selector.select(decoded["audience_segment"])
        budget = self._budget_allocator.allocate(
            budget_fraction=decoded["budget_fraction"],
            urgency_score=moment_profile.urgency_score,
        )

        decision = StrategicDecision(
            audience_segment=audience.name,
            daily_budget=budget.daily_budget,
            campaign_objective=budget.campaign_objective,
            budget_fraction=budget.budget_fraction,
            requires_approval=budget.requires_approval,
            action_idx=action_idx,
            log_prob=log_prob,
            value=value,
        )

        logger.info(
            "strategic_decision",
            audience=decision.audience_segment,
            budget=decision.daily_budget,
            urgency=moment_profile.urgency_score,
            action_idx=action_idx,
        )
        return decision

    def save_checkpoint(self) -> None:
        self._agent.save(self._checkpoint_path)

    def load_checkpoint(self) -> None:
        self._agent.load(self._checkpoint_path)
