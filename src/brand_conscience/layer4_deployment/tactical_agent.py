"""Tactical RL agent — per-minute bid and placement optimization."""

from __future__ import annotations

import torch

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.models.rl.networks import TacticalNetwork
from brand_conscience.models.rl.ppo import PPOAgent

logger = get_logger(__name__)


class TacticalState:
    """Encode current campaign metrics into a state vector for the tactical agent."""

    STATE_DIM = 12

    @staticmethod
    def encode(
        current_ctr: float = 0.0,
        current_cpc: float = 0.0,
        current_spend: float = 0.0,
        daily_budget: float = 0.0,
        hours_remaining: float = 24.0,
        current_bid: float = 0.0,
        target_cpc: float = 0.0,
        impressions: int = 0,
        clicks: int = 0,
        conversions: int = 0,
        frequency: float = 0.0,
        spend_velocity: float = 0.0,
    ) -> torch.Tensor:
        """Encode tactical state.

        Returns:
            Tensor of shape (STATE_DIM,).
        """
        budget_pacing = current_spend / max(daily_budget, 1.0)
        return torch.tensor(
            [
                current_ctr,
                min(current_cpc / 10.0, 1.0),
                budget_pacing,
                hours_remaining / 24.0,
                min(current_bid / 10.0, 1.0),
                min(target_cpc / 10.0, 1.0) if target_cpc > 0 else 0.5,
                min(impressions / 100000, 1.0),
                min(clicks / 5000, 1.0),
                min(conversions / 100, 1.0),
                min(frequency / 10.0, 1.0),
                min(spend_velocity, 1.0),
                1.0 if budget_pacing > 0.8 else 0.0,
            ],
            dtype=torch.float32,
        )


class TacticalAgent:
    """PPO-based tactical agent for real-time bid optimization.

    Runs every few minutes on live campaigns.
    """

    # Actions: bid multipliers for different placements
    ACTION_DIM = 4  # Feed, Stories, Reels, Other

    def __init__(self) -> None:
        settings = get_settings()
        rl_cfg = settings.models.tactical_rl  # type: ignore[attr-defined]

        self._network = TacticalNetwork(
            state_dim=TacticalState.STATE_DIM,
            action_dim=self.ACTION_DIM,
        )
        self._agent = PPOAgent(
            network=self._network,
            learning_rate=rl_cfg.learning_rate,
            gamma=rl_cfg.gamma,
            clip_epsilon=rl_cfg.clip_epsilon,
        )
        self._checkpoint_path = rl_cfg.checkpoint_path

    @traced(name="tactical_decide", tags=["layer4", "tactical"])
    def decide(self, state: torch.Tensor) -> dict:
        """Make tactical bid adjustments.

        Args:
            state: Tactical state vector.

        Returns:
            Dict with bid multipliers per placement.
        """
        settings = get_settings()
        action, log_prob, value = self._agent.select_action(state)

        # Convert continuous actions to bid multipliers (centered around 1.0)
        multipliers = torch.sigmoid(action) * 2.0  # range [0, 2]

        # Cap multipliers
        max_mult = settings.tactical.max_bid_multiplier
        warn_mult = settings.tactical.warning_bid_multiplier

        placements = ["feed", "stories", "reels", "other"]
        result = {}
        for i, placement in enumerate(placements):
            mult = float(multipliers[i].item()) if i < len(multipliers) else 1.0
            mult = min(mult, max_mult)

            if mult > warn_mult:
                logger.warning(
                    "high_bid_multiplier",
                    placement=placement,
                    multiplier=mult,
                    warning_threshold=warn_mult,
                )

            result[placement] = round(mult, 3)

        logger.info("tactical_decision", multipliers=result)
        return {
            "bid_multipliers": result,
            "log_prob": log_prob,
            "value": value,
        }

    def save_checkpoint(self) -> None:
        self._agent.save(self._checkpoint_path)

    def load_checkpoint(self) -> None:
        self._agent.load(self._checkpoint_path)
