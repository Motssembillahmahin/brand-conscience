"""Strategic state vector for the RL agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from brand_conscience.common.logging import get_logger

if TYPE_CHECKING:
    from brand_conscience.layer0_awareness.signals import MomentProfile

logger = get_logger(__name__)


class StrategicState:
    """Encode the current business state into a vector for the strategic RL agent.

    State vector components:
    - Moment profile features (urgency, signal counts, action encoding)
    - Portfolio state (active campaigns, total spend, budget remaining)
    - Time features (hour, day of week)
    """

    STATE_DIM = 19

    @staticmethod
    def encode(
        moment_profile: MomentProfile,
        active_campaigns: int = 0,
        total_spend: float = 0.0,
        budget_remaining: float = 0.0,
        hour: int = 0,
        day_of_week: int = 0,
    ) -> torch.Tensor:
        """Encode state into a fixed-dimension tensor.

        Returns:
            Tensor of shape (STATE_DIM,).
        """
        # Action type one-hot (5 values)
        action_map = {"launch": 0, "adjust": 1, "pause": 2, "refresh": 3, "hold": 4}
        action_idx = action_map.get(moment_profile.recommended_action.value, 4)
        action_onehot = [0.0] * 5
        action_onehot[action_idx] = 1.0

        features = [
            moment_profile.urgency_score,
            len(moment_profile.business_signals) / 10.0,  # normalized
            len(moment_profile.cultural_signals) / 10.0,
            len(moment_profile.creative_signals) / 10.0,
            *action_onehot,
            float(active_campaigns) / 20.0,
            total_spend / 10000.0,
            budget_remaining / 10000.0,
            hour / 24.0,
            day_of_week / 7.0,
            # Derived features
            1.0 if moment_profile.urgency_score > 0.7 else 0.0,
            any(not s.is_safe for s in moment_profile.cultural_signals) * 1.0,
            max((s.fatigue_score for s in moment_profile.creative_signals), default=0.0),
            len(moment_profile.affected_categories) / 5.0,
            len(moment_profile.affected_audiences) / 5.0,
        ]

        return torch.tensor(features, dtype=torch.float32)

    @staticmethod
    def decode_action(action_idx: int, settings: dict | None = None) -> dict:
        """Decode an action index into audience segment and budget level.

        The action space is: n_segments × n_budget_levels.
        Default: 4 segments × 5 budget levels = 20 actions.
        """
        segments = ["broad_interest", "retargeting", "lookalike", "custom_audience"]
        budget_levels = [0.2, 0.4, 0.6, 0.8, 1.0]  # fraction of max budget

        n_budget = len(budget_levels)
        segment_idx = action_idx // n_budget
        budget_idx = action_idx % n_budget

        segment_idx = min(segment_idx, len(segments) - 1)
        budget_idx = min(budget_idx, len(budget_levels) - 1)

        return {
            "audience_segment": segments[segment_idx],
            "budget_fraction": budget_levels[budget_idx],
        }
