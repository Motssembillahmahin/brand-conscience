"""Actor-Critic neural networks for RL agents."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActorCriticNetwork(nn.Module):
    """Shared backbone with separate actor and critic heads.

    Used by both strategic and tactical RL agents with different dimensions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        continuous: bool = False,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        self.continuous = continuous

        # Shared backbone
        layers: list[nn.Module] = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        self.backbone = nn.Sequential(*layers)

        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(prev_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(prev_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(prev_dim, 1)

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits/params and state value.

        Args:
            state: State tensor of shape (batch, state_dim).

        Returns:
            (action_logits_or_mean, state_value) tuple.
        """
        features = self.backbone(state)
        value = self.critic(features).squeeze(-1)

        if self.continuous:
            action_mean = self.actor_mean(features)
            return action_mean, value
        else:
            action_logits = self.actor(features)
            return action_logits, value

    def get_action_distribution(
        self, state: torch.Tensor
    ) -> tuple[torch.distributions.Distribution, torch.Tensor]:
        """Get action distribution and state value.

        Returns:
            (distribution, value) tuple.
        """
        features = self.backbone(state)
        value = self.critic(features).squeeze(-1)

        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
        else:
            logits = self.actor(features)
            dist = torch.distributions.Categorical(logits=logits)

        return dist, value


class StrategicNetwork(ActorCriticNetwork):
    """Actor-Critic for the strategic agent (hourly decisions).

    Larger network: [512, 256] hidden dims.
    Discrete action space: audience segment × budget level.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[512, 256],
            continuous=False,
        )


class TacticalNetwork(ActorCriticNetwork):
    """Actor-Critic for the tactical agent (per-minute decisions).

    Smaller network: [256, 128] hidden dims.
    Continuous action space: bid multipliers per placement.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 128],
            continuous=True,
        )
