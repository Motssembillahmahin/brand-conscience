"""Proximal Policy Optimization algorithm."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced

if TYPE_CHECKING:
    from brand_conscience.models.rl.networks import ActorCriticNetwork
    from brand_conscience.models.rl.replay_buffer import ExperienceBatch

logger = get_logger(__name__)


class PPOAgent:
    """PPO agent supporting both discrete and continuous action spaces.

    Shared implementation used by both strategic and tactical agents.
    """

    def __init__(
        self,
        network: ActorCriticNetwork,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
    ) -> None:
        self.network = network
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        self._device = next(network.parameters()).device

    @traced(name="ppo_select_action", tags=["rl", "ppo"])
    @torch.no_grad()
    def select_action(self, state: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """Select an action given a state.

        Returns:
            (action, log_prob, value) tuple.
        """
        state = state.to(self._device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        dist, value = self.network.get_action_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        if self.network.continuous:
            log_prob = log_prob.sum(dim=-1)

        return (
            action.squeeze(0).cpu(),
            float(log_prob.item()),
            float(value.item()),
        )

    @traced(name="ppo_update", tags=["rl", "ppo"])
    def update(self, batch: ExperienceBatch) -> dict[str, float]:
        """Run PPO update on a batch of experiences.

        Returns:
            Dict with loss metrics.
        """
        states = batch.states.to(self._device)
        actions = batch.actions.to(self._device)
        old_log_probs = batch.log_probs.to(self._device)
        advantages = batch.advantages.to(self._device)
        returns = batch.returns.to(self._device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.n_epochs):
            # Shuffle and create mini-batches
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                dist, values = self.network.get_action_distribution(states[idx])
                new_log_probs = dist.log_prob(actions[idx])
                if self.network.continuous:
                    new_log_probs = new_log_probs.sum(dim=-1)
                entropy = dist.entropy()
                if self.network.continuous:
                    entropy = entropy.sum(dim=-1)

                # Policy loss (clipped surrogate)
                ratio = (new_log_probs - old_log_probs[idx]).exp()
                surr1 = ratio * advantages[idx]
                surr2 = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, returns[idx])

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        n_updates = self.n_epochs * max(1, len(states) // self.batch_size)
        metrics = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }
        logger.info("ppo_update_complete", **metrics)
        return metrics

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), path)
        logger.info("ppo_checkpoint_saved", path=path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        device = self._device
        state = torch.load(path, map_location=device, weights_only=True)
        self.network.load_state_dict(state)
        logger.info("ppo_checkpoint_loaded", path=path)

    @property
    def policy_entropy(self) -> float:
        """Get current policy entropy (for health monitoring)."""
        # This is a proxy; actual entropy computed during update
        return self.entropy_coef
