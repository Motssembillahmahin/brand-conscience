"""Experience replay buffer for RL training."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class Experience:
    """Single experience tuple."""

    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    log_prob: float
    value: float


@dataclass
class ExperienceBatch:
    """Batched experiences for training."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ReplayBuffer:
    """Fixed-size experience replay buffer with GAE advantage computation."""

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self._buffer: list[Experience] = []

    def add(self, experience: Experience) -> None:
        """Add an experience to the buffer."""
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)
        self._buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample a random batch of experiences."""
        return random.sample(self._buffer, min(batch_size, len(self._buffer)))

    def get_all(self) -> list[Experience]:
        """Return all experiences and clear the buffer."""
        experiences = self._buffer.copy()
        self._buffer.clear()
        return experiences

    def compute_batch(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> ExperienceBatch:
        """Compute advantages using GAE and return a training batch.

        Clears the buffer after computing.
        """
        experiences = self.get_all()
        if not experiences:
            raise ValueError("Buffer is empty")

        states = torch.stack([e.state for e in experiences])
        actions = torch.stack([e.action for e in experiences])
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32)
        log_probs = torch.tensor([e.log_prob for e in experiences], dtype=torch.float32)
        values = torch.tensor([e.value for e in experiences], dtype=torch.float32)

        # GAE computation
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(experiences))):
            next_value = 0.0 if t == len(experiences) - 1 else values[t + 1].item()

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values

        return ExperienceBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            log_probs=log_probs,
            values=values,
            advantages=advantages,
            returns=returns,
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
