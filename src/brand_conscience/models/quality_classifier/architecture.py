"""Quality classifier: CLIP embedding → 4-class MLP."""

from __future__ import annotations

import torch
import torch.nn as nn


class QualityClassifierNet(nn.Module):
    """MLP classifier on CLIP embeddings for image quality assessment.

    Input: 768-dim CLIP image embedding
    Hidden: [512, 256] with ReLU + dropout
    Output: 4 classes (excellent, good, acceptable, reject)
    """

    NUM_CLASSES = 4
    CLASS_NAMES = ["excellent", "good", "acceptable", "reject"]

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, self.NUM_CLASSES))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: CLIP embeddings of shape (batch, 768).

        Returns:
            Logits of shape (batch, 4).
        """
        return self.network(x)
