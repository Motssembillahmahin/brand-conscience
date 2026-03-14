"""Prompt scorer: Transformer regressor for prompt quality prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PromptScorerNet(nn.Module):
    """Transformer regressor that predicts prompt quality.

    Architecture: 4-layer transformer encoder (256d, 4 heads) with regression head.
    Input: tokenized prompt text
    Output: scalar score (0.0–1.0)
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        max_seq_len: int = 256,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(self.d_model))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Binary mask of shape (batch, seq_len). 1=attend, 0=ignore.

        Returns:
            Scores of shape (batch,) in range [0, 1].
        """
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.embedding(input_ids) + self.pos_encoding(positions)

        # Convert attention_mask to transformer format (True = ignore)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Pool: mean over non-masked positions
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        return self.regression_head(pooled).squeeze(-1)


class CLIPPromptScorerNet(nn.Module):
    """MLP regressor on 768-d CLIP text embeddings.

    Architecture: Linear(768→256) → ReLU → Dropout → Linear(256→64)
    → ReLU → Dropout → Linear(64→1) → Sigmoid
    Input: (batch, 768) float tensor (pre-computed CLIP text embeddings)
    Output: (batch,) scores in [0, 1]
    """

    def __init__(self, input_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: CLIP text embeddings of shape (batch, 768).

        Returns:
            Scores of shape (batch,) in range [0, 1].
        """
        return self.mlp(embeddings).squeeze(-1)
