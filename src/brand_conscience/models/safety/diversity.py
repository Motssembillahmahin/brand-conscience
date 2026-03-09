"""Creative diversity enforcer using CLIP embeddings."""

from __future__ import annotations

import torch

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.models.embeddings.clip_encoder import CLIPEncoder

logger = get_logger(__name__)


class DiversityEnforcer:
    """Ensure creative diversity by checking CLIP embedding distances.

    New creatives must have minimum cosine distance from all active creatives
    in the same campaign to prevent ad fatigue.
    """

    def __init__(
        self,
        clip_encoder: CLIPEncoder | None = None,
        min_distance: float | None = None,
    ) -> None:
        settings = get_settings()
        self._encoder = clip_encoder or CLIPEncoder()
        self._min_distance = min_distance or settings.safety.diversity.min_distance

    @traced(name="diversity_check", tags=["safety", "diversity"])
    def check(
        self,
        new_embedding: torch.Tensor,
        existing_embeddings: torch.Tensor,
    ) -> tuple[bool, float]:
        """Check if a new creative is sufficiently different from existing ones.

        Args:
            new_embedding: CLIP embedding of new creative (768,).
            existing_embeddings: CLIP embeddings of active creatives (N, 768).

        Returns:
            Tuple of (passes, min_distance_to_existing).
        """
        if existing_embeddings.shape[0] == 0:
            return True, 1.0

        similarities = self._encoder.cosine_similarity(
            new_embedding.unsqueeze(0), existing_embeddings
        )
        if similarities.dim() == 0:
            similarities = similarities.unsqueeze(0)

        # Distance = 1 - similarity
        distances = 1.0 - similarities
        min_dist = float(distances.min().item())
        passes = min_dist >= self._min_distance

        if not passes:
            logger.warning(
                "diversity_violation",
                min_distance=min_dist,
                threshold=self._min_distance,
            )

        return passes, min_dist

    @traced(name="diversity_score_portfolio", tags=["safety", "diversity"])
    def portfolio_diversity_score(
        self, embeddings: torch.Tensor
    ) -> float:
        """Compute overall diversity score for a set of creatives.

        Returns the mean pairwise distance (0 = all identical, 1 = maximally diverse).
        """
        if embeddings.shape[0] < 2:
            return 1.0

        # Pairwise similarities
        sim_matrix = self._encoder.cosine_similarity(embeddings, embeddings)
        n = embeddings.shape[0]

        # Get upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        pairwise_distances = 1.0 - sim_matrix[mask]

        return float(pairwise_distances.mean().item())
