"""Brand safety classifier using CLIP text embeddings."""

from __future__ import annotations

import torch

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.models.embeddings.clip_encoder import CLIPEncoder

logger = get_logger(__name__)


class BrandSafetyClassifier:
    """Screen text content for brand safety risks using CLIP embeddings.

    Compares input text embeddings against pre-computed risk topic embeddings.
    Flags content with cosine similarity above threshold.
    """

    def __init__(
        self,
        clip_encoder: CLIPEncoder | None = None,
        risk_categories: list[str] | None = None,
        threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self._encoder = clip_encoder or CLIPEncoder()
        self._risk_categories = (
            risk_categories or settings.safety.brand_safety.risk_categories
        )
        self._threshold = threshold or settings.safety.brand_safety.similarity_threshold
        self._risk_embeddings: torch.Tensor | None = None

    def _ensure_risk_embeddings(self) -> None:
        """Compute risk topic embeddings if not already done."""
        if self._risk_embeddings is not None:
            return
        self._risk_embeddings = self._encoder.encode_text(self._risk_categories)
        logger.info(
            "risk_embeddings_computed",
            n_categories=len(self._risk_categories),
        )

    @traced(name="brand_safety_check", tags=["safety", "brand"])
    def check(self, text: str) -> tuple[bool, list[dict[str, float]]]:
        """Check text for brand safety risks.

        Args:
            text: The text content to screen.

        Returns:
            Tuple of (is_safe, flagged_categories).
            flagged_categories is a list of dicts with 'category' and 'similarity' keys.
        """
        self._ensure_risk_embeddings()
        assert self._risk_embeddings is not None

        text_embedding = self._encoder.encode_text([text])
        similarities = self._encoder.cosine_similarity(
            text_embedding, self._risk_embeddings
        )

        if similarities.dim() == 0:
            similarities = similarities.unsqueeze(0)

        flagged: list[dict[str, float]] = []
        for i, sim in enumerate(similarities.tolist()):
            if sim > self._threshold:
                flagged.append({
                    "category": self._risk_categories[i],
                    "similarity": sim,
                })

        is_safe = len(flagged) == 0
        if not is_safe:
            logger.warning("brand_safety_flagged", text=text[:100], flagged=flagged)

        return is_safe, flagged

    @traced(name="brand_safety_check_batch", tags=["safety", "brand"])
    def check_batch(self, texts: list[str]) -> list[tuple[bool, list[dict[str, float]]]]:
        """Check multiple texts for brand safety."""
        return [self.check(text) for text in texts]
