"""Gate 2: CLIP cosine similarity for brand alignment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from brand_conscience.common.config import get_settings

if TYPE_CHECKING:
    import torch

from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.common.types import GateResult
from brand_conscience.models.embeddings.clip_encoder import CLIPEncoder

logger = get_logger(__name__)


class BrandAlignmentGate:
    """Check if generated images align with brand reference embeddings."""

    def __init__(
        self,
        clip_encoder: CLIPEncoder | None = None,
        brand_embeddings: torch.Tensor | None = None,
        threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self._encoder = clip_encoder or CLIPEncoder()
        self._brand_embeddings = brand_embeddings
        self._threshold = threshold or settings.creative.brand_alignment_threshold

    def set_brand_references(self, embeddings: torch.Tensor) -> None:
        """Set brand reference embeddings."""
        self._brand_embeddings = embeddings

    @traced(name="brand_alignment_gate", tags=["layer3", "gate", "brand"])
    def evaluate(self, image_path: str) -> tuple[GateResult, float]:
        """Evaluate brand alignment for an image.

        Args:
            image_path: Path to the generated image.

        Returns:
            (result, similarity_score) tuple.
        """
        from PIL import Image

        if self._brand_embeddings is None:
            logger.warning("no_brand_references", msg="Passing by default")
            return GateResult.PASSED, 1.0

        image = Image.open(image_path).convert("RGB")
        image_embedding = self._encoder.encode_image(image)

        similarity = self._encoder.cosine_similarity(image_embedding, self._brand_embeddings)
        score = float(similarity.max().item()) if similarity.dim() > 0 else float(similarity.item())

        result = GateResult.PASSED if score >= self._threshold else GateResult.REJECTED

        logger.info(
            "brand_alignment_gate",
            image=image_path,
            score=score,
            threshold=self._threshold,
            result=result.value,
        )
        return result, score
