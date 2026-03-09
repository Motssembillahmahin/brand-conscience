"""Gate 3: Creative diversity / originality enforcer."""

from __future__ import annotations

import torch

from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.common.types import GateResult
from brand_conscience.models.embeddings.clip_encoder import CLIPEncoder
from brand_conscience.models.safety.diversity import DiversityEnforcer

logger = get_logger(__name__)


class OriginalityGate:
    """Ensure new creatives are sufficiently different from existing ones."""

    def __init__(
        self,
        clip_encoder: CLIPEncoder | None = None,
        diversity_enforcer: DiversityEnforcer | None = None,
    ) -> None:
        self._encoder = clip_encoder or CLIPEncoder()
        self._enforcer = diversity_enforcer or DiversityEnforcer(clip_encoder=self._encoder)
        self._active_embeddings: torch.Tensor = torch.empty(0, 768)

    def set_active_creatives(self, embeddings: torch.Tensor) -> None:
        """Set embeddings of currently active creatives."""
        self._active_embeddings = embeddings

    @traced(name="originality_gate", tags=["layer3", "gate", "originality"])
    def evaluate(self, image_path: str) -> tuple[GateResult, float]:
        """Evaluate originality of an image against active creatives.

        Args:
            image_path: Path to the generated image.

        Returns:
            (result, min_distance) tuple.
        """
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        new_embedding = self._encoder.encode_image(image)

        passes, min_dist = self._enforcer.check(
            new_embedding, self._active_embeddings
        )

        result = GateResult.PASSED if passes else GateResult.REJECTED

        logger.info(
            "originality_gate",
            image=image_path,
            min_distance=min_dist,
            result=result.value,
        )
        return result, min_dist
