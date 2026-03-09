"""Gate 1: CLIP quality classifier gate."""

from __future__ import annotations

from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.common.types import GateResult, QualityTier
from brand_conscience.models.embeddings.clip_encoder import CLIPEncoder
from brand_conscience.models.quality_classifier.inference import QualityClassifier

logger = get_logger(__name__)


class QualityGate:
    """Reject low-quality generated images using CLIP quality classifier."""

    def __init__(
        self,
        clip_encoder: CLIPEncoder | None = None,
        classifier: QualityClassifier | None = None,
    ) -> None:
        self._encoder = clip_encoder or CLIPEncoder()
        self._classifier = classifier or QualityClassifier()

    @traced(name="quality_gate_evaluate", tags=["layer3", "gate", "quality"])
    def evaluate(self, image_path: str) -> tuple[GateResult, QualityTier, float]:
        """Evaluate an image through the quality gate.

        Args:
            image_path: Path to the generated image.

        Returns:
            (result, tier, confidence) tuple.
        """
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        embedding = self._encoder.encode_image(image)
        tier = self._classifier.classify(embedding)
        passes = self._classifier.passes_gate(tier)

        result = GateResult.PASSED if passes else GateResult.REJECTED

        logger.info(
            "quality_gate",
            image=image_path,
            tier=tier.value,
            result=result.value,
        )
        return result, tier, 0.0  # confidence placeholder
