"""Gate 4: Performance prediction gate."""

from __future__ import annotations

import torch

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.common.types import GateResult
from brand_conscience.models.embeddings.clip_encoder import CLIPEncoder

logger = get_logger(__name__)


class PerformanceGate:
    """Predict ad performance from CLIP embeddings and historical patterns.

    Uses nearest-neighbor lookup in historical embedding→performance mapping.
    """

    def __init__(
        self,
        clip_encoder: CLIPEncoder | None = None,
        threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self._encoder = clip_encoder or CLIPEncoder()
        self._threshold = threshold or settings.creative.performance_prediction_threshold
        self._historical_embeddings: torch.Tensor | None = None
        self._historical_performance: list[float] = []

    def load_historical_data(
        self,
        embeddings: torch.Tensor,
        performance_scores: list[float],
    ) -> None:
        """Load historical embedding → performance mapping."""
        self._historical_embeddings = embeddings
        self._historical_performance = performance_scores

    @traced(name="performance_gate", tags=["layer3", "gate", "performance"])
    def evaluate(self, image_path: str) -> tuple[GateResult, float]:
        """Predict performance and gate the creative.

        Args:
            image_path: Path to the generated image.

        Returns:
            (result, predicted_ctr) tuple.
        """
        if self._historical_embeddings is None or len(self._historical_performance) == 0:
            logger.warning("no_historical_data", msg="Passing by default")
            return GateResult.PASSED, 0.0

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        embedding = self._encoder.encode_image(image)

        # k-NN prediction: average performance of top-5 most similar historical ads
        similarities = self._encoder.cosine_similarity(
            embedding.unsqueeze(0), self._historical_embeddings
        )
        k = min(5, len(self._historical_performance))
        top_k_indices = similarities.topk(k).indices.tolist()

        predicted_ctr = sum(
            self._historical_performance[i] for i in top_k_indices
        ) / k

        result = (
            GateResult.PASSED
            if predicted_ctr >= self._threshold
            else GateResult.REJECTED
        )

        logger.info(
            "performance_gate",
            image=image_path,
            predicted_ctr=predicted_ctr,
            threshold=self._threshold,
            result=result.value,
        )
        return result, predicted_ctr
