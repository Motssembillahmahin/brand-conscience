"""Quality classifier inference wrapper."""

from __future__ import annotations

from pathlib import Path

import torch

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.common.types import QualityTier
from brand_conscience.models.quality_classifier.architecture import QualityClassifierNet

logger = get_logger(__name__)


class QualityClassifier:
    """Classify CLIP image embeddings into quality tiers."""

    TIER_MAP = {
        0: QualityTier.EXCELLENT,
        1: QualityTier.GOOD,
        2: QualityTier.ACCEPTABLE,
        3: QualityTier.REJECT,
    }

    def __init__(self, checkpoint_path: str | None = None) -> None:
        settings = get_settings()
        self._checkpoint_path = (
            checkpoint_path or settings.models.quality_classifier.checkpoint_path  # type: ignore[attr-defined]
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: QualityClassifierNet | None = None

    def _load(self) -> None:
        if self._model is not None:
            return

        self._model = QualityClassifierNet(
            hidden_dims=get_settings().models.quality_classifier.hidden_dims,  # type: ignore[attr-defined]
            dropout=get_settings().models.quality_classifier.dropout,  # type: ignore[attr-defined]
        )

        path = Path(self._checkpoint_path)
        if path.exists():
            state = torch.load(path, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state)
            logger.info("quality_classifier_loaded", path=str(path))
        else:
            logger.warning("quality_classifier_no_checkpoint", path=str(path))

        self._model = self._model.to(self._device)
        self._model.eval()

    @traced(name="quality_classify", tags=["models", "quality"])
    @torch.no_grad()
    def classify(self, embedding: torch.Tensor) -> QualityTier:
        """Classify a single CLIP embedding.

        Args:
            embedding: Tensor of shape (768,).

        Returns:
            QualityTier enum value.
        """
        self._load()
        assert self._model is not None
        logits = self._model(embedding.unsqueeze(0).to(self._device))
        pred = logits.argmax(dim=-1).item()
        return self.TIER_MAP[int(pred)]

    @traced(name="quality_classify_batch", tags=["models", "quality"])
    @torch.no_grad()
    def classify_batch(self, embeddings: torch.Tensor) -> list[QualityTier]:
        """Classify a batch of CLIP embeddings.

        Args:
            embeddings: Tensor of shape (N, 768).

        Returns:
            List of QualityTier values.
        """
        self._load()
        assert self._model is not None
        logits = self._model(embeddings.to(self._device))
        preds = logits.argmax(dim=-1).tolist()
        return [self.TIER_MAP[int(p)] for p in preds]

    def passes_gate(self, tier: QualityTier) -> bool:
        """Check if a quality tier passes the gate."""
        passing = get_settings().creative.quality_gate_classes
        return tier.value in passing
