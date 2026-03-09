"""Creative asset storage and metadata management."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from brand_conscience.common.database import get_session
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.db.tables import Creative

if TYPE_CHECKING:
    from brand_conscience.layer3_creative.evaluation import EvaluationResult

logger = get_logger(__name__)


class AssetManager:
    """Manage creative assets — storage, metadata, and retrieval."""

    def __init__(self, storage_dir: str = "creatives") -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    @traced(name="store_creative", tags=["layer3", "asset"])
    def store(
        self,
        evaluation: EvaluationResult,
        prompt_text: str,
        prompt_score: float,
    ) -> str:
        """Store an approved creative with its metadata.

        Args:
            evaluation: Evaluation result from the pipeline.
            prompt_text: The prompt that generated this creative.
            prompt_score: Score from the prompt scorer.

        Returns:
            Creative ID.
        """
        creative_id = str(uuid.uuid4())

        with get_session() as session:
            creative = Creative(
                id=uuid.UUID(creative_id),
                prompt_text=prompt_text,
                prompt_score=prompt_score,
                image_path=evaluation.image_path,
                quality_tier=evaluation.quality_tier,
                brand_alignment_score=evaluation.brand_alignment_score,
                originality_score=evaluation.originality_score,
                predicted_ctr=evaluation.predicted_ctr,
                gate_results=evaluation.gate_results,
                created_at=datetime.now(UTC),
            )
            session.add(creative)

        logger.info(
            "creative_stored",
            creative_id=creative_id,
            quality=evaluation.quality_tier.value,
            predicted_ctr=evaluation.predicted_ctr,
        )
        return creative_id

    def get_active_creative_paths(self, campaign_id: str) -> list[str]:
        """Get image paths of active creatives for a campaign."""
        with get_session() as session:
            from brand_conscience.db.tables import Ad

            ads = (
                session.query(Ad)
                .join(Creative)
                .filter(Ad.ad_set.has(campaign_id=uuid.UUID(campaign_id)))
                .all()
            )
            return [ad.creative.image_path for ad in ads]
