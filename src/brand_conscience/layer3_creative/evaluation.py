"""4-gate creative evaluation pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field

from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.common.types import GateResult, QualityTier
from brand_conscience.layer3_creative.gates.brand_alignment import BrandAlignmentGate
from brand_conscience.layer3_creative.gates.originality_gate import OriginalityGate
from brand_conscience.layer3_creative.gates.performance_gate import PerformanceGate
from brand_conscience.layer3_creative.gates.quality_gate import QualityGate
from brand_conscience.models.embeddings.clip_encoder import CLIPEncoder

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result of running a creative through all 4 gates."""

    image_path: str
    overall_result: GateResult = GateResult.REJECTED
    quality_tier: QualityTier = QualityTier.REJECT
    brand_alignment_score: float = 0.0
    originality_score: float = 0.0
    predicted_ctr: float = 0.0
    gate_results: dict = field(default_factory=dict)
    rejection_reason: str | None = None


class EvaluationPipeline:
    """Orchestrate the 4-gate creative evaluation pipeline.

    Gates run sequentially — if a creative fails any gate, subsequent gates
    are skipped (fail-fast).

    Order:
    1. Quality (CLIP classifier)
    2. Brand Alignment (CLIP cosine similarity)
    3. Originality (diversity enforcer)
    4. Performance Prediction (historical k-NN)
    """

    def __init__(self, clip_encoder: CLIPEncoder | None = None) -> None:
        encoder = clip_encoder or CLIPEncoder()
        self._quality_gate = QualityGate(clip_encoder=encoder)
        self._brand_gate = BrandAlignmentGate(clip_encoder=encoder)
        self._originality_gate = OriginalityGate(clip_encoder=encoder)
        self._performance_gate = PerformanceGate(clip_encoder=encoder)

    @traced(name="evaluate_creative", tags=["layer3", "evaluation"])
    def evaluate(self, image_path: str) -> EvaluationResult:
        """Run a creative through all 4 evaluation gates.

        Args:
            image_path: Path to the generated image.

        Returns:
            EvaluationResult with scores from each gate.
        """
        result = EvaluationResult(image_path=image_path)

        # Gate 1: Quality
        q_result, tier, _ = self._quality_gate.evaluate(image_path)
        result.quality_tier = tier
        result.gate_results["quality"] = {
            "result": q_result.value,
            "tier": tier.value,
        }
        if q_result == GateResult.REJECTED:
            result.rejection_reason = f"Quality gate: tier={tier.value}"
            logger.info("creative_rejected", gate="quality", image=image_path)
            return result

        # Gate 2: Brand Alignment
        b_result, b_score = self._brand_gate.evaluate(image_path)
        result.brand_alignment_score = b_score
        result.gate_results["brand_alignment"] = {
            "result": b_result.value,
            "score": b_score,
        }
        if b_result == GateResult.REJECTED:
            result.rejection_reason = f"Brand alignment gate: score={b_score:.3f}"
            logger.info("creative_rejected", gate="brand_alignment", image=image_path)
            return result

        # Gate 3: Originality
        o_result, o_dist = self._originality_gate.evaluate(image_path)
        result.originality_score = o_dist
        result.gate_results["originality"] = {
            "result": o_result.value,
            "min_distance": o_dist,
        }
        if o_result == GateResult.REJECTED:
            result.rejection_reason = f"Originality gate: distance={o_dist:.3f}"
            logger.info("creative_rejected", gate="originality", image=image_path)
            return result

        # Gate 4: Performance Prediction
        p_result, p_ctr = self._performance_gate.evaluate(image_path)
        result.predicted_ctr = p_ctr
        result.gate_results["performance"] = {
            "result": p_result.value,
            "predicted_ctr": p_ctr,
        }
        if p_result == GateResult.REJECTED:
            result.rejection_reason = f"Performance gate: predicted_ctr={p_ctr:.4f}"
            logger.info("creative_rejected", gate="performance", image=image_path)
            return result

        # All gates passed
        result.overall_result = GateResult.PASSED
        logger.info(
            "creative_approved",
            image=image_path,
            quality=tier.value,
            brand_score=b_score,
            originality=o_dist,
            predicted_ctr=p_ctr,
        )
        return result

    def evaluate_batch(self, image_paths: list[str]) -> list[EvaluationResult]:
        """Evaluate a batch of creatives."""
        return [self.evaluate(path) for path in image_paths]
