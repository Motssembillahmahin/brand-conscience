"""Celery tasks for Layer 3 creative production."""

from __future__ import annotations

from brand_conscience.common.logging import bind_context, get_logger
from brand_conscience.common.tracing import traced

logger = get_logger(__name__)


@traced(name="run_creative_cycle", tags=["layer3", "task"])
def run_creative_cycle(
    passing_prompts: list[str],
    prompt_scores: dict[str, float] | None = None,
) -> dict:
    """Execute a full creative generation and evaluation cycle.

    Called after Layer 2 produces passing prompts.
    """
    bind_context(layer="layer3_creative")

    from brand_conscience.common.config import get_settings
    from brand_conscience.common.types import GateResult
    from brand_conscience.layer3_creative.asset_manager import AssetManager
    from brand_conscience.layer3_creative.evaluation import EvaluationPipeline
    from brand_conscience.layer3_creative.gemini_client import GeminiClient

    settings = get_settings()
    client = GeminiClient()
    pipeline = EvaluationPipeline()
    manager = AssetManager()

    all_paths: list[str] = []
    for i, prompt in enumerate(passing_prompts):
        paths = client.generate_variants(
            prompt=prompt,
            output_dir=f"creatives/batch_{i:03d}",
            n_variants=settings.creative.variants_per_prompt,
        )
        all_paths.extend(paths)

    results = pipeline.evaluate_batch(all_paths)

    approved_ids: list[str] = []
    for r in results:
        if r.overall_result == GateResult.PASSED:
            cid = manager.store(
                evaluation=r,
                prompt_text="",
                prompt_score=prompt_scores.get("", 0.0) if prompt_scores else 0.0,
            )
            approved_ids.append(cid)

    logger.info(
        "creative_cycle_complete",
        generated=len(all_paths),
        approved=len(approved_ids),
    )

    return {
        "generated": len(all_paths),
        "approved": len(approved_ids),
        "creative_ids": approved_ids,
    }
