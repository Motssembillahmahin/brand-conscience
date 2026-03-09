"""LangGraph subgraph for Layer 3 — Creative Production."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from brand_conscience.common.logging import get_logger

logger = get_logger(__name__)


class CreativeState(TypedDict, total=False):
    """State for the creative subgraph."""

    passing_prompts: list[str]
    prompt_scores: dict  # prompt → score mapping
    generated_paths: list[str]
    evaluation_results: list[dict]
    approved_creative_ids: list[str]


def generate_images(state: CreativeState) -> dict[str, Any]:
    """Generate image variants for each passing prompt."""
    from brand_conscience.common.config import get_settings
    from brand_conscience.layer3_creative.gemini_client import GeminiClient

    settings = get_settings()
    client = GeminiClient()
    all_paths: list[str] = []

    for i, prompt in enumerate(state.get("passing_prompts", [])):
        paths = client.generate_variants(
            prompt=prompt,
            output_dir=f"creatives/batch_{i:03d}",
            n_variants=settings.creative.variants_per_prompt,
        )
        all_paths.extend(paths)

    return {"generated_paths": all_paths}


def evaluate_creatives(state: CreativeState) -> dict[str, Any]:
    """Run all generated images through the 4-gate evaluation pipeline."""
    from brand_conscience.layer3_creative.evaluation import EvaluationPipeline

    pipeline = EvaluationPipeline()
    results = pipeline.evaluate_batch(state.get("generated_paths", []))

    return {
        "evaluation_results": [
            {
                "image_path": r.image_path,
                "overall_result": r.overall_result.value,
                "quality_tier": r.quality_tier.value,
                "brand_alignment_score": r.brand_alignment_score,
                "originality_score": r.originality_score,
                "predicted_ctr": r.predicted_ctr,
                "rejection_reason": r.rejection_reason,
            }
            for r in results
        ]
    }


def store_approved(state: CreativeState) -> dict[str, Any]:
    """Store approved creatives in the database."""
    from brand_conscience.common.types import GateResult
    from brand_conscience.layer3_creative.asset_manager import AssetManager
    from brand_conscience.layer3_creative.evaluation import EvaluationResult, QualityTier

    manager = AssetManager()
    creative_ids: list[str] = []

    for r in state.get("evaluation_results", []):
        if r["overall_result"] == GateResult.PASSED.value:
            eval_result = EvaluationResult(
                image_path=r["image_path"],
                overall_result=GateResult.PASSED,
                quality_tier=QualityTier(r["quality_tier"]),
                brand_alignment_score=r["brand_alignment_score"],
                originality_score=r["originality_score"],
                predicted_ctr=r["predicted_ctr"],
            )
            cid = manager.store(
                evaluation=eval_result,
                prompt_text="",  # TODO: map back to prompt
                prompt_score=0.0,
            )
            creative_ids.append(cid)

    return {"approved_creative_ids": creative_ids}


def build_creative_graph() -> StateGraph:
    """Build the Layer 3 creative production subgraph."""
    graph = StateGraph(CreativeState)

    graph.add_node("generate_images", generate_images)
    graph.add_node("evaluate_creatives", evaluate_creatives)
    graph.add_node("store_approved", store_approved)

    graph.set_entry_point("generate_images")
    graph.add_edge("generate_images", "evaluate_creatives")
    graph.add_edge("evaluate_creatives", "store_approved")
    graph.add_edge("store_approved", END)

    return graph
