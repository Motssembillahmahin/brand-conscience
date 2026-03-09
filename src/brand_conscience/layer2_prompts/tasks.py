"""Celery tasks for Layer 2 prompt engineering."""

from __future__ import annotations

from brand_conscience.common.logging import bind_context, get_logger
from brand_conscience.common.tracing import traced

logger = get_logger(__name__)


@traced(name="run_prompt_cycle", tags=["layer2", "task"])
def run_prompt_cycle(strategic_decision: dict, context: dict) -> dict:
    """Execute a prompt building and scoring cycle.

    Called after Layer 1 produces a strategic decision.
    """
    bind_context(layer="layer2_prompts")

    from brand_conscience.layer2_prompts.prompt_builder import PromptBuilder
    from brand_conscience.layer2_prompts.scoring_gate import ScoringGate

    builder = PromptBuilder()
    prompts = builder.build(
        strategic_decision=strategic_decision,
        context=context,
    )

    gate = ScoringGate()
    results = gate.filter(prompts)
    passing = [r.prompt for r in results if r.passed]

    logger.info(
        "prompt_cycle_complete",
        built=len(prompts),
        passing=len(passing),
    )

    return {
        "prompts": [
            {"prompt": r.prompt, "score": r.score, "passed": r.passed}
            for r in results
        ],
        "passing_prompts": passing,
    }
