"""Prompt scoring gate — filters prompts by quality score."""

from __future__ import annotations

from dataclasses import dataclass

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.models.prompt_scorer.inference import PromptScorer

logger = get_logger(__name__)


@dataclass
class ScoredPrompt:
    """A prompt with its quality score and gate result."""

    prompt: str
    score: float
    passed: bool


class ScoringGate:
    """Filter prompts based on prompt scorer model predictions."""

    def __init__(self, scorer: PromptScorer | None = None) -> None:
        self._scorer = scorer or PromptScorer()

    @traced(name="score_and_filter_prompts", tags=["layer2", "scoring"])
    def filter(self, prompts: list[str]) -> list[ScoredPrompt]:
        """Score all prompts and filter by threshold.

        Args:
            prompts: List of prompt strings from prompt builder.

        Returns:
            List of ScoredPrompt instances (both passed and failed).
        """
        settings = get_settings()
        threshold = settings.prompts.scorer_threshold

        scores = self._scorer.score_batch(prompts)

        results = []
        for prompt, score in zip(prompts, scores, strict=True):
            passed = score >= threshold
            results.append(ScoredPrompt(prompt=prompt, score=score, passed=passed))

        passed_count = sum(1 for r in results if r.passed)
        logger.info(
            "prompts_scored",
            total=len(prompts),
            passed=passed_count,
            rejected=len(prompts) - passed_count,
            threshold=threshold,
        )
        return results

    def get_passing_prompts(self, prompts: list[str]) -> list[str]:
        """Convenience method to get only passing prompt strings."""
        results = self.filter(prompts)
        return [r.prompt for r in results if r.passed]
