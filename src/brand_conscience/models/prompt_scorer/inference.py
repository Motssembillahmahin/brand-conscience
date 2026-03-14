"""Prompt scorer inference with batching and caching."""

from __future__ import annotations

from pathlib import Path

import torch

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.models.prompt_scorer.architecture import PromptScorerNet
from brand_conscience.models.prompt_scorer.tokenizer import PromptTokenizer

logger = get_logger(__name__)


class PromptScorer:
    """Score prompts for predicted ad performance quality."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        vocab_path: str | None = None,
    ) -> None:
        settings = get_settings()
        model_cfg = settings.models.prompt_scorer  # type: ignore[attr-defined]
        self._checkpoint_path = checkpoint_path or model_cfg.checkpoint_path
        self._vocab_path = vocab_path
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: PromptScorerNet | None = None
        self._tokenizer: PromptTokenizer | None = None

    def _load(self) -> None:
        if self._model is not None:
            return

        settings = get_settings()
        model_cfg = settings.models.prompt_scorer  # type: ignore[attr-defined]

        # Load tokenizer first so we know the correct vocab size
        if self._vocab_path and Path(self._vocab_path).exists():
            self._tokenizer = PromptTokenizer.load(self._vocab_path)
        else:
            self._tokenizer = PromptTokenizer()

        self._model = PromptScorerNet(
            vocab_size=self._tokenizer.vocab_size,
            d_model=model_cfg.hidden_dim,
            n_heads=model_cfg.heads,
            n_layers=model_cfg.layers,
        )

        path = Path(self._checkpoint_path)
        if path.exists():
            state = torch.load(path, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state)
            logger.info("prompt_scorer_loaded", path=str(path))
        else:
            logger.warning("prompt_scorer_no_checkpoint", path=str(path))

        self._model = self._model.to(self._device)
        self._model.eval()

    @traced(name="prompt_score", tags=["models", "prompt_scorer"])
    @torch.no_grad()
    def score(self, prompt: str) -> float:
        """Score a single prompt.

        Args:
            prompt: The ad prompt text.

        Returns:
            Score in range [0.0, 1.0].
        """
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None

        encoded = self._tokenizer.encode(prompt)
        input_ids = encoded["input_ids"].unsqueeze(0).to(self._device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(self._device)

        score = self._model(input_ids, attention_mask)
        return float(score.item())

    @traced(name="prompt_score_batch", tags=["models", "prompt_scorer"])
    @torch.no_grad()
    def score_batch(self, prompts: list[str]) -> list[float]:
        """Score a batch of prompts.

        Args:
            prompts: List of ad prompt texts.

        Returns:
            List of scores in range [0.0, 1.0].
        """
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None

        encoded = self._tokenizer.encode_batch(prompts)
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        scores = self._model(input_ids, attention_mask)
        return scores.tolist()

    def passes_threshold(self, score: float) -> bool:
        """Check if a score passes the configured threshold."""
        return score >= get_settings().prompts.scorer_threshold
