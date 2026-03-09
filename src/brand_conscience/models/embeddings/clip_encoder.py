"""OpenCLIP wrapper for image and text embeddings."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced

logger = get_logger(__name__)


class CLIPEncoder:
    """Wrapper around OpenCLIP for generating image and text embeddings.

    Uses ViT-L-14 pretrained on LAION-2B. Produces 768-dimensional embeddings.
    Model weights are frozen — inference only.
    """

    def __init__(
        self,
        model_name: str | None = None,
        pretrained: str | None = None,
        device: str | None = None,
    ) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.models.clip.model_name  # type: ignore[attr-defined]
        self._pretrained = pretrained or settings.models.clip.pretrained  # type: ignore[attr-defined]
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        """Lazy-load the CLIP model."""
        if self._model is not None:
            return

        import open_clip

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained
        )
        self._tokenizer = open_clip.get_tokenizer(self._model_name)
        self._model = self._model.to(self._device)
        self._model.eval()
        logger.info(
            "clip_model_loaded",
            model=self._model_name,
            pretrained=self._pretrained,
            device=self._device,
        )

    @traced(name="clip_encode_image", tags=["models", "clip"])
    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image to a normalized embedding vector.

        Returns:
            Tensor of shape (768,) — L2-normalized.
        """
        self._load()
        preprocessed = self._preprocess(image).unsqueeze(0).to(self._device)
        embedding = self._model.encode_image(preprocessed)
        return F.normalize(embedding.squeeze(0), dim=-1).cpu()

    @traced(name="clip_encode_images_batch", tags=["models", "clip"])
    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Encode a batch of PIL images.

        Returns:
            Tensor of shape (N, 768) — L2-normalized.
        """
        self._load()
        batch = torch.stack([self._preprocess(img) for img in images]).to(self._device)
        embeddings = self._model.encode_image(batch)
        return F.normalize(embeddings, dim=-1).cpu()

    @traced(name="clip_encode_text", tags=["models", "clip"])
    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of text strings.

        Returns:
            Tensor of shape (N, 768) — L2-normalized.
        """
        self._load()
        tokens = self._tokenizer(texts).to(self._device)
        embeddings = self._model.encode_text(tokens)
        return F.normalize(embeddings, dim=-1).cpu()

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between two sets of embeddings.

        Args:
            a: Tensor of shape (N, D) or (D,)
            b: Tensor of shape (M, D) or (D,)

        Returns:
            Tensor of shape (N, M) or scalar.
        """
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        return (a @ b.T).squeeze()

    @property
    def embedding_dim(self) -> int:
        return 768
