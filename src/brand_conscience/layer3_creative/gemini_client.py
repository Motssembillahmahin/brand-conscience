"""Google Gemini image generation client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced

logger = get_logger(__name__)


class GeminiClient:
    """Generate images using Google Gemini API."""

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.gemini.api_key
        self._model = settings.gemini.model
        self._max_retries = settings.gemini.max_retries
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @traced(name="gemini_generate_image", tags=["layer3", "gemini"])
    def generate_image(
        self,
        prompt: str,
        output_path: str,
        aspect_ratio: str = "1:1",
    ) -> str | None:
        """Generate a single image from a prompt.

        Args:
            prompt: Image generation prompt.
            output_path: Path to save the generated image.
            aspect_ratio: Aspect ratio (e.g., '1:1', '16:9', '9:16').

        Returns:
            Path to saved image, or None if generation failed.
        """
        for attempt in range(self._max_retries):
            try:
                client = self._get_client()
                response = client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config={
                        "response_modalities": ["IMAGE"],
                    },
                )

                # Extract image from response
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                            Path(output_path).write_bytes(part.inline_data.data)
                            logger.info(
                                "image_generated",
                                path=output_path,
                                prompt=prompt[:80],
                            )
                            return output_path

                logger.warning(
                    "gemini_no_image_in_response",
                    attempt=attempt + 1,
                    prompt=prompt[:80],
                )
            except Exception as exc:
                logger.error(
                    "gemini_generation_failed",
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt == self._max_retries - 1:
                    return None

        return None

    @traced(name="gemini_generate_variants", tags=["layer3", "gemini"])
    def generate_variants(
        self,
        prompt: str,
        output_dir: str,
        n_variants: int = 3,
    ) -> list[str]:
        """Generate multiple image variants for a prompt.

        Args:
            prompt: Image generation prompt.
            output_dir: Directory to save generated images.
            n_variants: Number of variants to generate.

        Returns:
            List of paths to saved images.
        """
        paths: list[str] = []
        for i in range(n_variants):
            output_path = str(Path(output_dir) / f"variant_{i:03d}.png")
            result = self.generate_image(prompt, output_path)
            if result:
                paths.append(result)

        logger.info(
            "variants_generated",
            requested=n_variants,
            successful=len(paths),
            prompt=prompt[:80],
        )
        return paths
