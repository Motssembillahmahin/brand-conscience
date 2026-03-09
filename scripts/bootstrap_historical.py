"""Bootstrap historical ad data: CLIP embed historical ads → performance mapping.

Usage:
    uv run python scripts/bootstrap_historical.py \
        --ads-dir /path/to/historical/ads --output data/embeddings.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch
from PIL import Image

from brand_conscience.common.config import load_settings
from brand_conscience.common.logging import configure_logging, get_logger
from brand_conscience.models.embeddings.clip_encoder import CLIPEncoder

logger = get_logger(__name__)


@click.command()
@click.option(
    "--ads-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory of historical ad images",
)
@click.option(
    "--metadata",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with performance metadata",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output path for embeddings tensor",
)
@click.option("--batch-size", default=32, help="Batch size for CLIP encoding")
def bootstrap(ads_dir: str, metadata: str, output: str, batch_size: int) -> None:
    """CLIP-embed historical ads and map to performance data."""
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_format)

    logger.info("bootstrap_start", ads_dir=ads_dir, metadata=metadata)

    # Load metadata
    with open(metadata) as f:
        meta = json.load(f)

    encoder = CLIPEncoder()
    ads_path = Path(ads_dir)

    all_embeddings = []
    all_performance = []

    image_files = sorted(ads_path.glob("*.png")) + sorted(ads_path.glob("*.jpg"))
    logger.info("found_images", count=len(image_files))

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i : i + batch_size]
        images = [Image.open(f).convert("RGB") for f in batch_files]
        embeddings = encoder.encode_images(images)
        all_embeddings.append(embeddings)

        for f in batch_files:
            perf = meta.get(f.stem, {"ctr": 0.0, "conversions": 0, "roas": 0.0})
            all_performance.append(perf)

        logger.info("batch_processed", batch=i // batch_size + 1, total=len(image_files))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "embeddings": torch.cat(all_embeddings, dim=0),
        "performance": all_performance,
        "file_names": [f.name for f in image_files],
    }
    torch.save(result, output_path)
    logger.info("bootstrap_complete", output=str(output_path), n_ads=len(image_files))


if __name__ == "__main__":
    bootstrap()
