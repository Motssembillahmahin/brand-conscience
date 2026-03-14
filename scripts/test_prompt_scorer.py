"""Test prompt scorer by generating images for top/bottom scored prompts.

Usage:
    uv run python scripts/test_prompt_scorer.py \
        --data data/prompt_performance_merged.json \
        --checkpoint model_checkpoints/prompt_scorer.pt \
        --vocab model_checkpoints/prompt_scorer_vocab.json \
        --output-dir test_outputs/prompt_scorer \
        --top-n 5

    uv run python scripts/test_prompt_scorer.py \
        --data data/prompt_performance_merged.json \
        --checkpoint model_checkpoints/prompt_scorer_clip.pt \
        --output-dir test_outputs/prompt_scorer_clip \
        --model-type clip_mlp \
        --top-n 5
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import click

from brand_conscience.common.config import load_settings
from brand_conscience.common.logging import configure_logging, get_logger
from brand_conscience.layer3_creative.gemini_client import GeminiClient
from brand_conscience.models.prompt_scorer.inference import PromptScorer

logger = get_logger(__name__)


def _generate_html_report(
    top_results: list[dict],
    bottom_results: list[dict],
    output_path: str,
) -> None:
    """Generate a self-contained HTML report with embedded images."""

    def _render_row(item: dict) -> str:
        img_html = ""
        if item.get("image_path") and Path(item["image_path"]).exists():
            img_data = Path(item["image_path"]).read_bytes()
            b64 = base64.b64encode(img_data).decode()
            img_html = f'<img src="data:image/png;base64,{b64}" width="256" />'
        else:
            img_html = "<em>generation failed</em>"

        return f"""<tr>
            <td style="max-width:400px;word-wrap:break-word">{item["prompt"]}</td>
            <td>{item["model_score"]:.4f}</td>
            <td>{item["judge_score"]:.4f}</td>
            <td>{img_html}</td>
        </tr>"""

    top_rows = "\n".join(_render_row(r) for r in top_results)
    bottom_rows = "\n".join(_render_row(r) for r in bottom_results)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Prompt Scorer Test Report</title>
<style>
    body {{ font-family: sans-serif; margin: 2rem; }}
    h1 {{ color: #333; }}
    h2 {{ margin-top: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
    img {{ border-radius: 4px; }}
    .top {{ border-left: 4px solid #2ecc71; }}
    .bottom {{ border-left: 4px solid #e74c3c; }}
</style>
</head>
<body>
<h1>Prompt Scorer Visual Validation</h1>
<p>Comparing images generated from the highest and lowest model-scored prompts.</p>

<h2 class="top">Top Scored Prompts (High Model Score)</h2>
<table>
<tr><th>Prompt</th><th>Model Score</th><th>Judge Score</th><th>Generated Image</th></tr>
{top_rows}
</table>

<h2 class="bottom">Bottom Scored Prompts (Low Model Score)</h2>
<table>
<tr><th>Prompt</th><th>Model Score</th><th>Judge Score</th><th>Generated Image</th></tr>
{bottom_rows}
</table>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html)


@click.command()
@click.option(
    "--data",
    required=True,
    type=click.Path(exists=True),
    help="JSON with prompt-performance pairs",
)
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True),
    help="Prompt scorer checkpoint path",
)
@click.option(
    "--vocab",
    default=None,
    type=click.Path(exists=True),
    help="Vocabulary JSON path (transformer mode only)",
)
@click.option(
    "--output-dir",
    default="test_outputs/prompt_scorer",
    help="Directory for images and report",
)
@click.option("--top-n", default=5, help="Number of prompts from each end to test")
@click.option("--delay", default=2.0, help="Seconds between Gemini calls (rate limit)")
@click.option(
    "--model-type",
    default="transformer",
    type=click.Choice(["transformer", "clip_mlp"]),
    help="Model architecture to test",
)
def test_scorer(
    data: str,
    checkpoint: str,
    vocab: str | None,
    output_dir: str,
    top_n: int,
    delay: float,
    model_type: str,
) -> None:
    """Test prompt scorer by generating images for top/bottom scored prompts."""
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_format)

    # Load training data
    with open(data) as f:
        dataset = json.load(f)
    logger.info("data_loaded", n_samples=len(dataset))

    # Score all prompts with the trained model
    scorer = PromptScorer(
        checkpoint_path=checkpoint,
        vocab_path=vocab,
        model_type=model_type,
    )
    prompts = [d["prompt"] for d in dataset]
    model_scores = scorer.score_batch(prompts)

    scored = [
        {"prompt": d["prompt"], "judge_score": d["score"], "model_score": ms}
        for d, ms in zip(dataset, model_scores, strict=True)
    ]
    scored.sort(key=lambda x: x["model_score"], reverse=True)

    top_prompts = scored[:top_n]
    bottom_prompts = scored[-top_n:]

    logger.info(
        "prompts_selected",
        top_range=f"{top_prompts[-1]['model_score']:.4f}-{top_prompts[0]['model_score']:.4f}",
        bottom_range=f"{bottom_prompts[-1]['model_score']:.4f}-{bottom_prompts[0]['model_score']:.4f}",
    )

    # Generate images
    gemini = GeminiClient()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_items = [("top", i, p) for i, p in enumerate(top_prompts)] + [
        ("bottom", i, p) for i, p in enumerate(bottom_prompts)
    ]

    for group, idx, item in all_items:
        img_path = str(out / f"{group}_{idx:02d}.png")
        logger.info("generating_image", group=group, idx=idx, prompt=item["prompt"][:80])
        result = gemini.generate_image(item["prompt"], img_path)
        item["image_path"] = result
        if idx < len(all_items) - 1:
            time.sleep(delay)

    # Generate report
    report_path = str(out / "report.html")
    _generate_html_report(top_prompts, bottom_prompts, report_path)

    n_success = sum(1 for _, _, item in all_items if item.get("image_path"))
    logger.info(
        "report_generated",
        path=report_path,
        total_images=len(all_items),
        successful=n_success,
    )
    click.echo(f"\nReport saved to: {report_path}")
    click.echo(f"Images generated: {n_success}/{len(all_items)}")


if __name__ == "__main__":
    test_scorer()
