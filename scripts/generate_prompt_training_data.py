"""Generate prompt scorer training data using Claude as an LLM judge.

Generates diverse ad prompts using the PromptBuilder with varied contexts,
then scores each prompt via Anthropic Claude for predicted ad performance quality.

Usage:
    uv run python scripts/generate_prompt_training_data.py \
        --output data/prompt_performance.json \
        --n-samples 200

TODO(live-campaigns): When live campaign data is available, replace or augment
    this synthetic scoring with real performance metrics (CTR, conversion rate,
    ROAS) from Layer 5 feedback. See `_build_score_from_metrics()` stub below.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import click

from brand_conscience.common.config import load_settings
from brand_conscience.common.logging import configure_logging, get_logger
from brand_conscience.layer2_prompts.prompt_builder import PromptBuilder
from brand_conscience.layer2_prompts.templates import TemplateRegistry

logger = get_logger(__name__)

# ── Diverse context pools for prompt generation ──────────────────────────────

PRODUCTS = [
    "Widget Pro",
    "AeroFit Running Shoes",
    "Lumina Smart Lamp",
    "PureBlend Smoothie Maker",
    "NovaDrive SSD",
    "ZenBrew Coffee Maker",
    "TerraGlow Skincare Serum",
    "VoltEdge Power Bank",
    "SilkTouch Bedding Set",
    "SnapLens Camera Kit",
    "AquaPure Water Filter",
    "FlexCore Yoga Mat",
    "PixelFrame Digital Art Display",
    "CloudSync Wireless Earbuds",
    "IronForge Cast Iron Skillet",
]

BRAND_NAMES = [
    "Lumina Co.",
    "TerraGlow",
    "PureBlend",
    "VoltEdge",
    "ZenBrew",
    "AeroFit",
    "NovaTech",
    "SilkTouch Home",
    "SnapLens Optics",
    "FlexCore",
]

BRAND_TONES = [
    "modern and minimalist",
    "warm and approachable",
    "bold and energetic",
    "luxurious and refined",
    "playful and youthful",
    "eco-conscious and natural",
    "tech-forward and sleek",
    "rustic and authentic",
    "professional and trustworthy",
    "adventurous and daring",
]

BRAND_VALUES = [
    "quality and innovation",
    "sustainability and transparency",
    "performance and durability",
    "comfort and wellness",
    "creativity and self-expression",
    "simplicity and elegance",
    "community and connection",
    "precision and craftsmanship",
]

AUDIENCES = [
    "retargeting",
    "broad_interest",
    "lookalike",
    "custom_audience",
    "young professionals 25-35",
    "health-conscious millennials",
    "tech enthusiasts",
    "eco-friendly shoppers",
    "luxury buyers",
    "budget-conscious families",
    "fitness community",
    "creative professionals",
]

STYLES = [
    "modern, clean, professional",
    "vibrant and colorful pop art",
    "dark moody cinematic",
    "bright natural lifestyle",
    "minimalist flat design",
    "vintage retro aesthetic",
    "high-contrast editorial",
    "soft pastel dreamy",
    "urban streetwear grit",
    "elegant black and white",
]

SCENARIOS = [
    "morning routine in a sunlit apartment",
    "outdoor workout at sunrise",
    "cozy evening reading session",
    "busy professional in a modern office",
    "family gathering around a dinner table",
    "solo travel adventure",
    "friends at a rooftop party",
    "weekend farmers market visit",
    "peaceful yoga session in nature",
    "late night creative work session",
]

MOODS = [
    "confident and aspirational",
    "calm and serene",
    "energetic and exciting",
    "warm and nostalgic",
    "bold and rebellious",
    "fresh and optimistic",
    "cozy and intimate",
    "dramatic and powerful",
]

SEASONS = ["spring", "summer", "autumn", "winter", "holiday", "back-to-school"]

OFFERS = [
    "20% off first purchase",
    "free shipping this weekend",
    "buy one get one free",
    "limited edition bundle",
    "early access for members",
    "flash sale — 48 hours only",
]

ACTION_TYPES = ["launch", "refresh", "adjust"]

# ── LLM judge prompt ────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are an expert advertising creative director scoring ad image generation prompts.

Rate the following ad prompt on a scale from 0.0 to 1.0 based on these criteria:
- **Clarity** (0.25): Is the visual description specific and unambiguous?
- **Brand alignment** (0.25): Does it convey a coherent brand message?
- **Emotional impact** (0.25): Will it evoke the intended emotional response?
- **Actionability** (0.25): Can an image generator produce a compelling result from this?

Respond with ONLY a JSON object: {"score": <float>, "reasoning": "<one sentence>"}
Do not include any other text."""


def _random_context() -> dict:
    """Generate a randomized context dict for prompt building."""
    return {
        "product_name": random.choice(PRODUCTS),
        "brand_name": random.choice(BRAND_NAMES),
        "brand_tone": random.choice(BRAND_TONES),
        "brand_values": random.choice(BRAND_VALUES),
        "style": random.choice(STYLES),
        "scenario": random.choice(SCENARIOS),
        "mood": random.choice(MOODS),
        "season": random.choice(SEASONS),
        "offer_details": random.choice(OFFERS),
        "urgency_message": random.choice(
            ["act now", "limited time", "while supplies last", "don't miss out"]
        ),
        "key_message": random.choice(
            [
                "discover the difference",
                "elevate your everyday",
                "built for performance",
                "designed with care",
                "experience the future",
            ]
        ),
        "setting": random.choice(
            [
                "minimalist studio",
                "natural outdoor setting",
                "modern kitchen",
                "urban rooftop",
                "cozy living room",
            ]
        ),
        "narrative": random.choice(
            [
                "brand story",
                "customer transformation",
                "behind the scenes",
                "heritage and craft",
                "innovation journey",
            ]
        ),
        "theme": random.choice(["celebration", "renewal", "gratitude", "adventure", "warmth"]),
        "cultural_context": random.choice(
            ["contemporary", "multicultural", "local community", "global perspective"]
        ),
    }


def _score_prompt_with_claude(client: object, model: str, prompt: str) -> dict | None:
    """Score a single prompt using Claude as an LLM judge.

    Returns:
        Dict with 'score' and 'reasoning', or None on failure.
    """
    text = ""
    try:
        response = client.messages.create(  # type: ignore[union-attr]
            model=model,
            max_tokens=256,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f'Ad prompt to evaluate:\n"{prompt}"'},
            ],
        )
        text = response.content[0].text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        result = json.loads(text)
        score = float(result["score"])
        score = max(0.0, min(1.0, score))
        return {"score": score, "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("judge_parse_failed", error=str(exc), response=text[:200])
        return None
    except Exception as exc:
        logger.error("judge_api_failed", error=str(exc))
        return None


# ── TODO(live-campaigns): Real performance scoring ──────────────────────────
#
# def _build_score_from_metrics(campaign_id: str, creative_id: str) -> float:
#     """Compute a prompt quality score from real campaign performance.
#
#     When live campaign data is available in the database, this function
#     should replace or augment the LLM judge scores. Steps:
#
#     1. Query Layer 5 feedback for the creative's campaign metrics:
#        - CTR (click-through rate) — normalize to [0, 1] against category avg
#        - Conversion rate — normalize similarly
#        - ROAS (return on ad spend) — cap and scale
#        - Engagement rate (likes + comments + shares / impressions)
#
#     2. Weighted combination (tunable):
#        score = 0.3 * norm_ctr + 0.3 * norm_conversion + 0.25 * norm_roas
#                + 0.15 * norm_engagement
#
#     3. Use this score as the ground-truth label for retraining:
#        - Pair with the original prompt text
#        - Append to data/prompt_performance.json
#        - Trigger `make train-scorer` via Celery task
#
#     Integration point: brand_conscience.layer5_feedback.feedback_collector
#     DB query: brand_conscience.db.queries.get_aggregate_metrics()
#     Celery task: brand_conscience.tasks.run_model_retrain
#     """
#     raise NotImplementedError("Awaiting live campaign data")


@click.command()
@click.option("--output", required=True, type=click.Path(), help="Output JSON path")
@click.option("--n-samples", default=200, help="Number of prompt-score pairs to generate")
@click.option("--delay", default=1.0, help="Seconds between API calls (rate limiting)")
@click.option("--seed", default=42, help="Random seed for reproducibility")
def generate(output: str, n_samples: int, delay: float, seed: int) -> None:
    """Generate training data for the prompt scorer model."""
    random.seed(seed)

    settings = load_settings()
    configure_logging(settings.log_level, settings.log_format)

    if not settings.anthropic.api_key:
        logger.error("anthropic_api_key_missing")
        raise click.ClickException("ANTHROPIC_API_KEY is required. Set it in .env or environment.")

    # Initialize Anthropic client for prompt judging
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic.api_key)
    judge_model = settings.anthropic.model

    builder = PromptBuilder(registry=TemplateRegistry())

    dataset: list[dict[str, str | float]] = []
    failures = 0

    logger.info("generation_started", target_samples=n_samples)

    for i in range(n_samples):
        context = _random_context()
        action_type = random.choice(ACTION_TYPES)
        audience = random.choice(AUDIENCES)

        prompts = builder.build(
            strategic_decision={"audience_segment": audience},
            context=context,
            action_type=action_type,
            max_prompts=1,
        )

        if not prompts:
            failures += 1
            continue

        prompt_text = prompts[0]

        # Score via Claude judge
        result = _score_prompt_with_claude(client, judge_model, prompt_text)
        if result is None:
            failures += 1
            continue

        dataset.append(
            {
                "prompt": prompt_text,
                "score": result["score"],
                "reasoning": result["reasoning"],
                "template_context": {
                    "product": context["product_name"],
                    "audience": audience,
                    "action_type": action_type,
                    "style": context["style"],
                },
            }
        )

        if (i + 1) % 20 == 0:
            logger.info("generation_progress", completed=i + 1, total=n_samples, failures=failures)

        time.sleep(delay)

    # Save dataset
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, indent=2))

    logger.info(
        "generation_complete",
        samples=len(dataset),
        failures=failures,
        output=str(out_path),
        score_mean=sum(d["score"] for d in dataset) / max(len(dataset), 1),
    )


if __name__ == "__main__":
    generate()
