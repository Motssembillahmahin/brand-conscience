"""Integration test: Layer 2 → Layer 3 pipeline."""

from __future__ import annotations

import pytest

from brand_conscience.layer2_prompts.prompt_builder import PromptBuilder


@pytest.mark.integration
def test_prompt_builder_output_format():
    """Verify that prompts from the builder are valid strings for Gemini."""
    builder = PromptBuilder()
    prompts = builder.build(
        strategic_decision={"audience_segment": "retargeting"},
        context={
            "product_name": "Widget Pro",
            "brand_tone": "modern and minimalist",
        },
    )

    assert len(prompts) > 0
    has_product_ref = False
    for prompt in prompts:
        assert isinstance(prompt, str)
        assert len(prompt) > 20  # meaningful prompt
        if "Widget Pro" in prompt:
            has_product_ref = True
    assert has_product_ref, "At least one prompt should reference the product name"
