"""Tests for prompt builder."""

from __future__ import annotations

from brand_conscience.layer2_prompts.prompt_builder import PromptBuilder


def test_build_produces_prompts():
    builder = PromptBuilder()
    prompts = builder.build(
        strategic_decision={"audience_segment": "retargeting"},
        context={"product_name": "Widget X"},
    )
    assert len(prompts) > 0
    assert all(isinstance(p, str) for p in prompts)


def test_prompts_contain_context():
    builder = PromptBuilder()
    prompts = builder.build(
        strategic_decision={"audience_segment": "retargeting"},
        context={"product_name": "SuperGadget"},
    )
    assert any("SuperGadget" in p for p in prompts)


def test_max_prompts_limit():
    builder = PromptBuilder()
    prompts = builder.build(
        strategic_decision={},
        context={},
        max_prompts=2,
    )
    assert len(prompts) <= 2
