"""End-to-end test: Full pipeline with mocked external APIs."""

from __future__ import annotations

import pytest


@pytest.mark.e2e
def test_full_pipeline_structure():
    """Verify the full pipeline graph can be built."""
    from brand_conscience.app import build_pipeline_graph

    graph = build_pipeline_graph()
    assert graph is not None


@pytest.mark.e2e
def test_pipeline_awareness_only():
    """Test running just the awareness layer of the pipeline."""
    from brand_conscience.layer0_awareness.business_monitor import BusinessMonitor
    from brand_conscience.layer0_awareness.creative_monitor import CreativeMonitor
    from brand_conscience.layer0_awareness.cultural_monitor import CulturalMonitor
    from brand_conscience.layer0_awareness.moment_profile import MomentProfileBuilder

    business = BusinessMonitor().collect_signals()
    cultural = CulturalMonitor().collect_signals()
    creative = CreativeMonitor().collect_signals()

    builder = MomentProfileBuilder()
    profile = builder.build(business, cultural, creative)

    assert profile.urgency_score >= 0.0
    assert profile.urgency_score <= 1.0
    assert profile.recommended_action is not None
