"""Tests for moment profile aggregation."""

from __future__ import annotations

from brand_conscience.common.types import ActionType
from brand_conscience.layer0_awareness.moment_profile import MomentProfileBuilder
from brand_conscience.layer0_awareness.signals import (
    BusinessSignal,
    CreativeSignal,
    CulturalSignal,
)


def test_empty_signals_produces_hold():
    builder = MomentProfileBuilder()
    profile = builder.build([], [], [])
    assert profile.urgency_score == 0.0
    assert profile.recommended_action == ActionType.HOLD


def test_high_business_severity_triggers_launch():
    builder = MomentProfileBuilder()
    signal = BusinessSignal(
        source="test",
        metric_name="revenue",
        category="electronics",
        change_pct=-0.25,
        severity=0.9,
    )
    profile = builder.build([signal], [], [])
    assert profile.urgency_score > 0.4
    assert profile.recommended_action in (ActionType.LAUNCH, ActionType.ADJUST)


def test_unsafe_cultural_triggers_pause():
    builder = MomentProfileBuilder()
    signal = CulturalSignal(
        source="test",
        topic="controversy",
        relevance=0.8,
        is_safe=False,
        safety_flags=[{"category": "political_controversy", "similarity": 0.7}],
    )
    profile = builder.build([], [signal], [])
    assert profile.recommended_action == ActionType.PAUSE


def test_creative_fatigue_triggers_refresh():
    builder = MomentProfileBuilder()
    signal = CreativeSignal(
        source="test",
        campaign_id="c1",
        creative_id="cr1",
        metric_name="ctr",
        fatigue_score=0.85,
    )
    profile = builder.build([], [], [signal])
    assert profile.recommended_action == ActionType.REFRESH


def test_moment_profile_to_dict():
    builder = MomentProfileBuilder()
    profile = builder.build([], [], [])
    d = profile.to_dict()
    assert "urgency_score" in d
    assert "recommended_action" in d
    assert "context_summary" in d
    assert isinstance(d["business_signals"], list)
