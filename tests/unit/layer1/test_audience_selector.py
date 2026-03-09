"""Tests for audience selector."""

from __future__ import annotations

from brand_conscience.layer1_strategy.audience_selector import AudienceSelector


def test_select_known_segment():
    selector = AudienceSelector()
    segment = selector.select("retargeting")
    assert segment.name == "retargeting"
    assert segment.estimated_size > 0


def test_select_unknown_falls_back():
    selector = AudienceSelector()
    segment = selector.select("nonexistent_segment")
    assert segment.name == "broad_interest"


def test_available_segments():
    selector = AudienceSelector()
    segments = selector.get_available_segments()
    assert "broad_interest" in segments
    assert "retargeting" in segments
    assert "lookalike" in segments
    assert "custom_audience" in segments


def test_meta_targeting():
    selector = AudienceSelector()
    segment = selector.select("retargeting")
    targeting = selector.get_meta_targeting(segment)
    assert "geo_locations" in targeting
