"""Tests for feedback collector."""

from __future__ import annotations

from brand_conscience.layer5_feedback.collector import MetricsCollector


def test_fetch_from_meta_returns_dict():
    collector = MetricsCollector()
    result = collector._fetch_from_meta("test-campaign")
    assert isinstance(result, dict)
    assert "impressions" in result
    assert "clicks" in result
    assert "spend" in result
