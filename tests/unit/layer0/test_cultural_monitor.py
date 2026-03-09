"""Tests for cultural monitor."""

from __future__ import annotations

from brand_conscience.layer0_awareness.cultural_monitor import CulturalMonitor


def test_collect_signals_returns_list():
    monitor = CulturalMonitor()
    signals = monitor.collect_signals()
    assert isinstance(signals, list)
