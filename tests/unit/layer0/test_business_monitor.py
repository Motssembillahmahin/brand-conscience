"""Tests for business monitor."""

from __future__ import annotations

from brand_conscience.layer0_awareness.business_monitor import BusinessMonitor
from brand_conscience.layer0_awareness.signals import BusinessSignal


def test_collect_signals_returns_list():
    monitor = BusinessMonitor()
    signals = monitor.collect_signals()
    assert isinstance(signals, list)


def test_compute_severity_high():
    monitor = BusinessMonitor()
    signal = BusinessSignal(source="test", change_pct=-0.35)
    severity = monitor.compute_severity(signal)
    assert severity == 1.0


def test_compute_severity_medium():
    monitor = BusinessMonitor()
    signal = BusinessSignal(source="test", change_pct=-0.20)
    severity = monitor.compute_severity(signal)
    assert severity == 0.7


def test_compute_severity_low():
    monitor = BusinessMonitor()
    signal = BusinessSignal(source="test", change_pct=-0.08)
    severity = monitor.compute_severity(signal)
    assert severity == 0.4


def test_compute_severity_minimal():
    monitor = BusinessMonitor()
    signal = BusinessSignal(source="test", change_pct=0.02)
    severity = monitor.compute_severity(signal)
    assert severity == 0.1
