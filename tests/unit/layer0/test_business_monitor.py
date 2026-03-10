"""Tests for business monitor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from brand_conscience.layer0_awareness.business_monitor import BusinessMonitor
from brand_conscience.layer0_awareness.signals import BusinessSignal


def test_collect_signals_returns_list():
    monitor = BusinessMonitor()
    with (
        patch.object(monitor, "_check_revenue", return_value=[]),
        patch.object(monitor, "_check_inventory", return_value=[]),
        patch.object(monitor, "_check_crm_events", return_value=[]),
    ):
        signals = monitor.collect_signals()
    assert isinstance(signals, list)


def test_collect_signals_computes_severity():
    signal = BusinessSignal(source="test", change_pct=-0.35)
    monitor = BusinessMonitor()
    with (
        patch.object(monitor, "_check_revenue", return_value=[signal]),
        patch.object(monitor, "_check_inventory", return_value=[]),
        patch.object(monitor, "_check_crm_events", return_value=[]),
    ):
        signals = monitor.collect_signals()
    assert len(signals) == 1
    assert signals[0].severity == 1.0


def test_check_revenue_detects_decline():
    """Revenue decline signal when recent revenue is below trailing average."""
    monitor = BusinessMonitor()

    mock_session = MagicMock()
    # recent_revenue = 50, trailing_revenue = 700 (avg 100), change = -50%
    mock_session.query.return_value.filter.return_value.scalar.side_effect = [50.0, 700.0, 10.0]

    with patch("brand_conscience.layer0_awareness.business_monitor.get_session") as mock_gs:
        mock_gs.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_gs.return_value.__exit__ = MagicMock(return_value=False)
        signals = monitor._check_revenue()

    assert len(signals) >= 1
    assert signals[0].category == "revenue_decline"


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
