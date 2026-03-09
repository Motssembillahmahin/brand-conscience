"""Tests for circuit breaker."""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock

import pytest

from brand_conscience.common.exceptions import CircuitBreakerTrippedError
from brand_conscience.layer4_deployment.circuit_breaker import CircuitBreaker


def test_normal_spend_does_not_trip():
    breaker = CircuitBreaker(notifier=MagicMock())
    # Spend $100 over 12 hours with $1000 daily budget → safe
    assert breaker.check_spend_velocity(100.0, 1000.0, 12.0)


def test_excessive_spend_trips():
    breaker = CircuitBreaker(notifier=MagicMock())
    # Spend $500 in 1 hour with $100 daily budget → trips
    with pytest.raises(CircuitBreakerTrippedError):
        breaker.check_spend_velocity(500.0, 100.0, 1.0)


def test_campaign_overspend_trips():
    breaker = CircuitBreaker(notifier=MagicMock())
    with pytest.raises(CircuitBreakerTrippedError):
        breaker.check_campaign_spend(160.0, 100.0)


def test_campaign_within_budget():
    breaker = CircuitBreaker(notifier=MagicMock())
    assert breaker.check_campaign_spend(120.0, 100.0)


def test_is_tripped_after_trip():
    breaker = CircuitBreaker(notifier=MagicMock())
    with contextlib.suppress(CircuitBreakerTrippedError):
        breaker.check_spend_velocity(500.0, 100.0, 1.0)
    assert breaker.is_tripped


def test_manual_reset():
    breaker = CircuitBreaker(notifier=MagicMock())
    with contextlib.suppress(CircuitBreakerTrippedError):
        breaker.check_spend_velocity(500.0, 100.0, 1.0)
    breaker.reset()
    assert not breaker.is_tripped
