"""Tests for safety mechanisms."""

from __future__ import annotations

import pytest

from brand_conscience.common.exceptions import (
    BidCapExceededError,
    CircuitBreakerTrippedError,
    InvalidTransitionError,
)
from brand_conscience.common.types import CampaignStatus, validate_campaign_transition


def test_valid_campaign_transitions():
    assert validate_campaign_transition(CampaignStatus.DRAFT, CampaignStatus.PENDING_APPROVAL)
    assert validate_campaign_transition(CampaignStatus.PENDING_APPROVAL, CampaignStatus.LIVE)
    assert validate_campaign_transition(CampaignStatus.LIVE, CampaignStatus.PAUSED)
    assert validate_campaign_transition(CampaignStatus.PAUSED, CampaignStatus.LIVE)
    assert validate_campaign_transition(CampaignStatus.LIVE, CampaignStatus.COMPLETED)


def test_invalid_campaign_transitions():
    assert not validate_campaign_transition(CampaignStatus.DRAFT, CampaignStatus.LIVE)
    assert not validate_campaign_transition(CampaignStatus.COMPLETED, CampaignStatus.LIVE)
    assert not validate_campaign_transition(CampaignStatus.DRAFT, CampaignStatus.COMPLETED)


def test_invalid_transition_error():
    exc = InvalidTransitionError("draft", "live")
    assert "draft" in str(exc)
    assert "live" in str(exc)
    assert exc.current == "draft"
    assert exc.target == "live"


def test_circuit_breaker_error():
    exc = CircuitBreakerTrippedError(
        reason="Spend velocity exceeded",
        campaigns_paused=["c1", "c2"],
    )
    assert "Spend velocity" in str(exc)
    assert exc.campaigns_paused == ["c1", "c2"]


def test_bid_cap_error():
    with pytest.raises(BidCapExceededError):
        raise BidCapExceededError("Bid $50.00 exceeds hard cap $25.00")
