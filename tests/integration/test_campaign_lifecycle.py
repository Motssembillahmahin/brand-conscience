"""Integration test: Campaign lifecycle state machine."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from brand_conscience.common.types import CampaignStatus
from brand_conscience.db.tables import Campaign


@pytest.mark.integration
def test_full_lifecycle(db_session):
    """Test full campaign lifecycle: DRAFT → PENDING_APPROVAL → LIVE → PAUSED → COMPLETED."""
    campaign = Campaign(
        id=uuid.uuid4(),
        name="Lifecycle Test",
        status=CampaignStatus.DRAFT,
        objective="conversions",
        daily_budget=100.0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db_session.add(campaign)
    db_session.commit()

    # DRAFT → PENDING_APPROVAL
    campaign.status = CampaignStatus.PENDING_APPROVAL
    db_session.commit()
    assert campaign.status == CampaignStatus.PENDING_APPROVAL

    # PENDING_APPROVAL → LIVE
    campaign.status = CampaignStatus.LIVE
    db_session.commit()
    assert campaign.status == CampaignStatus.LIVE

    # LIVE → PAUSED
    campaign.status = CampaignStatus.PAUSED
    db_session.commit()
    assert campaign.status == CampaignStatus.PAUSED

    # PAUSED → COMPLETED
    campaign.status = CampaignStatus.COMPLETED
    db_session.commit()
    assert campaign.status == CampaignStatus.COMPLETED
