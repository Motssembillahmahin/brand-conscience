"""Tests for database tables and session management."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from brand_conscience.common.types import CampaignStatus
from brand_conscience.db.tables import Campaign, MomentSnapshot


def test_create_campaign(db_session):
    campaign = Campaign(
        id=uuid.uuid4(),
        name="Test Campaign",
        status=CampaignStatus.DRAFT,
        objective="conversions",
        daily_budget=500.0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db_session.add(campaign)
    db_session.commit()

    result = db_session.get(Campaign, campaign.id)
    assert result is not None
    assert result.name == "Test Campaign"
    assert result.status == CampaignStatus.DRAFT
    assert result.daily_budget == 500.0


def test_create_moment_snapshot(db_session):
    snapshot = MomentSnapshot(
        id=uuid.uuid4(),
        timestamp=datetime.now(UTC),
        urgency_score=0.75,
        recommended_action="launch",
        business_signals=[{"type": "revenue_drop"}],
        cultural_signals=[],
        creative_signals=[],
        affected_categories=["electronics"],
        affected_audiences=["retargeting"],
        context_summary="Test snapshot",
    )
    db_session.add(snapshot)
    db_session.commit()

    result = db_session.get(MomentSnapshot, snapshot.id)
    assert result is not None
    assert result.urgency_score == 0.75
    assert result.recommended_action == "launch"
