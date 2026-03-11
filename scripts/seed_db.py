"""Seed the database with initial data for development.

Usage:
    uv run python scripts/seed_db.py
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import click

from brand_conscience.common.config import load_settings
from brand_conscience.common.database import get_session, init_database
from brand_conscience.common.logging import configure_logging, get_logger
from brand_conscience.common.types import ActionType, CampaignStatus
from brand_conscience.db.tables import Campaign, MomentSnapshot

logger = get_logger(__name__)


@click.command()
def seed() -> None:
    """Seed the database with sample data."""
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_format)
    init_database(settings)

    with get_session() as session:
        # Create sample moment snapshot
        moment = MomentSnapshot(
            id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            urgency_score=0.65,
            recommended_action=ActionType.HOLD.value,
            business_signals=[],
            cultural_signals=[],
            creative_signals=[],
            affected_categories=["general"],
            affected_audiences=["broad_interest"],
            context_summary="Initial seed — no active signals.",
        )
        session.add(moment)
        session.flush()  # ensure moment row exists before campaign FK reference

        # Create sample campaign
        campaign = Campaign(
            id=uuid.uuid4(),
            name="Seed Campaign",
            status=CampaignStatus.DRAFT,
            objective="awareness",
            daily_budget=100.0,
            moment_profile_id=moment.id,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(campaign)

        logger.info("database_seeded", moment_id=str(moment.id), campaign_id=str(campaign.id))


if __name__ == "__main__":
    seed()
