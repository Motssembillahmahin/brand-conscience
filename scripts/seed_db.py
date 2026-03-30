"""Seed the database with realistic data for development and Layer 0 testing.

Usage:
    uv run python scripts/seed_db.py
"""

from __future__ import annotations

import random
import uuid
from datetime import UTC, datetime, timedelta

import click

from brand_conscience.common.config import load_settings
from brand_conscience.common.database import get_session, init_database
from brand_conscience.common.logging import configure_logging, get_logger
from brand_conscience.common.types import ActionType, CampaignStatus
from brand_conscience.db.tables import Campaign, MomentSnapshot, PerformanceMetric

logger = get_logger(__name__)

# Reproducible random data
random.seed(42)


def _generate_performance_metrics(
    campaign_id: uuid.UUID,
    daily_budget: float,
    days: int = 8,
) -> list[PerformanceMetric]:
    """Generate hourly PerformanceMetric rows over `days` days.

    Pattern:
    - Days 1-6: steady revenue (~$80-120/day), healthy CTR, normal conversions
    - Day 7 (yesterday): revenue drops to ~$40 (triggers revenue_decline)
    - Day 8 (today): low spend so far (triggers budget_underspend if past noon)
    - Conversion rate dips on last day (triggers conversion_decline)
    """
    now = datetime.now(UTC)
    metrics: list[PerformanceMetric] = []

    for day_offset in range(days, 0, -1):
        day_start = (now - timedelta(days=day_offset)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # How many hours to generate for this day
        hours = max(now.hour, 1) if day_offset == 0 else 24

        for hour in range(hours):
            ts = day_start + timedelta(hours=hour)

            # Base metrics per hour (steady state)
            base_impressions = random.randint(400, 600)
            base_ctr = random.uniform(0.02, 0.04)
            base_clicks = int(base_impressions * base_ctr)
            base_spend = daily_budget / 24 * random.uniform(0.8, 1.2)
            base_conversion_rate = random.uniform(0.05, 0.10)
            base_conversions = max(1, int(base_clicks * base_conversion_rate))
            base_revenue = base_conversions * random.uniform(15.0, 25.0)

            # Apply day-specific patterns
            if day_offset == 1:
                # Yesterday: revenue crash (~40% of normal)
                base_revenue *= 0.4
                base_conversions = max(0, int(base_conversions * 0.4))
                base_spend *= 1.1  # spending more for less (bad ROAS)
            elif day_offset == 0:
                # Today: very low spend (underspend scenario)
                base_spend *= 0.2
                base_impressions = int(base_impressions * 0.3)
                base_clicks = max(0, int(base_clicks * 0.3))
                # Conversion rate tanks
                base_conversions = max(0, int(base_clicks * 0.02))
                base_revenue = base_conversions * random.uniform(10.0, 15.0)

            ctr = base_clicks / base_impressions if base_impressions > 0 else 0.0
            cpc = base_spend / base_clicks if base_clicks > 0 else 0.0
            roas = base_revenue / base_spend if base_spend > 0 else 0.0

            metrics.append(
                PerformanceMetric(
                    id=uuid.uuid4(),
                    campaign_id=campaign_id,
                    timestamp=ts,
                    impressions=base_impressions,
                    clicks=base_clicks,
                    spend=round(base_spend, 2),
                    conversions=base_conversions,
                    revenue=round(base_revenue, 2),
                    ctr=round(ctr, 4),
                    cpc=round(cpc, 2),
                    roas=round(roas, 2),
                )
            )

    # Today's partial hours (up to now)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for hour in range(max(now.hour, 1)):
        ts = today_start + timedelta(hours=hour)
        impressions = random.randint(100, 200)
        clicks = max(0, int(impressions * random.uniform(0.01, 0.02)))
        spend = round(daily_budget / 24 * 0.2, 2)
        conversions = max(0, int(clicks * 0.02))
        revenue = round(conversions * random.uniform(10.0, 15.0), 2)

        metrics.append(
            PerformanceMetric(
                id=uuid.uuid4(),
                campaign_id=campaign_id,
                timestamp=ts,
                impressions=impressions,
                clicks=clicks,
                spend=spend,
                conversions=conversions,
                revenue=revenue,
                ctr=round(clicks / impressions if impressions else 0, 4),
                cpc=round(spend / clicks if clicks else 0, 2),
                roas=round(revenue / spend if spend else 0, 2),
            )
        )

    return metrics


@click.command()
def seed() -> None:
    """Seed the database with sample data."""
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_format)
    init_database(settings)

    with get_session() as session:
        # Create moment snapshot
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
        session.flush()

        # Campaign 1: live campaign with declining revenue (triggers Layer 0 signals)
        campaign_1 = Campaign(
            id=uuid.uuid4(),
            name="Summer Sale — Brand Awareness",
            status=CampaignStatus.LIVE,
            objective="awareness",
            daily_budget=200.0,
            moment_profile_id=moment.id,
            created_at=datetime.now(UTC) - timedelta(days=8),
            updated_at=datetime.now(UTC),
        )
        session.add(campaign_1)
        session.flush()

        metrics_1 = _generate_performance_metrics(campaign_1.id, campaign_1.daily_budget)
        session.add_all(metrics_1)

        # Campaign 2: live campaign with budget overspend pattern
        campaign_2 = Campaign(
            id=uuid.uuid4(),
            name="Product Launch — Conversions",
            status=CampaignStatus.LIVE,
            objective="conversions",
            daily_budget=150.0,
            moment_profile_id=moment.id,
            created_at=datetime.now(UTC) - timedelta(days=5),
            updated_at=datetime.now(UTC),
        )
        session.add(campaign_2)
        session.flush()

        metrics_2 = _generate_performance_metrics(campaign_2.id, campaign_2.daily_budget, days=5)
        # Spike today's spend to trigger overspend
        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        for m in metrics_2:
            if m.timestamp >= today_start:
                m.spend = round(campaign_2.daily_budget / 8, 2)  # burns budget fast
        session.add_all(metrics_2)

        # Campaign 3: draft campaign (should be ignored by inventory check)
        campaign_3 = Campaign(
            id=uuid.uuid4(),
            name="Holiday Preview — Draft",
            status=CampaignStatus.DRAFT,
            objective="awareness",
            daily_budget=100.0,
            moment_profile_id=moment.id,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(campaign_3)

        total_metrics = len(metrics_1) + len(metrics_2)
        logger.info(
            "database_seeded",
            campaigns=3,
            performance_metrics=total_metrics,
            moment_id=str(moment.id),
        )
        click.echo(f"Seeded: 3 campaigns, {total_metrics} performance metrics, 1 moment snapshot")


if __name__ == "__main__":
    seed()
