"""Typed database query functions."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import desc, func

from brand_conscience.common.database import get_session
from brand_conscience.common.types import CampaignStatus
from brand_conscience.db.tables import (
    ABTestGroup,
    AuditLog,
    Campaign,
    Creative,
    ModelCheckpoint,
    MomentSnapshot,
    PerformanceMetric,
)

# Campaigns


def get_campaign(campaign_id: str) -> Campaign | None:
    with get_session() as session:
        return session.get(Campaign, uuid.UUID(campaign_id))


def get_campaigns_by_status(status: CampaignStatus) -> list[Campaign]:
    with get_session() as session:
        return (
            session.query(Campaign)
            .filter_by(status=status)
            .order_by(desc(Campaign.created_at))
            .all()
        )


def get_live_campaigns() -> list[Campaign]:
    return get_campaigns_by_status(CampaignStatus.LIVE)


def get_total_daily_spend() -> float:
    today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    with get_session() as session:
        result = (
            session.query(func.sum(PerformanceMetric.spend))
            .filter(PerformanceMetric.timestamp >= today_start)
            .scalar()
        )
        return float(result or 0.0)


# Moment snapshots


def get_latest_moment_snapshot() -> MomentSnapshot | None:
    with get_session() as session:
        return session.query(MomentSnapshot).order_by(desc(MomentSnapshot.timestamp)).first()


def get_moment_snapshots_since(hours: int) -> list[MomentSnapshot]:
    since = datetime.now(UTC) - timedelta(hours=hours)
    with get_session() as session:
        return (
            session.query(MomentSnapshot)
            .filter(MomentSnapshot.timestamp >= since)
            .order_by(desc(MomentSnapshot.timestamp))
            .all()
        )


# Performance metrics


def get_campaign_metrics(
    campaign_id: str,
    hours: int | None = None,
) -> list[PerformanceMetric]:
    with get_session() as session:
        query = session.query(PerformanceMetric).filter_by(campaign_id=uuid.UUID(campaign_id))
        if hours:
            since = datetime.now(UTC) - timedelta(hours=hours)
            query = query.filter(PerformanceMetric.timestamp >= since)
        return query.order_by(desc(PerformanceMetric.timestamp)).all()


def get_aggregate_metrics(campaign_id: str) -> dict:
    with get_session() as session:
        result = (
            session.query(
                func.sum(PerformanceMetric.impressions),
                func.sum(PerformanceMetric.clicks),
                func.sum(PerformanceMetric.spend),
                func.sum(PerformanceMetric.conversions),
                func.sum(PerformanceMetric.revenue),
            )
            .filter_by(campaign_id=uuid.UUID(campaign_id))
            .one()
        )
        impressions = int(result[0] or 0)
        clicks = int(result[1] or 0)
        spend = float(result[2] or 0.0)
        conversions = int(result[3] or 0)
        revenue = float(result[4] or 0.0)

        return {
            "impressions": impressions,
            "clicks": clicks,
            "spend": spend,
            "conversions": conversions,
            "revenue": revenue,
            "ctr": clicks / impressions if impressions > 0 else 0.0,
            "cpc": spend / clicks if clicks > 0 else 0.0,
            "roas": revenue / spend if spend > 0 else 0.0,
        }


# Creatives


def get_creative(creative_id: str) -> Creative | None:
    with get_session() as session:
        return session.get(Creative, uuid.UUID(creative_id))


def get_recent_creatives(limit: int = 50) -> list[Creative]:
    with get_session() as session:
        return session.query(Creative).order_by(desc(Creative.created_at)).limit(limit).all()


# A/B Testing


def get_ab_test_groups(campaign_id: str) -> list[ABTestGroup]:
    with get_session() as session:
        return session.query(ABTestGroup).filter_by(campaign_id=uuid.UUID(campaign_id)).all()


# Model checkpoints


def get_active_checkpoint(model_name: str) -> ModelCheckpoint | None:
    with get_session() as session:
        return (
            session.query(ModelCheckpoint).filter_by(model_name=model_name, is_active=True).first()
        )


def get_checkpoint_history(model_name: str) -> list[ModelCheckpoint]:
    with get_session() as session:
        return (
            session.query(ModelCheckpoint)
            .filter_by(model_name=model_name)
            .order_by(desc(ModelCheckpoint.version))
            .all()
        )


# Audit log


def create_audit_entry(
    layer: str,
    action: str,
    entity_type: str,
    entity_id: str,
    details: dict | None = None,
    trace_id: str | None = None,
) -> str:
    entry_id = str(uuid.uuid4())
    with get_session() as session:
        entry = AuditLog(
            id=uuid.UUID(entry_id),
            timestamp=datetime.now(UTC),
            layer=layer,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details,
            trace_id=trace_id,
        )
        session.add(entry)
    return entry_id
