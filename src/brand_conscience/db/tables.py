"""SQLAlchemy ORM table definitions."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from brand_conscience.common.database import Base
from brand_conscience.common.types import CampaignStatus, QualityTier


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _utc_now() -> datetime:
    from datetime import UTC

    return datetime.now(UTC)


class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[CampaignStatus] = mapped_column(
        Enum(CampaignStatus), default=CampaignStatus.DRAFT, nullable=False
    )
    objective: Mapped[str] = mapped_column(String(100), nullable=False)
    daily_budget: Mapped[float] = mapped_column(Float, nullable=False)
    total_spend: Mapped[float] = mapped_column(Float, default=0.0)
    meta_campaign_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    moment_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("moment_snapshots.id"), nullable=True
    )
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now
    )

    ad_sets: Mapped[list[AdSet]] = relationship(back_populates="campaign", cascade="all, delete")
    performance_metrics: Mapped[list[PerformanceMetric]] = relationship(
        back_populates="campaign", cascade="all, delete"
    )

    __table_args__ = (Index("ix_campaigns_status", "status"),)


class AdSet(Base):
    __tablename__ = "ad_sets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    campaign_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    audience_segment: Mapped[str] = mapped_column(String(100), nullable=False)
    daily_budget: Mapped[float] = mapped_column(Float, nullable=False)
    bid_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    meta_adset_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    targeting: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    campaign: Mapped[Campaign] = relationship(back_populates="ad_sets")
    ads: Mapped[list[Ad]] = relationship(back_populates="ad_set", cascade="all, delete")


class Ad(Base):
    __tablename__ = "ads"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    ad_set_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ad_sets.id"), nullable=False
    )
    creative_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("creatives.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    meta_ad_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    ab_test_group: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    ad_set: Mapped[AdSet] = relationship(back_populates="ads")
    creative: Mapped[Creative] = relationship(back_populates="ads")


class Creative(Base):
    __tablename__ = "creatives"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_score: Mapped[float] = mapped_column(Float, nullable=False)
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    quality_tier: Mapped[QualityTier] = mapped_column(Enum(QualityTier), nullable=False)
    brand_alignment_score: Mapped[float] = mapped_column(Float, nullable=False)
    originality_score: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_ctr: Mapped[float | None] = mapped_column(Float, nullable=True)
    clip_embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    gate_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    ads: Mapped[list[Ad]] = relationship(back_populates="creative")


class MomentSnapshot(Base):
    __tablename__ = "moment_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    urgency_score: Mapped[float] = mapped_column(Float, nullable=False)
    recommended_action: Mapped[str] = mapped_column(String(50), nullable=False)
    business_signals: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cultural_signals: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    creative_signals: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    affected_categories: Mapped[list | None] = mapped_column(JSON, nullable=True)
    affected_audiences: Mapped[list | None] = mapped_column(JSON, nullable=True)
    context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_moment_snapshots_timestamp", "timestamp"),)


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    campaign_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    impressions: Mapped[int] = mapped_column(Integer, default=0)
    clicks: Mapped[int] = mapped_column(Integer, default=0)
    spend: Mapped[float] = mapped_column(Float, default=0.0)
    conversions: Mapped[int] = mapped_column(Integer, default=0)
    revenue: Mapped[float] = mapped_column(Float, default=0.0)
    ctr: Mapped[float | None] = mapped_column(Float, nullable=True)
    cpc: Mapped[float | None] = mapped_column(Float, nullable=True)
    roas: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    campaign: Mapped[Campaign] = relationship(back_populates="performance_metrics")

    __table_args__ = (Index("ix_performance_metrics_campaign_time", "campaign_id", "timestamp"),)


class ABTestGroup(Base):
    __tablename__ = "ab_test_groups"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    campaign_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False
    )
    group_name: Mapped[str] = mapped_column(String(100), nullable=False)
    variant_id: Mapped[str] = mapped_column(String(100), nullable=False)
    alpha: Mapped[float] = mapped_column(Float, default=1.0)
    beta: Mapped[float] = mapped_column(Float, default=1.0)
    impressions: Mapped[int] = mapped_column(Integer, default=0)
    conversions: Mapped[int] = mapped_column(Integer, default=0)
    is_holdout: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class ModelCheckpoint(Base):
    __tablename__ = "model_checkpoints"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(String(500), nullable=False)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    __table_args__ = (Index("ix_model_checkpoints_active", "model_name", "is_active"),)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    layer: Mapped[str] = mapped_column(String(50), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    __table_args__ = (
        Index("ix_audit_logs_timestamp", "timestamp"),
        Index("ix_audit_logs_entity", "entity_type", "entity_id"),
    )
