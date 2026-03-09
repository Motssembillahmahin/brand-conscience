"""Initial schema.

Revision ID: 001
Revises: None
Create Date: 2026-03-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID

from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Moment snapshots
    op.create_table(
        "moment_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("urgency_score", sa.Float, nullable=False),
        sa.Column("recommended_action", sa.String(50), nullable=False),
        sa.Column("business_signals", JSON, nullable=True),
        sa.Column("cultural_signals", JSON, nullable=True),
        sa.Column("creative_signals", JSON, nullable=True),
        sa.Column("affected_categories", JSON, nullable=True),
        sa.Column("affected_audiences", JSON, nullable=True),
        sa.Column("context_summary", sa.Text, nullable=True),
    )
    op.create_index("ix_moment_snapshots_timestamp", "moment_snapshots", ["timestamp"])

    # Campaigns
    op.create_table(
        "campaigns",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="draft"),
        sa.Column("objective", sa.String(100), nullable=False),
        sa.Column("daily_budget", sa.Float, nullable=False),
        sa.Column("total_spend", sa.Float, server_default="0"),
        sa.Column("meta_campaign_id", sa.String(100), nullable=True),
        sa.Column(
            "moment_profile_id",
            UUID(as_uuid=True),
            sa.ForeignKey("moment_snapshots.id"),
            nullable=True,
        ),
        sa.Column("config", JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_campaigns_status", "campaigns", ["status"])

    # Ad sets
    op.create_table(
        "ad_sets",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "campaign_id",
            UUID(as_uuid=True),
            sa.ForeignKey("campaigns.id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("audience_segment", sa.String(100), nullable=False),
        sa.Column("daily_budget", sa.Float, nullable=False),
        sa.Column("bid_amount", sa.Float, nullable=True),
        sa.Column("meta_adset_id", sa.String(100), nullable=True),
        sa.Column("targeting", JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Creatives
    op.create_table(
        "creatives",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("prompt_text", sa.Text, nullable=False),
        sa.Column("prompt_score", sa.Float, nullable=False),
        sa.Column("image_path", sa.String(500), nullable=False),
        sa.Column("quality_tier", sa.String(50), nullable=False),
        sa.Column("brand_alignment_score", sa.Float, nullable=False),
        sa.Column("originality_score", sa.Float, nullable=False),
        sa.Column("predicted_ctr", sa.Float, nullable=True),
        sa.Column("clip_embedding", JSON, nullable=True),
        sa.Column("gate_results", JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Ads
    op.create_table(
        "ads",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "ad_set_id",
            UUID(as_uuid=True),
            sa.ForeignKey("ad_sets.id"),
            nullable=False,
        ),
        sa.Column(
            "creative_id",
            UUID(as_uuid=True),
            sa.ForeignKey("creatives.id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("meta_ad_id", sa.String(100), nullable=True),
        sa.Column("ab_test_group", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Performance metrics
    op.create_table(
        "performance_metrics",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "campaign_id",
            UUID(as_uuid=True),
            sa.ForeignKey("campaigns.id"),
            nullable=False,
        ),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("impressions", sa.Integer, server_default="0"),
        sa.Column("clicks", sa.Integer, server_default="0"),
        sa.Column("spend", sa.Float, server_default="0"),
        sa.Column("conversions", sa.Integer, server_default="0"),
        sa.Column("revenue", sa.Float, server_default="0"),
        sa.Column("ctr", sa.Float, nullable=True),
        sa.Column("cpc", sa.Float, nullable=True),
        sa.Column("roas", sa.Float, nullable=True),
        sa.Column("raw_data", JSON, nullable=True),
    )
    op.create_index(
        "ix_performance_metrics_campaign_time",
        "performance_metrics",
        ["campaign_id", "timestamp"],
    )

    # A/B test groups
    op.create_table(
        "ab_test_groups",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "campaign_id",
            UUID(as_uuid=True),
            sa.ForeignKey("campaigns.id"),
            nullable=False,
        ),
        sa.Column("group_name", sa.String(100), nullable=False),
        sa.Column("variant_id", sa.String(100), nullable=False),
        sa.Column("alpha", sa.Float, server_default="1.0"),
        sa.Column("beta", sa.Float, server_default="1.0"),
        sa.Column("impressions", sa.Integer, server_default="0"),
        sa.Column("conversions", sa.Integer, server_default="0"),
        sa.Column("is_holdout", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Model checkpoints
    op.create_table(
        "model_checkpoints",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("checkpoint_path", sa.String(500), nullable=False),
        sa.Column("metrics", JSON, nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_model_checkpoints_active",
        "model_checkpoints",
        ["model_name", "is_active"],
    )

    # Audit logs
    op.create_table(
        "audit_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("layer", sa.String(50), nullable=False),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("entity_id", sa.String(100), nullable=False),
        sa.Column("details", JSON, nullable=True),
        sa.Column("trace_id", sa.String(100), nullable=True),
    )
    op.create_index("ix_audit_logs_timestamp", "audit_logs", ["timestamp"])
    op.create_index("ix_audit_logs_entity", "audit_logs", ["entity_type", "entity_id"])


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("model_checkpoints")
    op.drop_table("ab_test_groups")
    op.drop_table("performance_metrics")
    op.drop_table("ads")
    op.drop_table("creatives")
    op.drop_table("ad_sets")
    op.drop_table("campaigns")
    op.drop_table("moment_snapshots")
