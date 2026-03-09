"""Celery periodic tasks for Layer 0 monitoring."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from brand_conscience.common.database import get_session
from brand_conscience.common.logging import bind_context, get_logger
from brand_conscience.common.tracing import get_trace_headers, traced
from brand_conscience.db.tables import MomentSnapshot
from brand_conscience.layer0_awareness.business_monitor import BusinessMonitor
from brand_conscience.layer0_awareness.creative_monitor import CreativeMonitor
from brand_conscience.layer0_awareness.cultural_monitor import CulturalMonitor
from brand_conscience.layer0_awareness.moment_profile import MomentProfileBuilder

logger = get_logger(__name__)


@traced(name="run_monitoring_cycle", tags=["layer0", "task"])
def run_monitoring_cycle() -> dict:
    """Execute a full monitoring cycle and persist the moment profile.

    Called by Celery beat on schedule (business: 15m, cultural: 1h, creative: 4h)
    or manually via CLI.
    """
    bind_context(layer="layer0_awareness")

    business = BusinessMonitor().collect_signals()
    cultural = CulturalMonitor().collect_signals()
    creative = CreativeMonitor().collect_signals()

    builder = MomentProfileBuilder()
    profile = builder.build(business, cultural, creative)

    # Persist to database
    snapshot = MomentSnapshot(
        id=uuid.UUID(profile.id),
        timestamp=profile.timestamp,
        urgency_score=profile.urgency_score,
        recommended_action=profile.recommended_action.value,
        business_signals=profile.to_dict()["business_signals"],
        cultural_signals=profile.to_dict()["cultural_signals"],
        creative_signals=profile.to_dict()["creative_signals"],
        affected_categories=profile.affected_categories,
        affected_audiences=profile.affected_audiences,
        context_summary=profile.context_summary,
    )

    with get_session() as session:
        session.add(snapshot)

    logger.info(
        "monitoring_cycle_complete",
        profile_id=profile.id,
        urgency=profile.urgency_score,
        action=profile.recommended_action.value,
    )

    return profile.to_dict()
