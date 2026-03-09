"""Meta metrics collection for feedback loop."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from brand_conscience.common.database import get_session
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.db.tables import PerformanceMetric

logger = get_logger(__name__)


class MetricsCollector:
    """Collect campaign performance metrics from Meta Marketing API."""

    @traced(name="collect_campaign_metrics", tags=["layer5", "collector"])
    def collect(self, campaign_id: str, meta_campaign_id: str) -> dict:
        """Fetch latest metrics for a campaign from Meta API.

        Args:
            campaign_id: Internal campaign UUID.
            meta_campaign_id: Meta platform campaign ID.

        Returns:
            Dict with collected metrics.
        """
        # TODO: integrate with Meta Marketing API
        metrics = self._fetch_from_meta(meta_campaign_id)

        # Compute derived metrics
        if metrics["impressions"] > 0:
            metrics["ctr"] = metrics["clicks"] / metrics["impressions"]
        if metrics["clicks"] > 0:
            metrics["cpc"] = metrics["spend"] / metrics["clicks"]
        if metrics["spend"] > 0:
            metrics["roas"] = metrics["revenue"] / metrics["spend"]

        # Persist
        with get_session() as session:
            record = PerformanceMetric(
                id=uuid.uuid4(),
                campaign_id=uuid.UUID(campaign_id),
                timestamp=datetime.now(UTC),
                impressions=metrics.get("impressions", 0),
                clicks=metrics.get("clicks", 0),
                spend=metrics.get("spend", 0.0),
                conversions=metrics.get("conversions", 0),
                revenue=metrics.get("revenue", 0.0),
                ctr=metrics.get("ctr"),
                cpc=metrics.get("cpc"),
                roas=metrics.get("roas"),
                raw_data=metrics,
            )
            session.add(record)

        logger.info(
            "metrics_collected",
            campaign_id=campaign_id,
            impressions=metrics.get("impressions", 0),
            spend=metrics.get("spend", 0.0),
        )
        return metrics

    def _fetch_from_meta(self, meta_campaign_id: str) -> dict:
        """Fetch metrics from Meta Marketing API.

        Returns raw metrics dict.
        """
        # TODO: actual Meta API integration
        return {
            "impressions": 0,
            "clicks": 0,
            "spend": 0.0,
            "conversions": 0,
            "revenue": 0.0,
        }
