"""Meta metrics collection for feedback loop."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from brand_conscience.common.database import get_session
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.db.tables import PerformanceMetric
from brand_conscience.layer4_deployment.meta_client import MetaClient

logger = get_logger(__name__)


class MetricsCollector:
    """Collect campaign performance metrics from Meta Marketing API."""

    def __init__(self, meta_client: MetaClient | None = None) -> None:
        self._meta = meta_client or MetaClient()

    @traced(name="collect_campaign_metrics", tags=["layer5", "collector"])
    def collect(self, campaign_id: str, meta_campaign_id: str) -> dict:
        """Fetch latest metrics for a campaign from Meta API.

        Args:
            campaign_id: Internal campaign UUID.
            meta_campaign_id: Meta platform campaign ID.

        Returns:
            Dict with collected metrics.
        """
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
        """Fetch metrics from Meta Marketing API via MetaClient.

        Calls the Meta Insights API for the campaign and normalizes
        the response into our standard metrics dict.
        """
        if not meta_campaign_id:
            logger.warning("meta_campaign_id_empty", msg="Skipping Meta API fetch")
            return {
                "impressions": 0,
                "clicks": 0,
                "spend": 0.0,
                "conversions": 0,
                "revenue": 0.0,
            }

        try:
            insights = self._meta.get_campaign_insights(
                meta_campaign_id,
                fields=["impressions", "clicks", "spend", "conversions", "actions"],
            )

            # Parse conversion value from actions array
            revenue = 0.0
            conversions = 0
            for action in insights.get("actions", []):
                if action.get("action_type") == "offsite_conversion.fb_pixel_purchase":
                    conversions = int(action.get("value", 0))
                if action.get("action_type") == "offsite_conversion.fb_pixel_purchase":
                    revenue = float(action.get("value", 0.0))

            return {
                "impressions": int(insights.get("impressions", 0)),
                "clicks": int(insights.get("clicks", 0)),
                "spend": float(insights.get("spend", 0.0)),
                "conversions": conversions,
                "revenue": revenue,
            }
        except Exception as exc:
            logger.error(
                "meta_insights_fetch_failed",
                meta_campaign_id=meta_campaign_id,
                error=str(exc),
            )
            return {
                "impressions": 0,
                "clicks": 0,
                "spend": 0.0,
                "conversions": 0,
                "revenue": 0.0,
            }
