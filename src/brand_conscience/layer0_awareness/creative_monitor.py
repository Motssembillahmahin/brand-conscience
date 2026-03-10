"""Creative monitor — ad fatigue, competitor shifts (4-hour cadence)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import func

from brand_conscience.common.database import get_session
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.db.tables import Ad, Campaign, Creative, PerformanceMetric
from brand_conscience.layer0_awareness.signals import CreativeSignal

logger = get_logger(__name__)


class CreativeMonitor:
    """Monitor creative performance and detect fatigue.

    Tracks CTR trends, frequency saturation, and competitor
    creative shifts by querying the campaign performance database.
    """

    FATIGUE_CTR_DECLINE_THRESHOLD = 0.30
    FATIGUE_FREQUENCY_THRESHOLD = 5.0
    MIN_IMPRESSIONS_FOR_SIGNAL = 100

    @traced(name="creative_monitor_collect", tags=["layer0", "creative"])
    def collect_signals(self) -> list[CreativeSignal]:
        """Collect creative performance signals.

        Returns:
            List of CreativeSignal instances.
        """
        signals: list[CreativeSignal] = []

        signals.extend(self._check_ctr_trends())
        signals.extend(self._check_frequency_saturation())
        signals.extend(self._check_competitor_shifts())

        logger.info("creative_signals_collected", count=len(signals))
        return signals

    def _check_ctr_trends(self) -> list[CreativeSignal]:
        """Detect CTR decline patterns indicating creative fatigue.

        Compares each campaign's recent 24h CTR against its prior 7-day
        average CTR. A significant decline triggers a fatigue signal.
        """
        signals: list[CreativeSignal] = []
        now = datetime.now(UTC)
        recent_start = now - timedelta(hours=24)
        prior_start = now - timedelta(days=7)

        with get_session() as session:
            live_campaigns = session.query(Campaign).filter(Campaign.status == "live").all()

            for campaign in live_campaigns:
                cid = campaign.id

                # Recent 24h metrics
                recent = (
                    session.query(
                        func.sum(PerformanceMetric.impressions),
                        func.sum(PerformanceMetric.clicks),
                    )
                    .filter(
                        PerformanceMetric.campaign_id == cid,
                        PerformanceMetric.timestamp >= recent_start,
                    )
                    .one()
                )

                # Prior 7-day metrics
                prior = (
                    session.query(
                        func.sum(PerformanceMetric.impressions),
                        func.sum(PerformanceMetric.clicks),
                    )
                    .filter(
                        PerformanceMetric.campaign_id == cid,
                        PerformanceMetric.timestamp >= prior_start,
                        PerformanceMetric.timestamp < recent_start,
                    )
                    .one()
                )

                recent_imps = int(recent[0] or 0)
                recent_clicks = int(recent[1] or 0)
                prior_imps = int(prior[0] or 0)
                prior_clicks = int(prior[1] or 0)

                if recent_imps < self.MIN_IMPRESSIONS_FOR_SIGNAL:
                    continue

                recent_ctr = recent_clicks / recent_imps if recent_imps > 0 else 0.0
                prior_ctr = prior_clicks / prior_imps if prior_imps > 0 else 0.0

                if prior_ctr > 0:
                    ctr_change = (recent_ctr - prior_ctr) / prior_ctr
                    if ctr_change < -0.10:  # >10% CTR decline
                        days_active = (now - campaign.created_at).days
                        signals.append(
                            CreativeSignal(
                                source="ctr_trend_monitor",
                                campaign_id=str(cid),
                                metric_name="ctr_decline",
                                current_value=recent_ctr,
                                trend_direction=ctr_change,
                                fatigue_score=self.compute_fatigue_score(
                                    ctr_change, days_active, 0.0
                                ),
                            )
                        )

        return signals

    def _check_frequency_saturation(self) -> list[CreativeSignal]:
        """Detect campaigns where impressions-per-click ratio is too high.

        A high impressions/click ratio suggests the audience is seeing ads
        too frequently without engaging, indicating saturation.
        """
        signals: list[CreativeSignal] = []
        now = datetime.now(UTC)
        recent_start = now - timedelta(hours=48)

        with get_session() as session:
            live_campaigns = session.query(Campaign).filter(Campaign.status == "live").all()

            for campaign in live_campaigns:
                metrics = (
                    session.query(
                        func.sum(PerformanceMetric.impressions),
                        func.sum(PerformanceMetric.clicks),
                    )
                    .filter(
                        PerformanceMetric.campaign_id == campaign.id,
                        PerformanceMetric.timestamp >= recent_start,
                    )
                    .one()
                )

                impressions = int(metrics[0] or 0)
                clicks = int(metrics[1] or 0)

                if impressions < self.MIN_IMPRESSIONS_FOR_SIGNAL or clicks == 0:
                    continue

                # Proxy for frequency: impressions per click
                imps_per_click = impressions / clicks
                if imps_per_click > self.FATIGUE_FREQUENCY_THRESHOLD * 10:
                    days_active = (now - campaign.created_at).days
                    signals.append(
                        CreativeSignal(
                            source="frequency_monitor",
                            campaign_id=str(campaign.id),
                            metric_name="frequency_saturation",
                            current_value=imps_per_click,
                            trend_direction=-1.0,
                            fatigue_score=self.compute_fatigue_score(
                                -0.2, days_active, imps_per_click / 10
                            ),
                        )
                    )

        return signals

    def _check_competitor_shifts(self) -> list[CreativeSignal]:
        """Detect when our creative diversity is too low.

        Uses the creative portfolio as a proxy for competitive pressure —
        if all active creatives are similar (low diversity), the portfolio
        may be vulnerable to creative fatigue.
        """
        signals: list[CreativeSignal] = []

        with get_session() as session:
            # Count active creatives per campaign
            live_campaigns = session.query(Campaign).filter(Campaign.status == "live").all()

            for campaign in live_campaigns:
                creative_count = (
                    session.query(func.count(Creative.id))
                    .join(Ad, Ad.creative_id == Creative.id)
                    .join(
                        Campaign,
                        Campaign.id == Ad.ad_set_id,  # through ad_set
                    )
                    .filter(
                        Ad.ad_set_id.in_(
                            session.query(Campaign.id).filter(Campaign.id == campaign.id)
                        )
                    )
                    .scalar()
                )
                creative_count = int(creative_count or 0)

                if creative_count <= 1:
                    signals.append(
                        CreativeSignal(
                            source="diversity_monitor",
                            campaign_id=str(campaign.id),
                            metric_name="low_creative_diversity",
                            current_value=float(creative_count),
                            trend_direction=-1.0,
                            fatigue_score=0.6,
                        )
                    )

        return signals

    def compute_fatigue_score(self, ctr_change: float, days_active: int, frequency: float) -> float:
        """Compute fatigue score for a creative.

        Returns:
            Score 0-1, higher = more fatigued.
        """
        fatigue = 0.0

        if ctr_change < 0:
            fatigue += min(abs(ctr_change) / self.FATIGUE_CTR_DECLINE_THRESHOLD, 1.0) * 0.5

        if days_active > 7:
            fatigue += min((days_active - 7) / 14, 1.0) * 0.25

        if frequency > self.FATIGUE_FREQUENCY_THRESHOLD:
            fatigue += min((frequency - self.FATIGUE_FREQUENCY_THRESHOLD) / 5.0, 1.0) * 0.25

        return min(fatigue, 1.0)
