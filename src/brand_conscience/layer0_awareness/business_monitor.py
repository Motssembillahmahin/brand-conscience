"""Business monitor — revenue, inventory, CRM signals (15-minute cadence)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import func

from brand_conscience.common.database import get_session
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.db.tables import Campaign, PerformanceMetric
from brand_conscience.layer0_awareness.signals import BusinessSignal

logger = get_logger(__name__)


class BusinessMonitor:
    """Collect business health signals.

    Monitors revenue trends, inventory levels, and CRM events
    to detect advertising opportunities and threats.
    """

    REVENUE_DECLINE_THRESHOLD = -0.10
    REVENUE_SURGE_THRESHOLD = 0.20
    SPEND_EFFICIENCY_THRESHOLD = 0.5

    def __init__(self) -> None:
        self._data_sources: list[str] = []

    @traced(name="business_monitor_collect", tags=["layer0", "business"])
    def collect_signals(self) -> list[BusinessSignal]:
        """Collect all business signals.

        Returns:
            List of BusinessSignal instances.
        """
        signals: list[BusinessSignal] = []

        signals.extend(self._check_revenue())
        signals.extend(self._check_inventory())
        signals.extend(self._check_crm_events())

        for signal in signals:
            signal.severity = self.compute_severity(signal)

        logger.info("business_signals_collected", count=len(signals))
        return signals

    def _check_revenue(self) -> list[BusinessSignal]:
        """Check revenue trends by comparing recent vs trailing 7-day average.

        Queries PerformanceMetric table for revenue data, compares last 24h
        against the prior 7-day average to detect surges or declines.
        """
        now = datetime.now(UTC)
        recent_start = now - timedelta(hours=24)
        trailing_start = now - timedelta(days=7)

        signals: list[BusinessSignal] = []

        with get_session() as session:
            # Revenue in last 24 hours
            recent_revenue = (
                session.query(func.sum(PerformanceMetric.revenue))
                .filter(PerformanceMetric.timestamp >= recent_start)
                .scalar()
            )
            recent_revenue = float(recent_revenue or 0.0)

            # Revenue in prior 7 days (daily average)
            trailing_revenue = (
                session.query(func.sum(PerformanceMetric.revenue))
                .filter(
                    PerformanceMetric.timestamp >= trailing_start,
                    PerformanceMetric.timestamp < recent_start,
                )
                .scalar()
            )
            trailing_daily_avg = float(trailing_revenue or 0.0) / 7.0

            # ROAS check: recent spend vs revenue
            recent_spend = (
                session.query(func.sum(PerformanceMetric.spend))
                .filter(PerformanceMetric.timestamp >= recent_start)
                .scalar()
            )
            recent_spend = float(recent_spend or 0.0)

        if trailing_daily_avg > 0:
            change_pct = (recent_revenue - trailing_daily_avg) / trailing_daily_avg
            if change_pct <= self.REVENUE_DECLINE_THRESHOLD:
                signals.append(
                    BusinessSignal(
                        source="revenue_monitor",
                        metric_name="daily_revenue",
                        current_value=recent_revenue,
                        baseline_value=trailing_daily_avg,
                        change_pct=change_pct,
                        category="revenue_decline",
                    )
                )
            elif change_pct >= self.REVENUE_SURGE_THRESHOLD:
                signals.append(
                    BusinessSignal(
                        source="revenue_monitor",
                        metric_name="daily_revenue",
                        current_value=recent_revenue,
                        baseline_value=trailing_daily_avg,
                        change_pct=change_pct,
                        category="revenue_surge",
                    )
                )

        # ROAS efficiency signal
        if recent_spend > 0:
            roas = recent_revenue / recent_spend
            if roas < self.SPEND_EFFICIENCY_THRESHOLD:
                signals.append(
                    BusinessSignal(
                        source="revenue_monitor",
                        metric_name="roas",
                        current_value=roas,
                        baseline_value=1.0,
                        change_pct=roas - 1.0,
                        category="low_roas",
                    )
                )

        return signals

    def _check_inventory(self) -> list[BusinessSignal]:
        """Check budget utilization across active campaigns.

        Detects campaigns with very low or very high budget utilization
        as proxy for inventory/capacity signals.
        """
        signals: list[BusinessSignal] = []
        now = datetime.now(UTC)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        with get_session() as session:
            active_campaigns = session.query(Campaign).filter(Campaign.status == "live").all()

            for campaign in active_campaigns:
                daily_spend = (
                    session.query(func.sum(PerformanceMetric.spend))
                    .filter(
                        PerformanceMetric.campaign_id == campaign.id,
                        PerformanceMetric.timestamp >= today_start,
                    )
                    .scalar()
                )
                daily_spend = float(daily_spend or 0.0)

                if campaign.daily_budget > 0:
                    utilization = daily_spend / campaign.daily_budget
                    # Underspend: less than 30% of budget used past noon
                    if utilization < 0.3 and now.hour >= 12:
                        signals.append(
                            BusinessSignal(
                                source="budget_monitor",
                                metric_name="budget_utilization",
                                current_value=utilization,
                                baseline_value=0.5,
                                change_pct=utilization - 0.5,
                                category="budget_underspend",
                                raw_data={"campaign_id": str(campaign.id)},
                            )
                        )
                    # Overspend: already hit 90% before end of day
                    elif utilization >= 0.9 and now.hour < 18:
                        signals.append(
                            BusinessSignal(
                                source="budget_monitor",
                                metric_name="budget_utilization",
                                current_value=utilization,
                                baseline_value=0.7,
                                change_pct=utilization - 0.7,
                                category="budget_overspend",
                                raw_data={"campaign_id": str(campaign.id)},
                            )
                        )

        return signals

    def _check_crm_events(self) -> list[BusinessSignal]:
        """Detect notable CRM events from campaign performance patterns.

        Uses conversion and click rate changes as proxies for customer
        engagement shifts (churn indicators, lead surges).
        """
        signals: list[BusinessSignal] = []
        now = datetime.now(UTC)
        recent_start = now - timedelta(hours=24)
        prior_start = now - timedelta(days=7)

        with get_session() as session:
            # Recent conversion rate
            recent = (
                session.query(
                    func.sum(PerformanceMetric.conversions),
                    func.sum(PerformanceMetric.clicks),
                )
                .filter(PerformanceMetric.timestamp >= recent_start)
                .one()
            )

            prior = (
                session.query(
                    func.sum(PerformanceMetric.conversions),
                    func.sum(PerformanceMetric.clicks),
                )
                .filter(
                    PerformanceMetric.timestamp >= prior_start,
                    PerformanceMetric.timestamp < recent_start,
                )
                .one()
            )

        recent_conversions = int(recent[0] or 0)
        recent_clicks = int(recent[1] or 0)
        prior_conversions = int(prior[0] or 0)
        prior_clicks = int(prior[1] or 0)

        # Conversion rate change
        recent_cvr = recent_conversions / recent_clicks if recent_clicks > 0 else 0.0
        prior_daily_cvr = (prior_conversions / prior_clicks) if prior_clicks > 0 else 0.0

        if prior_daily_cvr > 0:
            cvr_change = (recent_cvr - prior_daily_cvr) / prior_daily_cvr
            if abs(cvr_change) >= 0.15:
                category = "conversion_surge" if cvr_change > 0 else "conversion_decline"
                signals.append(
                    BusinessSignal(
                        source="crm_monitor",
                        metric_name="conversion_rate",
                        current_value=recent_cvr,
                        baseline_value=prior_daily_cvr,
                        change_pct=cvr_change,
                        category=category,
                    )
                )

        return signals

    def compute_severity(self, signal: BusinessSignal) -> float:
        """Compute severity score for a business signal.

        Severity is based on the magnitude and direction of change.
        """
        abs_change = abs(signal.change_pct)
        if abs_change >= 0.3:
            return 1.0
        elif abs_change >= 0.15:
            return 0.7
        elif abs_change >= 0.05:
            return 0.4
        return 0.1
