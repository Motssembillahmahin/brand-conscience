"""Slack reporting — daily summaries and alerts."""

from __future__ import annotations

from brand_conscience.common.logging import get_logger
from brand_conscience.common.notifications import SlackNotifier
from brand_conscience.common.tracing import traced

logger = get_logger(__name__)


class Reporter:
    """Generate and send performance reports via Slack."""

    def __init__(self, notifier: SlackNotifier | None = None) -> None:
        self._notifier = notifier or SlackNotifier()

    @traced(name="send_daily_summary", tags=["layer5", "reporting"])
    def send_daily_summary(self, metrics: dict) -> None:
        """Send a daily performance summary to Slack.

        Args:
            metrics: Aggregated daily metrics dict.
        """
        summary = self._format_daily_summary(metrics)
        self._notifier.send_daily_summary(summary)
        logger.info("daily_summary_sent")

    @traced(name="send_drift_alert", tags=["layer5", "reporting"])
    def send_drift_alert(
        self,
        model_name: str,
        psi_score: float,
        severity: str,
    ) -> None:
        """Send a drift detection alert."""
        message = (
            f"Drift detected in `{model_name}`\n"
            f"PSI: {psi_score:.3f} | Severity: {severity}\n"
            f"Automatic retraining has been triggered."
        )
        self._notifier.send_ops_alert(message)
        logger.info("drift_alert_sent", model_name=model_name, psi=psi_score)

    @traced(name="send_retrain_result", tags=["layer5", "reporting"])
    def send_retrain_result(
        self,
        model_name: str,
        old_metrics: dict,
        new_metrics: dict,
        promoted: bool,
    ) -> None:
        """Send model retrain result notification."""
        status = "Promoted" if promoted else "Rejected (no improvement)"
        message = (
            f"Model retrain complete: `{model_name}`\n"
            f"Status: {status}\n"
            f"Old metrics: {old_metrics}\n"
            f"New metrics: {new_metrics}"
        )
        self._notifier.send_ops_alert(message)

    def _format_daily_summary(self, metrics: dict) -> str:
        """Format daily metrics into a readable summary."""
        lines = [
            "Daily Performance Summary",
            "=" * 30,
            f"Active campaigns: {metrics.get('active_campaigns', 0)}",
            f"Total spend: ${metrics.get('total_spend', 0):,.2f}",
            f"Total impressions: {metrics.get('total_impressions', 0):,}",
            f"Avg CTR: {metrics.get('avg_ctr', 0):.2%}",
            f"Avg ROAS: {metrics.get('avg_roas', 0):.2f}x",
            f"Campaigns launched: {metrics.get('campaigns_launched', 0)}",
            f"Campaigns paused: {metrics.get('campaigns_paused', 0)}",
            f"Circuit breaker trips: {metrics.get('circuit_breaker_trips', 0)}",
        ]
        return "\n".join(lines)
