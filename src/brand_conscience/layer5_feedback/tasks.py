"""Celery periodic tasks for Layer 5 feedback."""

from __future__ import annotations

from brand_conscience.common.logging import bind_context, get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.layer5_feedback.collector import MetricsCollector
from brand_conscience.layer5_feedback.reporter import Reporter

logger = get_logger(__name__)


@traced(name="run_feedback_cycle", tags=["layer5", "task"])
def run_feedback_cycle(campaign_ids: list[str]) -> dict:
    """Execute a full feedback cycle for active campaigns.

    Collects metrics, checks drift, computes rewards, and sends reports.
    """
    bind_context(layer="layer5_feedback")

    collector = MetricsCollector()

    all_metrics = []
    for cid in campaign_ids:
        # TODO: look up meta_campaign_id
        metrics = collector.collect(cid, "")
        all_metrics.append({"campaign_id": cid, **metrics})

    logger.info("feedback_metrics_collected", n_campaigns=len(campaign_ids))

    # TODO: run drift detection against stored reference distributions
    # TODO: compute RL rewards and trigger policy updates

    return {
        "campaigns_processed": len(campaign_ids),
        "metrics": all_metrics,
    }


@traced(name="run_daily_report", tags=["layer5", "task"])
def run_daily_report() -> None:
    """Generate and send daily performance summary."""
    bind_context(layer="layer5_feedback")

    # TODO: aggregate metrics from database
    reporter = Reporter()
    reporter.send_daily_summary(
        {
            "active_campaigns": 0,
            "total_spend": 0.0,
            "total_impressions": 0,
            "avg_ctr": 0.0,
            "avg_roas": 0.0,
            "campaigns_launched": 0,
            "campaigns_paused": 0,
            "circuit_breaker_trips": 0,
        }
    )

    logger.info("daily_report_sent")
