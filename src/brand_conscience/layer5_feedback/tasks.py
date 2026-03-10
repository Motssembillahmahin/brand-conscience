"""Celery periodic tasks for Layer 5 feedback."""

from __future__ import annotations

import numpy as np

from brand_conscience.common.logging import bind_context, get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.db.queries import (
    get_campaign,
    get_live_campaigns,
    get_total_daily_spend,
)
from brand_conscience.layer5_feedback.collector import MetricsCollector
from brand_conscience.layer5_feedback.reporter import Reporter
from brand_conscience.models.rl.reward import strategic_reward, tactical_reward

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
        # Look up meta_campaign_id from campaign record
        campaign = get_campaign(cid)
        meta_campaign_id = campaign.meta_campaign_id if campaign else ""
        metrics = collector.collect(cid, meta_campaign_id or "")
        all_metrics.append({"campaign_id": cid, **metrics})

    logger.info("feedback_metrics_collected", n_campaigns=len(campaign_ids))

    # Run drift detection against stored reference distributions
    drift_results = _run_drift_detection(all_metrics)

    # Compute RL rewards and log them
    rewards = _compute_rewards(all_metrics)

    return {
        "campaigns_processed": len(campaign_ids),
        "metrics": all_metrics,
        "drift_results": drift_results,
        "rewards": rewards,
    }


def _run_drift_detection(all_metrics: list[dict]) -> list[dict]:
    """Run PSI drift detection on collected metrics distributions."""
    from brand_conscience.layer5_feedback.drift_detector import DriftDetector

    detector = DriftDetector()
    results: list[dict] = []

    if len(all_metrics) < 10:
        return results

    # Check drift on key metric distributions
    for metric_name in ("ctr", "cpc", "roas"):
        values = [m.get(metric_name, 0.0) for m in all_metrics if m.get(metric_name) is not None]
        if len(values) < 5:
            continue

        arr = np.array(values, dtype=np.float64)
        midpoint = len(arr) // 2
        reference = arr[:midpoint]
        current = arr[midpoint:]

        if len(reference) < 2 or len(current) < 2:
            continue

        severity, psi = detector.check_drift(reference, current)
        should_retrain = detector.should_retrain(psi)

        results.append(
            {
                "metric": metric_name,
                "psi": psi,
                "severity": severity.value,
                "should_retrain": should_retrain,
                "model_name": f"performance_{metric_name}",
            }
        )

        if should_retrain:
            reporter = Reporter()
            reporter.send_drift_alert(
                model_name=f"performance_{metric_name}",
                psi_score=psi,
                severity=severity.value,
            )

    return results


def _compute_rewards(all_metrics: list[dict]) -> list[dict]:
    """Compute RL rewards for each campaign from collected metrics."""
    rewards = []
    for m in all_metrics:
        roas = m.get("roas", 0.0) or 0.0
        spend = m.get("spend", 0.0) or 0.0
        cpc = m.get("cpc", 0.0) or 0.0

        # Strategic reward: ROAS-based
        sr = strategic_reward(
            roas=roas,
            audience_quality_score=min(roas / 2.0, 1.0),
            budget_efficiency=min(spend / 500.0, 1.0),
        )

        # Tactical reward: efficiency-based
        cpc_eff = max(0.0, 1.0 - cpc / 5.0)  # lower CPC = better
        tr = tactical_reward(
            cpc_efficiency=cpc_eff,
            delivery_pacing=min(spend / 500.0, 1.0),
            spend_velocity_compliance=1.0,
        )

        rewards.append(
            {
                "campaign_id": m.get("campaign_id", ""),
                "strategic_reward": sr,
                "tactical_reward": tr,
            }
        )

    return rewards


@traced(name="run_daily_report", tags=["layer5", "task"])
def run_daily_report() -> None:
    """Generate and send daily performance summary."""
    bind_context(layer="layer5_feedback")

    # Aggregate metrics from database
    live_campaigns = get_live_campaigns()
    total_spend = get_total_daily_spend()

    total_impressions = 0
    total_clicks = 0
    total_revenue = 0.0
    for campaign in live_campaigns:
        from brand_conscience.db.queries import get_aggregate_metrics

        agg = get_aggregate_metrics(str(campaign.id))
        total_impressions += agg.get("impressions", 0)
        total_clicks += agg.get("clicks", 0)
        total_revenue += agg.get("revenue", 0.0)

    avg_ctr = total_clicks / total_impressions if total_impressions > 0 else 0.0
    avg_roas = total_revenue / total_spend if total_spend > 0 else 0.0

    reporter = Reporter()
    reporter.send_daily_summary(
        {
            "active_campaigns": len(live_campaigns),
            "total_spend": total_spend,
            "total_impressions": total_impressions,
            "avg_ctr": avg_ctr,
            "avg_roas": avg_roas,
            "campaigns_launched": 0,
            "campaigns_paused": 0,
            "circuit_breaker_trips": 0,
        }
    )

    logger.info("daily_report_sent")
