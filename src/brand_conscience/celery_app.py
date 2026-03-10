"""Celery application factory and beat schedule."""

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab
from celery.signals import task_prerun

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger

logger = get_logger(__name__)


def create_celery_app() -> Celery:
    """Create and configure the Celery application."""
    settings = get_settings()

    app = Celery(
        "brand_conscience",
        broker=settings.redis.url,
        backend=settings.redis.url,
    )

    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        worker_hijack_root_logger=False,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
    )

    # Beat schedule — periodic monitoring tasks
    app.conf.beat_schedule = {
        "business-monitor": {
            "task": "brand_conscience.tasks.run_business_monitor",
            "schedule": settings.monitoring.business_interval_minutes * 60,
        },
        "cultural-monitor": {
            "task": "brand_conscience.tasks.run_cultural_monitor",
            "schedule": settings.monitoring.cultural_interval_minutes * 60,
        },
        "creative-monitor": {
            "task": "brand_conscience.tasks.run_creative_monitor",
            "schedule": settings.monitoring.creative_interval_minutes * 60,
        },
        "tactical-optimization": {
            "task": "brand_conscience.tasks.run_tactical_optimization",
            "schedule": settings.tactical.update_interval_minutes * 60,
        },
        "drift-check": {
            "task": "brand_conscience.tasks.run_drift_check",
            "schedule": settings.drift.check_interval_hours * 3600,
        },
        "daily-report": {
            "task": "brand_conscience.tasks.run_daily_report",
            "schedule": crontab(hour=9, minute=0),
        },
    }

    # Register tasks
    @app.task(name="brand_conscience.tasks.run_business_monitor")
    def run_business_monitor() -> dict:
        from brand_conscience.layer0_awareness.tasks import run_monitoring_cycle

        return run_monitoring_cycle()

    @app.task(name="brand_conscience.tasks.run_cultural_monitor")
    def run_cultural_monitor() -> dict:
        from brand_conscience.layer0_awareness.tasks import run_monitoring_cycle

        return run_monitoring_cycle()

    @app.task(name="brand_conscience.tasks.run_creative_monitor")
    def run_creative_monitor() -> dict:
        from brand_conscience.layer0_awareness.tasks import run_monitoring_cycle

        return run_monitoring_cycle()

    @app.task(name="brand_conscience.tasks.run_tactical_optimization")
    def run_tactical_optimization() -> dict:
        from brand_conscience.db.queries import get_live_campaigns
        from brand_conscience.layer4_deployment.tasks import run_tactical_cycle

        campaigns = get_live_campaigns()
        if not campaigns:
            return {"status": "no_live_campaigns"}

        results = []
        for campaign in campaigns:
            result = run_tactical_cycle(str(campaign.id))
            results.append(result)

        return {"status": "complete", "campaigns_optimized": len(results), "results": results}

    @app.task(name="brand_conscience.tasks.run_drift_check")
    def run_drift_check() -> dict:
        import numpy as np

        from brand_conscience.db.queries import get_live_campaigns
        from brand_conscience.layer5_feedback.drift_detector import DriftDetector

        detector = DriftDetector()
        campaigns = get_live_campaigns()
        drift_found: list[dict] = []

        for campaign in campaigns:
            from brand_conscience.db.queries import get_campaign_metrics

            hours = settings.drift.check_interval_hours
            metrics = get_campaign_metrics(str(campaign.id), hours=hours)
            if len(metrics) < 10:
                continue

            for metric_name in ("ctr", "cpc", "roas"):
                values = [getattr(m, metric_name, None) for m in metrics]
                values = [v for v in values if v is not None]
                if len(values) < 4:
                    continue

                arr = np.array(values, dtype=np.float64)
                midpoint = len(arr) // 2
                severity, psi = detector.check_drift(arr[:midpoint], arr[midpoint:])
                if detector.should_retrain(psi):
                    drift_found.append(
                        {
                            "campaign_id": str(campaign.id),
                            "metric": metric_name,
                            "psi": psi,
                            "severity": severity.value,
                        }
                    )

        if drift_found:
            from brand_conscience.layer5_feedback.model_updater import ModelUpdater

            updater = ModelUpdater()
            for d in drift_found:
                updater.trigger_retrain(
                    model_name=f"performance_{d['metric']}",
                    reason=f"PSI={d['psi']:.3f} on campaign {d['campaign_id']}",
                )

        return {"status": "complete", "drift_results": drift_found}

    @app.task(name="brand_conscience.tasks.retrain_model")
    def retrain_model(model_name: str, reason: str) -> dict:
        from brand_conscience.common.logging import bind_context

        bind_context(model_name=model_name)
        # Model-specific retrain dispatch
        logger.info("retrain_job_started", model_name=model_name, reason=reason)
        # Actual training would go here per model type
        return {"model_name": model_name, "status": "completed", "reason": reason}

    @app.task(name="brand_conscience.tasks.run_daily_report")
    def run_daily_report_task() -> None:
        from brand_conscience.layer5_feedback.tasks import run_daily_report

        run_daily_report()

    return app


# Module-level Celery instance for `celery -A brand_conscience.celery_app`
app = create_celery_app()


@task_prerun.connect
def bind_task_context(sender: object = None, **kwargs: object) -> None:
    """Bind Celery task context to structlog on task start."""
    from brand_conscience.common.logging import bind_context

    task = kwargs.get("task")
    if task and hasattr(task, "name") and hasattr(task, "request"):
        bind_context(
            task_name=task.name,
            task_id=str(task.request.id),
        )
