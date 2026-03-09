"""Celery application factory and beat schedule."""

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab
from celery.signals import task_prerun

from brand_conscience.common.config import get_settings


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
        from brand_conscience.layer4_deployment.tasks import run_tactical_cycle

        # TODO: iterate over all live campaigns
        return {"status": "no_live_campaigns"}

    @app.task(name="brand_conscience.tasks.run_drift_check")
    def run_drift_check() -> dict:
        # TODO: implement drift check across all models
        return {"status": "no_drift_detected"}

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
