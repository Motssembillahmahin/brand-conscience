"""Integration test: Celery task registration."""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_celery_app_creates():
    """Verify Celery app can be created and tasks are registered."""
    from brand_conscience.celery_app import create_celery_app

    app = create_celery_app()
    assert app is not None
    assert "brand_conscience.tasks.run_business_monitor" in app.tasks
    assert "brand_conscience.tasks.run_daily_report" in app.tasks
