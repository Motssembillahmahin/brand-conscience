"""E2E test fixtures — requires running infrastructure (postgres, redis)."""

from __future__ import annotations

import pytest

from brand_conscience.common.config import load_settings
from brand_conscience.common.database import init_database


@pytest.fixture(autouse=True)
def _init_db() -> None:
    """Initialize database connection for e2e tests."""
    settings = load_settings()
    init_database(settings)
