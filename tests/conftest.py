"""Shared test fixtures."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from brand_conscience.common.config import Settings
from brand_conscience.common.database import Base
from brand_conscience.common.types import ActionType, CampaignStatus, QualityTier


@pytest.fixture
def settings() -> Settings:
    """Return test settings."""
    return Settings(
        app_env="test",
        log_level="DEBUG",
        log_format="console",
    )


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provide an in-memory SQLite session for unit tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def mock_slack() -> MagicMock:
    """Return a mock Slack client."""
    return MagicMock()


@pytest.fixture
def sample_moment_profile() -> dict:
    """Return a sample moment profile dict."""
    return {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "urgency_score": 0.75,
        "recommended_action": ActionType.LAUNCH.value,
        "business_signals": [
            {"type": "revenue_drop", "category": "electronics", "magnitude": -0.2}
        ],
        "cultural_signals": [],
        "creative_signals": [],
        "affected_categories": ["electronics"],
        "affected_audiences": ["retargeting"],
        "context_summary": "Revenue drop detected in electronics category",
    }


@pytest.fixture
def sample_campaign_data() -> dict:
    """Return sample campaign creation data."""
    return {
        "name": "Test Campaign - Electronics Promo",
        "status": CampaignStatus.DRAFT.value,
        "objective": "conversions",
        "daily_budget": 500.0,
    }


@pytest.fixture
def sample_creative_data() -> dict:
    """Return sample creative data."""
    return {
        "prompt_text": "Modern electronics on a clean white background",
        "prompt_score": 0.85,
        "image_path": "/creatives/test-image-001.png",
        "quality_tier": QualityTier.EXCELLENT.value,
        "brand_alignment_score": 0.78,
        "originality_score": 0.65,
        "predicted_ctr": 0.032,
    }
