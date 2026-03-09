"""Tests for configuration loading."""

from __future__ import annotations

from brand_conscience.common.config import Settings, load_settings


def test_default_settings():
    settings = Settings()
    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.database.pool_size == 10


def test_settings_nested_access():
    settings = Settings()
    assert settings.monitoring.business_interval_minutes == 15
    assert settings.monitoring.cultural_interval_minutes == 60
    assert settings.monitoring.creative_interval_minutes == 240


def test_safety_defaults():
    settings = Settings()
    assert settings.safety.circuit_breaker.velocity_multiplier == 3.0
    assert settings.safety.circuit_breaker.cooldown_minutes == 60
    assert settings.safety.brand_safety.similarity_threshold == 0.6


def test_urgency_weights():
    settings = Settings()
    weights = settings.monitoring.urgency_weights
    assert weights["business"] == 0.5
    assert weights["cultural"] == 0.3
    assert weights["creative"] == 0.2
    assert sum(weights.values()) == 1.0
