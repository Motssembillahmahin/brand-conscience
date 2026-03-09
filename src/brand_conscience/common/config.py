"""Configuration management using pydantic-settings + YAML."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class DatabaseSettings(BaseSettings):
    url: str = "postgresql+psycopg://brand_conscience:brand_conscience@localhost:5432/brand_conscience"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


class RedisSettings(BaseSettings):
    url: str = "redis://localhost:6379/0"


class MetaSettings(BaseSettings):
    app_id: str = ""
    app_secret: str = ""
    access_token: str = ""
    ad_account_id: str = ""


class GeminiSettings(BaseSettings):
    api_key: str = ""
    model: str = "gemini-2.0-flash-exp"
    max_retries: int = 3


class SlackSettings(BaseSettings):
    bot_token: str = ""
    channel_ops: str = "#brand-conscience-ops"
    channel_approvals: str = "#brand-conscience-approvals"


class OpikSettings(BaseSettings):
    api_key: str = ""
    url: str = "http://localhost:8080"
    project_name: str = "brand-conscience"


class MonitoringSettings(BaseSettings):
    business_interval_minutes: int = 15
    cultural_interval_minutes: int = 60
    creative_interval_minutes: int = 240
    urgency_threshold: float = 0.7
    urgency_weights: dict[str, float] = Field(
        default_factory=lambda: {"business": 0.5, "cultural": 0.3, "creative": 0.2}
    )


class StrategySettings(BaseSettings):
    update_interval_minutes: int = 60
    default_daily_budget: float = 500.0
    max_daily_budget: float = 5000.0
    audience_segments: list[str] = Field(
        default_factory=lambda: [
            "broad_interest",
            "retargeting",
            "lookalike",
            "custom_audience",
        ]
    )


class PromptsSettings(BaseSettings):
    scorer_threshold: float = 0.7
    max_prompts_per_moment: int = 5
    template_dir: str = "config/prompt_templates"


class CreativeSettings(BaseSettings):
    variants_per_prompt: int = 3
    quality_gate_classes: list[str] = Field(
        default_factory=lambda: ["excellent", "good"]
    )
    brand_alignment_threshold: float = 0.6
    originality_min_distance: float = 0.3
    performance_prediction_threshold: float = 0.5


class CircuitBreakerSettings(BaseSettings):
    velocity_multiplier: float = 3.0
    cooldown_minutes: int = 60
    resume_budget_fraction: float = 0.5
    window_hours: int = 2


class BrandSafetySettings(BaseSettings):
    similarity_threshold: float = 0.6
    risk_categories: list[str] = Field(
        default_factory=lambda: [
            "political_controversy",
            "natural_disaster",
            "social_unrest",
            "violence",
        ]
    )


class DiversitySettings(BaseSettings):
    min_distance: float = 0.3
    min_clusters: int = 5


class SafetySettings(BaseSettings):
    circuit_breaker: CircuitBreakerSettings = Field(
        default_factory=CircuitBreakerSettings
    )
    brand_safety: BrandSafetySettings = Field(default_factory=BrandSafetySettings)
    max_bid_cap_multiplier: float = 5.0
    diversity: DiversitySettings = Field(default_factory=DiversitySettings)


class DeploymentSettings(BaseSettings):
    spend_approval_threshold: float = 1000.0
    ab_test_holdout_fraction: float = 0.1
    thompson_convergence_threshold: float = 0.95
    attribution_click_days: int = 7
    attribution_view_days: int = 1


class TacticalSettings(BaseSettings):
    update_interval_minutes: int = 5
    max_bid_multiplier: float = 5.0
    warning_bid_multiplier: float = 2.0
    bid_change_cooldown_minutes: int = 5


class DriftSettings(BaseSettings):
    psi_threshold: float = 0.2
    check_interval_hours: int = 6
    retrain_lookback_days: int = 30
    min_samples_for_retrain: int = 1000


class Settings(BaseSettings):
    """Root application settings."""

    model_config = SettingsConfigDict(
        env_prefix="BC_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    app_env: str = "development"
    log_level: str = "INFO"
    log_format: str = "console"

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    meta: MetaSettings = Field(default_factory=MetaSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    slack: SlackSettings = Field(default_factory=SlackSettings)
    opik: OpikSettings = Field(default_factory=OpikSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    prompts: PromptsSettings = Field(default_factory=PromptsSettings)
    creative: CreativeSettings = Field(default_factory=CreativeSettings)
    deployment: DeploymentSettings = Field(default_factory=DeploymentSettings)
    tactical: TacticalSettings = Field(default_factory=TacticalSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    drift: DriftSettings = Field(default_factory=DriftSettings)


def load_settings(
    config_dir: Path = Path("config"),
    env_override: str | None = None,
) -> Settings:
    """Load settings from YAML files with environment-specific overrides.

    Load order: settings.yaml → settings.{env}.yaml → environment variables.
    """
    base = _load_yaml(config_dir / "settings.yaml")
    app_section = base.get("app", {})
    env = env_override or app_section.get("env", "development")

    override = _load_yaml(config_dir / f"settings.{env}.yaml")
    merged = _deep_merge(base, override)

    # Flatten app section into top level for Settings
    app_conf = merged.pop("app", {})
    merged["app_env"] = app_conf.get("env", env)
    merged["log_level"] = app_conf.get("log_level", "INFO")
    merged["log_format"] = app_conf.get("log_format", "console")

    return Settings(**merged)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings singleton."""
    return load_settings()
