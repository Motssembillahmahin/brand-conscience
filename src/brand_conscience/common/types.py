"""Shared types, enums, and type aliases."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TypedDict


class AppEnv(str, enum.Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TEST = "test"


class ActionType(str, enum.Enum):
    LAUNCH = "launch"
    ADJUST = "adjust"
    PAUSE = "pause"
    REFRESH = "refresh"
    HOLD = "hold"


class CampaignStatus(str, enum.Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    LIVE = "live"
    PAUSED = "paused"
    COMPLETED = "completed"


class QualityTier(str, enum.Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    REJECT = "reject"


class SignalType(str, enum.Enum):
    BUSINESS = "business"
    CULTURAL = "cultural"
    CREATIVE = "creative"


class GateResult(str, enum.Enum):
    PASSED = "passed"
    REJECTED = "rejected"


class DriftSeverity(str, enum.Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Type aliases
TraceId = str
CampaignId = str
AdSetId = str
AdId = str
CreativeId = str


class MomentProfileDict(TypedDict):
    timestamp: str
    urgency_score: float
    recommended_action: str
    affected_categories: list[str]
    affected_audiences: list[str]
    context_summary: str


class StrategicDecisionDict(TypedDict):
    audience_segments: list[str]
    daily_budget: float
    campaign_objective: str
    moment_profile_id: str


class GateScoreDict(TypedDict):
    gate_name: str
    score: float
    threshold: float
    result: str


class RewardDict(TypedDict):
    strategic_reward: float
    tactical_reward: float
    timestamp: str
    campaign_id: str


# Constants
VALID_CAMPAIGN_TRANSITIONS: dict[CampaignStatus, list[CampaignStatus]] = {
    CampaignStatus.DRAFT: [CampaignStatus.PENDING_APPROVAL],
    CampaignStatus.PENDING_APPROVAL: [CampaignStatus.LIVE, CampaignStatus.DRAFT],
    CampaignStatus.LIVE: [CampaignStatus.PAUSED, CampaignStatus.COMPLETED],
    CampaignStatus.PAUSED: [CampaignStatus.LIVE, CampaignStatus.COMPLETED],
    CampaignStatus.COMPLETED: [],
}


def validate_campaign_transition(
    current: CampaignStatus, target: CampaignStatus
) -> bool:
    """Check if a campaign status transition is valid."""
    return target in VALID_CAMPAIGN_TRANSITIONS.get(current, [])


def utc_now() -> datetime:
    """Return current UTC datetime."""
    from datetime import UTC

    return datetime.now(UTC)
