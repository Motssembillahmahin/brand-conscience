"""Signal dataclasses for Layer 0 monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from brand_conscience.common.types import ActionType, SignalType, utc_now

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class BaseSignal:
    """Base signal from any monitor."""

    signal_type: SignalType
    source: str
    timestamp: datetime = field(default_factory=utc_now)
    raw_data: dict | None = None


@dataclass
class BusinessSignal(BaseSignal):
    """Signal from the business monitor."""

    signal_type: SignalType = field(default=SignalType.BUSINESS, init=False)
    metric_name: str = ""
    current_value: float = 0.0
    baseline_value: float = 0.0
    change_pct: float = 0.0
    category: str = ""
    severity: float = 0.0  # 0-1 scale


@dataclass
class CulturalSignal(BaseSignal):
    """Signal from the cultural monitor."""

    signal_type: SignalType = field(default=SignalType.CULTURAL, init=False)
    topic: str = ""
    sentiment: float = 0.0  # -1 to 1
    velocity: float = 0.0  # rate of change
    relevance: float = 0.0  # 0-1 relevance to brand
    is_safe: bool = True
    safety_flags: list[dict] = field(default_factory=list)


@dataclass
class CreativeSignal(BaseSignal):
    """Signal from the creative monitor."""

    signal_type: SignalType = field(default=SignalType.CREATIVE, init=False)
    campaign_id: str = ""
    creative_id: str = ""
    metric_name: str = ""
    current_value: float = 0.0
    trend_direction: float = 0.0  # negative = declining
    fatigue_score: float = 0.0  # 0-1, higher = more fatigued


@dataclass
class MomentProfile:
    """Aggregated signal profile representing the current business moment."""

    id: str = ""
    timestamp: datetime = field(default_factory=utc_now)
    urgency_score: float = 0.0
    recommended_action: ActionType = ActionType.HOLD
    business_signals: list[BusinessSignal] = field(default_factory=list)
    cultural_signals: list[CulturalSignal] = field(default_factory=list)
    creative_signals: list[CreativeSignal] = field(default_factory=list)
    affected_categories: list[str] = field(default_factory=list)
    affected_audiences: list[str] = field(default_factory=list)
    context_summary: str = ""

    def to_dict(self) -> dict:
        """Serialize to dict for database storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "urgency_score": self.urgency_score,
            "recommended_action": self.recommended_action.value,
            "business_signals": [
                {
                    "metric": s.metric_name,
                    "category": s.category,
                    "change_pct": s.change_pct,
                    "severity": s.severity,
                }
                for s in self.business_signals
            ],
            "cultural_signals": [
                {
                    "topic": s.topic,
                    "sentiment": s.sentiment,
                    "relevance": s.relevance,
                    "is_safe": s.is_safe,
                }
                for s in self.cultural_signals
            ],
            "creative_signals": [
                {
                    "campaign_id": s.campaign_id,
                    "metric": s.metric_name,
                    "fatigue_score": s.fatigue_score,
                }
                for s in self.creative_signals
            ],
            "affected_categories": self.affected_categories,
            "affected_audiences": self.affected_audiences,
            "context_summary": self.context_summary,
        }
