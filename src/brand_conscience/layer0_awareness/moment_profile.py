"""Moment profile aggregation — combine signals into actionable profile."""

from __future__ import annotations

import uuid

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.common.types import ActionType, utc_now
from brand_conscience.layer0_awareness.signals import (
    BusinessSignal,
    CreativeSignal,
    CulturalSignal,
    MomentProfile,
)

logger = get_logger(__name__)


class MomentProfileBuilder:
    """Aggregate signals from all monitors into a MomentProfile."""

    @traced(name="build_moment_profile", tags=["layer0", "moment"])
    def build(
        self,
        business_signals: list[BusinessSignal],
        cultural_signals: list[CulturalSignal],
        creative_signals: list[CreativeSignal],
    ) -> MomentProfile:
        """Build a MomentProfile from collected signals.

        Args:
            business_signals: Signals from business monitor.
            cultural_signals: Signals from cultural monitor.
            creative_signals: Signals from creative monitor.

        Returns:
            Aggregated MomentProfile.
        """
        settings = get_settings()
        weights = settings.monitoring.urgency_weights

        business_urgency = self._compute_business_urgency(business_signals)
        cultural_urgency = self._compute_cultural_urgency(cultural_signals)
        creative_urgency = self._compute_creative_urgency(creative_signals)

        urgency_score = (
            weights["business"] * business_urgency
            + weights["cultural"] * cultural_urgency
            + weights["creative"] * creative_urgency
        )

        recommended_action = self._determine_action(
            urgency_score, business_signals, cultural_signals, creative_signals
        )

        affected_categories = self._extract_categories(
            business_signals, cultural_signals, creative_signals
        )
        affected_audiences = self._determine_audiences(
            business_signals, cultural_signals
        )

        profile = MomentProfile(
            id=str(uuid.uuid4()),
            timestamp=utc_now(),
            urgency_score=urgency_score,
            recommended_action=recommended_action,
            business_signals=business_signals,
            cultural_signals=cultural_signals,
            creative_signals=creative_signals,
            affected_categories=affected_categories,
            affected_audiences=affected_audiences,
            context_summary=self._generate_summary(
                urgency_score, recommended_action, business_signals,
                cultural_signals, creative_signals,
            ),
        )

        logger.info(
            "moment_profile_built",
            profile_id=profile.id,
            urgency=urgency_score,
            action=recommended_action.value,
            n_business=len(business_signals),
            n_cultural=len(cultural_signals),
            n_creative=len(creative_signals),
        )
        return profile

    def _compute_business_urgency(self, signals: list[BusinessSignal]) -> float:
        if not signals:
            return 0.0
        return min(max(s.severity for s in signals), 1.0)

    def _compute_cultural_urgency(self, signals: list[CulturalSignal]) -> float:
        if not signals:
            return 0.0
        unsafe = [s for s in signals if not s.is_safe]
        if unsafe:
            return 0.9
        relevant = [s for s in signals if s.relevance > 0.5]
        if relevant:
            return max(s.relevance for s in relevant)
        return 0.0

    def _compute_creative_urgency(self, signals: list[CreativeSignal]) -> float:
        if not signals:
            return 0.0
        return min(max(s.fatigue_score for s in signals), 1.0)

    def _determine_action(
        self,
        urgency: float,
        business: list[BusinessSignal],
        cultural: list[CulturalSignal],
        creative: list[CreativeSignal],
    ) -> ActionType:
        # Safety-triggered pause
        unsafe_cultural = any(not s.is_safe for s in cultural)
        if unsafe_cultural:
            return ActionType.PAUSE

        # High urgency with business signals → launch new campaign
        if urgency > 0.7 and business:
            return ActionType.LAUNCH

        # Creative fatigue → refresh
        fatigued = any(s.fatigue_score > 0.7 for s in creative)
        if fatigued:
            return ActionType.REFRESH

        # Moderate urgency → adjust existing
        if urgency > 0.4:
            return ActionType.ADJUST

        return ActionType.HOLD

    def _extract_categories(
        self,
        business: list[BusinessSignal],
        cultural: list[CulturalSignal],
        creative: list[CreativeSignal],
    ) -> list[str]:
        categories: set[str] = set()
        for s in business:
            if s.category:
                categories.add(s.category)
        return sorted(categories) or ["general"]

    def _determine_audiences(
        self,
        business: list[BusinessSignal],
        cultural: list[CulturalSignal],
    ) -> list[str]:
        audiences: list[str] = []
        # Revenue drops suggest retargeting
        revenue_drops = [
            s for s in business if s.metric_name == "revenue" and s.change_pct < -0.1
        ]
        if revenue_drops:
            audiences.append("retargeting")
        # High cultural relevance suggests broad interest
        relevant_cultural = [s for s in cultural if s.relevance > 0.5 and s.is_safe]
        if relevant_cultural:
            audiences.append("broad_interest")
        return audiences or ["broad_interest"]

    def _generate_summary(
        self,
        urgency: float,
        action: ActionType,
        business: list[BusinessSignal],
        cultural: list[CulturalSignal],
        creative: list[CreativeSignal],
    ) -> str:
        parts: list[str] = [f"Urgency: {urgency:.2f}. Recommended: {action.value}."]
        if business:
            parts.append(f"{len(business)} business signal(s) detected.")
        if cultural:
            safe = sum(1 for s in cultural if s.is_safe)
            parts.append(f"{len(cultural)} cultural signal(s) ({safe} safe).")
        if creative:
            parts.append(f"{len(creative)} creative signal(s).")
        return " ".join(parts)
