"""Creative monitor — ad fatigue, competitor shifts (4-hour cadence)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced

if TYPE_CHECKING:
    from brand_conscience.layer0_awareness.signals import CreativeSignal

logger = get_logger(__name__)


class CreativeMonitor:
    """Monitor creative performance and detect fatigue.

    Tracks CTR trends, frequency saturation, and competitor
    creative shifts.
    """

    FATIGUE_CTR_DECLINE_THRESHOLD = 0.30
    FATIGUE_FREQUENCY_THRESHOLD = 5.0

    @traced(name="creative_monitor_collect", tags=["layer0", "creative"])
    def collect_signals(self) -> list[CreativeSignal]:
        """Collect creative performance signals.

        Returns:
            List of CreativeSignal instances.
        """
        signals: list[CreativeSignal] = []

        signals.extend(self._check_ctr_trends())
        signals.extend(self._check_frequency_saturation())
        signals.extend(self._check_competitor_shifts())

        logger.info("creative_signals_collected", count=len(signals))
        return signals

    def _check_ctr_trends(self) -> list[CreativeSignal]:
        """Detect CTR decline patterns indicating creative fatigue."""
        # TODO: query Meta API for creative-level CTR trends
        return []

    def _check_frequency_saturation(self) -> list[CreativeSignal]:
        """Check if audience frequency caps are being hit."""
        # TODO: query Meta API for frequency data
        return []

    def _check_competitor_shifts(self) -> list[CreativeSignal]:
        """Detect significant changes in competitor creative strategies."""
        # TODO: integrate with competitive intelligence data
        return []

    def compute_fatigue_score(self, ctr_change: float, days_active: int, frequency: float) -> float:
        """Compute fatigue score for a creative.

        Returns:
            Score 0-1, higher = more fatigued.
        """
        fatigue = 0.0

        if ctr_change < 0:
            fatigue += min(abs(ctr_change) / self.FATIGUE_CTR_DECLINE_THRESHOLD, 1.0) * 0.5

        if days_active > 7:
            fatigue += min((days_active - 7) / 14, 1.0) * 0.25

        if frequency > self.FATIGUE_FREQUENCY_THRESHOLD:
            fatigue += min((frequency - self.FATIGUE_FREQUENCY_THRESHOLD) / 5.0, 1.0) * 0.25

        return min(fatigue, 1.0)
