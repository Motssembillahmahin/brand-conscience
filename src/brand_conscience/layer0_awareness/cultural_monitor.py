"""Cultural monitor — social trends, sentiment, brand safety (1-hour cadence)."""

from __future__ import annotations

from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.layer0_awareness.signals import CulturalSignal
from brand_conscience.models.safety.brand_safety import BrandSafetyClassifier

logger = get_logger(__name__)


class CulturalMonitor:
    """Collect cultural and social trend signals.

    Every signal passes through the brand safety classifier before
    being included in the MomentProfile.
    """

    def __init__(
        self,
        brand_safety: BrandSafetyClassifier | None = None,
    ) -> None:
        self._safety = brand_safety or BrandSafetyClassifier()

    @traced(name="cultural_monitor_collect", tags=["layer0", "cultural"])
    def collect_signals(self) -> list[CulturalSignal]:
        """Collect cultural signals with brand safety screening.

        Returns:
            List of CulturalSignal instances with safety status.
        """
        raw_signals = self._fetch_raw_signals()
        screened = self._screen_for_safety(raw_signals)

        logger.info(
            "cultural_signals_collected",
            total=len(raw_signals),
            safe=sum(1 for s in screened if s.is_safe),
            flagged=sum(1 for s in screened if not s.is_safe),
        )
        return screened

    def _fetch_raw_signals(self) -> list[CulturalSignal]:
        """Fetch raw cultural signals from social/news APIs."""
        # TODO: integrate with social media APIs, news APIs
        return []

    def _screen_for_safety(
        self, signals: list[CulturalSignal]
    ) -> list[CulturalSignal]:
        """Run brand safety classifier on each signal."""
        for signal in signals:
            if signal.topic:
                is_safe, flags = self._safety.check(signal.topic)
                signal.is_safe = is_safe
                signal.safety_flags = flags
        return signals
