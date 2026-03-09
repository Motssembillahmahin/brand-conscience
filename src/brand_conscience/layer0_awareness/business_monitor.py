"""Business monitor — revenue, inventory, CRM signals (15-minute cadence)."""

from __future__ import annotations

from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.layer0_awareness.signals import BusinessSignal

logger = get_logger(__name__)


class BusinessMonitor:
    """Collect business health signals.

    Monitors revenue trends, inventory levels, and CRM events
    to detect advertising opportunities and threats.
    """

    def __init__(self) -> None:
        self._data_sources: list[str] = []

    @traced(name="business_monitor_collect", tags=["layer0", "business"])
    def collect_signals(self) -> list[BusinessSignal]:
        """Collect all business signals.

        Returns:
            List of BusinessSignal instances.
        """
        signals: list[BusinessSignal] = []

        signals.extend(self._check_revenue())
        signals.extend(self._check_inventory())
        signals.extend(self._check_crm_events())

        logger.info("business_signals_collected", count=len(signals))
        return signals

    def _check_revenue(self) -> list[BusinessSignal]:
        """Check revenue trends by category against 7-day trailing average."""
        # TODO: integrate with actual revenue data source
        return []

    def _check_inventory(self) -> list[BusinessSignal]:
        """Check inventory levels for surplus or shortage."""
        # TODO: integrate with inventory management system
        return []

    def _check_crm_events(self) -> list[BusinessSignal]:
        """Check CRM for notable events (churn spikes, lead surges)."""
        # TODO: integrate with CRM API
        return []

    def compute_severity(self, signal: BusinessSignal) -> float:
        """Compute severity score for a business signal.

        Severity is based on the magnitude and direction of change.
        """
        abs_change = abs(signal.change_pct)
        if abs_change >= 0.3:
            return 1.0
        elif abs_change >= 0.15:
            return 0.7
        elif abs_change >= 0.05:
            return 0.4
        return 0.1
