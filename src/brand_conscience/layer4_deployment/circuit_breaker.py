"""Circuit breaker — spend velocity protection and max bid cap."""

from __future__ import annotations

from datetime import UTC, datetime

from brand_conscience.common.config import get_settings
from brand_conscience.common.exceptions import BidCapExceededError, CircuitBreakerTrippedError
from brand_conscience.common.logging import get_logger
from brand_conscience.common.notifications import SlackNotifier
from brand_conscience.common.tracing import traced

logger = get_logger(__name__)


class CircuitBreaker:
    """Prevent runaway spend through velocity monitoring and bid caps.

    Triggers:
    - Spend velocity exceeds N× daily budget rate in a 2-hour window
    - Single campaign exceeds 150% of allocated budget
    - Any bid exceeds max cap
    """

    def __init__(self, notifier: SlackNotifier | None = None) -> None:
        self._notifier = notifier or SlackNotifier()
        self._tripped = False
        self._trip_time: datetime | None = None

    @property
    def is_tripped(self) -> bool:
        """Check if the circuit breaker is currently tripped."""
        if not self._tripped:
            return False

        # Check if cooldown has elapsed
        settings = get_settings()
        if self._trip_time:
            elapsed = (datetime.now(UTC) - self._trip_time).total_seconds() / 60
            if elapsed >= settings.safety.circuit_breaker.cooldown_minutes:
                self._tripped = False
                self._trip_time = None
                logger.info("circuit_breaker_reset")
                return False

        return True

    @traced(name="check_spend_velocity", tags=["layer4", "safety"])
    def check_spend_velocity(
        self,
        current_spend: float,
        daily_budget: float,
        hours_elapsed: float,
    ) -> bool:
        """Check if spend velocity is within safe bounds.

        Returns:
            True if safe, False if tripped.
        """
        settings = get_settings()
        window_hours = settings.safety.circuit_breaker.window_hours
        velocity_mult = settings.safety.circuit_breaker.velocity_multiplier

        if hours_elapsed <= 0:
            return True

        # Expected spend rate
        expected_rate = daily_budget / 24.0

        # Actual spend rate (per hour)
        actual_rate = current_spend / hours_elapsed

        if actual_rate > expected_rate * velocity_mult:
            self._trip(
                reason=(
                    f"Spend velocity {actual_rate:.2f}/hr exceeds "
                    f"{velocity_mult}× expected rate {expected_rate:.2f}/hr"
                ),
                details={
                    "current_spend": current_spend,
                    "daily_budget": daily_budget,
                    "hours_elapsed": hours_elapsed,
                    "actual_rate": actual_rate,
                    "expected_rate": expected_rate,
                },
            )
            return False

        return True

    @traced(name="check_campaign_spend", tags=["layer4", "safety"])
    def check_campaign_spend(
        self,
        campaign_spend: float,
        campaign_budget: float,
    ) -> bool:
        """Check if a single campaign has exceeded its budget.

        Returns:
            True if safe, False if exceeded.
        """
        if campaign_spend > campaign_budget * 1.5:
            self._trip(
                reason=(
                    f"Campaign spend ${campaign_spend:.2f} exceeds "
                    f"150% of budget ${campaign_budget:.2f}"
                ),
                details={
                    "campaign_spend": campaign_spend,
                    "campaign_budget": campaign_budget,
                },
            )
            return False
        return True

    def check_bid(self, bid_amount: float, target_cpc: float) -> float:
        """Validate and cap a bid amount.

        Args:
            bid_amount: Proposed bid.
            target_cpc: Target CPC for the campaign.

        Returns:
            Capped bid amount.

        Raises:
            BidCapExceededError: If bid exceeds hard cap.
        """
        settings = get_settings()
        max_mult = settings.safety.max_bid_cap_multiplier
        hard_cap = target_cpc * max_mult

        if bid_amount > hard_cap:
            logger.error(
                "bid_cap_exceeded",
                bid=bid_amount,
                hard_cap=hard_cap,
                target_cpc=target_cpc,
            )
            raise BidCapExceededError(
                f"Bid ${bid_amount:.2f} exceeds hard cap ${hard_cap:.2f}"
            )

        warn_mult = settings.tactical.warning_bid_multiplier
        if bid_amount > target_cpc * warn_mult:
            logger.warning(
                "bid_warning",
                bid=bid_amount,
                warning_threshold=target_cpc * warn_mult,
            )

        return min(bid_amount, hard_cap)

    def _trip(self, reason: str, details: dict | None = None) -> None:
        """Trip the circuit breaker."""
        self._tripped = True
        self._trip_time = datetime.now(UTC)

        logger.error("circuit_breaker_tripped", reason=reason, details=details)

        self._notifier.send_ops_alert(
            f"CIRCUIT BREAKER TRIPPED\n{reason}\nDetails: {details}"
        )

        raise CircuitBreakerTrippedError(reason=reason)

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._tripped = False
        self._trip_time = None
        logger.info("circuit_breaker_manually_reset")
