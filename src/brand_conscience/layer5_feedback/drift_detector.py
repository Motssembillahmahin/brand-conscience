"""Model drift detection using PSI and KL divergence."""

from __future__ import annotations

import numpy as np

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.common.types import DriftSeverity

logger = get_logger(__name__)


class DriftDetector:
    """Detect distribution drift in model inputs/outputs.

    Uses Population Stability Index (PSI) to measure drift
    between reference and current distributions.
    """

    def __init__(self, psi_threshold: float | None = None) -> None:
        settings = get_settings()
        self._threshold = psi_threshold or settings.drift.psi_threshold

    @traced(name="compute_psi", tags=["layer5", "drift"])
    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Population Stability Index between two distributions.

        Args:
            reference: Reference distribution values.
            current: Current distribution values.
            n_bins: Number of bins for histogram.

        Returns:
            PSI value. >0.2 indicates significant drift.
        """
        # Create bins from reference distribution
        breakpoints = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            n_bins + 1,
        )

        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]

        # Normalize to proportions
        ref_pct = (ref_counts + 1) / (ref_counts.sum() + n_bins)
        cur_pct = (cur_counts + 1) / (cur_counts.sum() + n_bins)

        # PSI formula
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    @traced(name="compute_kl_divergence", tags=["layer5", "drift"])
    def compute_kl_divergence(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute KL divergence from reference to current distribution."""
        breakpoints = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            n_bins + 1,
        )

        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]

        ref_pct = (ref_counts + 1) / (ref_counts.sum() + n_bins)
        cur_pct = (cur_counts + 1) / (cur_counts.sum() + n_bins)

        kl = float(np.sum(ref_pct * np.log(ref_pct / cur_pct)))
        return kl

    def check_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> tuple[DriftSeverity, float]:
        """Check for drift and classify severity.

        Returns:
            (severity, psi_score) tuple.
        """
        psi = self.compute_psi(reference, current)

        if psi < 0.1:
            severity = DriftSeverity.NONE
        elif psi < 0.2:
            severity = DriftSeverity.LOW
        elif psi < 0.5:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.HIGH

        if severity in (DriftSeverity.MEDIUM, DriftSeverity.HIGH):
            logger.warning(
                "drift_detected",
                psi=psi,
                severity=severity.value,
                threshold=self._threshold,
            )

        return severity, psi

    def should_retrain(self, psi: float) -> bool:
        """Determine if model retraining should be triggered."""
        return psi > self._threshold
