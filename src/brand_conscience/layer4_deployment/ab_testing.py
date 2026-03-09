"""Thompson Sampling Multi-Armed Bandit for A/B testing."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import numpy as np

from brand_conscience.common.config import get_settings
from brand_conscience.common.database import get_session
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.db.tables import ABTestGroup

logger = get_logger(__name__)


class ThompsonSamplingMAB:
    """Thompson Sampling for creative variant A/B testing.

    Each variant is modeled as a Beta distribution.
    Budget allocation is proportional to sampled probabilities.
    """

    @traced(name="setup_ab_test", tags=["layer4", "ab_testing"])
    def setup_test(
        self,
        campaign_id: str,
        variant_ids: list[str],
    ) -> list[str]:
        """Set up A/B test groups for a campaign.

        Creates one group per variant plus a holdout group.

        Returns:
            List of group IDs.
        """
        settings = get_settings()
        group_ids: list[str] = []

        with get_session() as session:
            # Create variant groups
            for i, vid in enumerate(variant_ids):
                gid = str(uuid.uuid4())
                group = ABTestGroup(
                    id=uuid.UUID(gid),
                    campaign_id=uuid.UUID(campaign_id),
                    group_name=f"variant_{i}",
                    variant_id=vid,
                    alpha=1.0,
                    beta=1.0,
                    is_holdout=False,
                    created_at=datetime.now(UTC),
                )
                session.add(group)
                group_ids.append(gid)

            # Create holdout group
            holdout_id = str(uuid.uuid4())
            holdout = ABTestGroup(
                id=uuid.UUID(holdout_id),
                campaign_id=uuid.UUID(campaign_id),
                group_name="holdout",
                variant_id="holdout",
                alpha=1.0,
                beta=1.0,
                is_holdout=True,
                created_at=datetime.now(UTC),
            )
            session.add(holdout)
            group_ids.append(holdout_id)

        logger.info(
            "ab_test_setup",
            campaign_id=campaign_id,
            n_variants=len(variant_ids),
            holdout_fraction=settings.deployment.ab_test_holdout_fraction,
        )
        return group_ids

    @traced(name="thompson_sample", tags=["layer4", "ab_testing"])
    def get_allocation(self, campaign_id: str) -> dict[str, float]:
        """Sample from Beta distributions to determine budget allocation.

        Returns:
            Dict mapping variant_id → budget fraction.
        """
        settings = get_settings()
        holdout_frac = settings.deployment.ab_test_holdout_fraction

        with get_session() as session:
            groups = session.query(ABTestGroup).filter_by(campaign_id=uuid.UUID(campaign_id)).all()

        variants = [g for g in groups if not g.is_holdout]
        if not variants:
            return {}

        # Sample from Beta distributions
        samples = {}
        for v in variants:
            sample = np.random.beta(v.alpha, v.beta)
            samples[v.variant_id] = sample

        # Normalize to budget fractions (excluding holdout)
        total = sum(samples.values())
        allocation = {}
        for vid, sample in samples.items():
            allocation[vid] = (sample / total) * (1 - holdout_frac)

        allocation["holdout"] = holdout_frac

        logger.debug("thompson_allocation", campaign_id=campaign_id, allocation=allocation)
        return allocation

    @traced(name="update_ab_results", tags=["layer4", "ab_testing"])
    def update(
        self,
        campaign_id: str,
        variant_id: str,
        impressions: int,
        conversions: int,
    ) -> None:
        """Update Beta distribution parameters with new results."""
        with get_session() as session:
            group = (
                session.query(ABTestGroup)
                .filter_by(
                    campaign_id=uuid.UUID(campaign_id),
                    variant_id=variant_id,
                )
                .first()
            )
            if group:
                group.impressions += impressions
                group.conversions += conversions
                group.alpha += conversions
                group.beta += impressions - conversions

        logger.debug(
            "ab_test_updated",
            campaign_id=campaign_id,
            variant_id=variant_id,
            impressions=impressions,
            conversions=conversions,
        )

    def has_converged(self, campaign_id: str) -> tuple[bool, str | None]:
        """Check if the test has converged to a winning variant.

        Returns:
            (converged, winning_variant_id) tuple.
        """
        settings = get_settings()
        threshold = settings.deployment.thompson_convergence_threshold

        with get_session() as session:
            groups = (
                session.query(ABTestGroup)
                .filter_by(campaign_id=uuid.UUID(campaign_id), is_holdout=False)
                .all()
            )

        if not groups:
            return False, None

        # Monte Carlo estimation of P(best)
        n_samples = 10000
        samples = np.zeros((n_samples, len(groups)))
        for i, g in enumerate(groups):
            samples[:, i] = np.random.beta(g.alpha, g.beta, n_samples)

        winners = samples.argmax(axis=1)
        win_probs = np.bincount(winners, minlength=len(groups)) / n_samples

        best_idx = int(win_probs.argmax())
        best_prob = float(win_probs[best_idx])

        if best_prob >= threshold:
            return True, groups[best_idx].variant_id
        return False, None
