"""Integration test: Layer 0 → Layer 1 pipeline."""

from __future__ import annotations

import pytest

from brand_conscience.layer0_awareness.moment_profile import MomentProfileBuilder
from brand_conscience.layer0_awareness.signals import BusinessSignal
from brand_conscience.layer1_strategy.state import StrategicState


@pytest.mark.integration
def test_moment_profile_to_strategic_state():
    """Verify that a MomentProfile can be encoded into a strategic state vector."""
    builder = MomentProfileBuilder()
    signal = BusinessSignal(
        source="test",
        metric_name="revenue",
        category="electronics",
        change_pct=-0.2,
        severity=0.8,
    )
    profile = builder.build([signal], [], [])

    state = StrategicState.encode(
        moment_profile=profile,
        active_campaigns=2,
        total_spend=1500.0,
        budget_remaining=3500.0,
    )

    assert state.shape == (StrategicState.STATE_DIM,)
    assert state[0] == profile.urgency_score
