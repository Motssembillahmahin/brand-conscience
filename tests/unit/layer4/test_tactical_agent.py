"""Tests for tactical agent state encoding."""

from __future__ import annotations

import torch

from brand_conscience.layer4_deployment.tactical_agent import TacticalState


def test_encode_state_shape():
    state = TacticalState.encode(
        current_ctr=0.02,
        current_cpc=1.5,
        current_spend=200.0,
        daily_budget=1000.0,
    )
    assert state.shape == (TacticalState.STATE_DIM,)
    assert state.dtype == torch.float32


def test_encode_normalizes_values():
    state = TacticalState.encode(
        current_cpc=5.0,
        daily_budget=1000.0,
        current_spend=800.0,
    )
    # Budget pacing = 800/1000 = 0.8, so last feature should be 1.0
    assert state[-1] == 1.0


def test_encode_default_values():
    state = TacticalState.encode()
    assert state.shape == (TacticalState.STATE_DIM,)
    assert not state.isnan().any()
