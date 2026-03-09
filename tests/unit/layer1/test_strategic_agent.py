"""Tests for strategic agent and state encoding."""

from __future__ import annotations

import torch

from brand_conscience.common.types import ActionType
from brand_conscience.layer0_awareness.signals import MomentProfile
from brand_conscience.layer1_strategy.state import StrategicState


def test_encode_state_shape():
    profile = MomentProfile(urgency_score=0.5, recommended_action=ActionType.HOLD)
    state = StrategicState.encode(profile)
    assert state.shape == (StrategicState.STATE_DIM,)
    assert state.dtype == torch.float32


def test_decode_action():
    decoded = StrategicState.decode_action(0)
    assert "audience_segment" in decoded
    assert "budget_fraction" in decoded
    assert decoded["audience_segment"] in [
        "broad_interest",
        "retargeting",
        "lookalike",
        "custom_audience",
    ]


def test_decode_action_boundary():
    decoded = StrategicState.decode_action(19)
    assert decoded["audience_segment"] == "custom_audience"
    assert decoded["budget_fraction"] == 1.0


def test_state_urgency_encoding():
    low_urgency = MomentProfile(urgency_score=0.1, recommended_action=ActionType.HOLD)
    high_urgency = MomentProfile(urgency_score=0.9, recommended_action=ActionType.LAUNCH)

    low_state = StrategicState.encode(low_urgency)
    high_state = StrategicState.encode(high_urgency)

    assert low_state[0] == 0.1
    assert high_state[0] == 0.9
