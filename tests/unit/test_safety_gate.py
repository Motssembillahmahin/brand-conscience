"""Tests for the safety_gate pipeline node."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from brand_conscience.app import safety_gate


def _make_state(cultural_signals: list[dict]) -> dict:
    return {
        "moment_profile": {
            "cultural_signals": cultural_signals,
        },
        "should_proceed": True,
    }


@patch("brand_conscience.db.queries.get_live_campaigns")
def test_safety_gate_no_unsafe_signals_proceeds(mock_live):
    """No unsafe signals → should_proceed stays True, no campaigns queried."""
    state = _make_state([{"topic": "sports", "is_safe": True}])

    result = safety_gate(state)

    assert result["should_proceed"] is True
    mock_live.assert_not_called()


@patch("brand_conscience.common.notifications.SlackNotifier")
@patch("brand_conscience.layer4_deployment.campaign_manager.CampaignManager")
@patch("brand_conscience.models.safety.impact_matcher.SafetyImpactMatcher")
@patch("brand_conscience.db.queries.get_live_campaigns")
def test_safety_gate_pauses_matched_campaigns(
    mock_live,
    mock_matcher_cls,
    mock_mgr_cls,
    mock_notifier_cls,
):
    """Unsafe signal + matched campaign → pause it, continue for safe ones."""
    campaign_travel = SimpleNamespace(id="c-travel", name="Travel Deals")
    campaign_tech = SimpleNamespace(id="c-tech", name="Tech Sale")
    mock_live.return_value = [campaign_travel, campaign_tech]

    mock_result = MagicMock()
    mock_result.campaigns_to_pause = ["c-travel"]
    mock_result.campaigns_safe = ["c-tech"]
    mock_result.match_details = {
        "c-travel": MagicMock(campaign_name="Travel Deals", match_reasons=["category_match"]),
    }
    mock_matcher_cls.return_value.evaluate.return_value = mock_result

    state = _make_state(
        [
            {
                "topic": "earthquake",
                "is_safe": False,
                "safety_flags": [{"category": "natural_disaster"}],
            }
        ]
    )

    result = safety_gate(state)

    mock_mgr_cls.return_value.pause.assert_called_once_with("c-travel")
    mock_notifier_cls.return_value.send_safety_pause.assert_called_once()
    assert result["should_proceed"] is True


@patch("brand_conscience.common.notifications.SlackNotifier")
@patch("brand_conscience.layer4_deployment.campaign_manager.CampaignManager")
@patch("brand_conscience.models.safety.impact_matcher.SafetyImpactMatcher")
@patch("brand_conscience.db.queries.get_live_campaigns")
def test_safety_gate_all_paused_stops_pipeline(
    mock_live,
    mock_matcher_cls,
    mock_mgr_cls,
    mock_notifier_cls,
):
    """All campaigns matched → should_proceed becomes False."""
    campaign = SimpleNamespace(id="c1", name="Event Promo")
    mock_live.return_value = [campaign]

    mock_result = MagicMock()
    mock_result.campaigns_to_pause = ["c1"]
    mock_result.campaigns_safe = []
    mock_result.match_details = {
        "c1": MagicMock(campaign_name="Event Promo", match_reasons=["category_match"]),
    }
    mock_matcher_cls.return_value.evaluate.return_value = mock_result

    state = _make_state(
        [
            {
                "topic": "riots",
                "is_safe": False,
                "safety_flags": [{"category": "social_unrest"}],
            },
        ]
    )

    result = safety_gate(state)

    assert result["should_proceed"] is False


@patch("brand_conscience.db.queries.get_live_campaigns")
def test_safety_gate_no_live_campaigns_proceeds(mock_live):
    """Unsafe signals but no live campaigns → proceed (nothing to pause)."""
    mock_live.return_value = []

    state = _make_state(
        [
            {
                "topic": "crisis",
                "is_safe": False,
                "safety_flags": [{"category": "violence"}],
            },
        ]
    )

    result = safety_gate(state)

    assert result["should_proceed"] is True
