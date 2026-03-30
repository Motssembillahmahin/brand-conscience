"""Tests for SafetyImpactMatcher — targeted campaign pausing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from brand_conscience.models.safety.impact_matcher import (
    SafetyImpactMatcher,
)


def _make_campaign(
    name: str,
    objective: str,
    campaign_id: str = "c1",
    config: dict | None = None,
    ad_sets: list | None = None,
) -> SimpleNamespace:
    """Create a fake campaign object for testing."""
    return SimpleNamespace(
        id=campaign_id,
        name=name,
        objective=objective,
        config=config,
        ad_sets=ad_sets or [],
    )


def _make_unsafe_signal(
    topic: str,
    safety_flags: list[dict],
    raw_data: dict | None = None,
) -> dict:
    return {
        "topic": topic,
        "is_safe": False,
        "safety_flags": safety_flags,
        "raw_data": raw_data,
    }


class TestCategoryMatching:
    """Test category-based matching between risk categories and campaign keywords."""

    def test_travel_campaign_matches_natural_disaster(self):
        matcher = SafetyImpactMatcher()
        campaign = _make_campaign("Summer Travel Deals", "awareness")
        signal = _make_unsafe_signal(
            topic="earthquake in Turkey",
            safety_flags=[{"category": "natural_disaster"}],
        )

        result = matcher.evaluate([signal], [campaign])

        assert len(result.campaigns_to_pause) == 1
        assert result.campaigns_to_pause[0] == "c1"
        assert any("category_match" in r for r in result.match_details["c1"].match_reasons)

    def test_electronics_campaign_no_match_natural_disaster(self):
        matcher = SafetyImpactMatcher()
        campaign = _make_campaign("Electronics Sale", "conversions")
        signal = _make_unsafe_signal(
            topic="earthquake in Turkey",
            safety_flags=[{"category": "natural_disaster"}],
        )

        result = matcher.evaluate([signal], [campaign])

        assert len(result.campaigns_to_pause) == 0
        assert "c1" in result.campaigns_safe

    def test_political_campaign_matches_political_controversy(self):
        matcher = SafetyImpactMatcher()
        campaign = _make_campaign("Policy Awareness Campaign", "awareness")
        signal = _make_unsafe_signal(
            topic="election scandal",
            safety_flags=[{"category": "political_controversy"}],
        )

        result = matcher.evaluate([signal], [campaign])

        assert "c1" in result.campaigns_to_pause

    def test_config_keywords_used_for_matching(self):
        matcher = SafetyImpactMatcher()
        campaign = _make_campaign(
            "Spring Collection",
            "conversions",
            config={"keywords": ["outdoor", "adventure"]},
        )
        signal = _make_unsafe_signal(
            topic="flooding",
            safety_flags=[{"category": "natural_disaster"}],
        )

        result = matcher.evaluate([signal], [campaign])

        assert "c1" in result.campaigns_to_pause


class TestAudienceMatching:
    """Test geographic/audience overlap matching."""

    def test_geo_overlap_triggers_match(self):
        matcher = SafetyImpactMatcher()
        ad_set = SimpleNamespace(
            targeting={"geo_locations": {"countries": ["US"], "regions": ["Texas"]}}
        )
        campaign = _make_campaign("Texas Promo", "awareness", ad_sets=[ad_set])
        signal = _make_unsafe_signal(
            topic="flooding in Texas",
            safety_flags=[{"category": "natural_disaster"}],
            raw_data={"geo": ["Texas"]},
        )

        result = matcher.evaluate([signal], [campaign])

        assert "c1" in result.campaigns_to_pause
        assert any("audience_match" in r for r in result.match_details["c1"].match_reasons)

    def test_no_geo_overlap_no_audience_match(self):
        matcher = SafetyImpactMatcher()
        ad_set = SimpleNamespace(
            targeting={"geo_locations": {"countries": ["US"], "regions": ["New York"]}}
        )
        campaign = _make_campaign("NY Promo", "awareness", ad_sets=[ad_set])
        signal = _make_unsafe_signal(
            topic="flooding in Texas",
            safety_flags=[{"category": "natural_disaster"}],
            raw_data={"geo": ["Texas"]},
        )

        result = matcher.evaluate([signal], [campaign])

        # No category match (name is "NY Promo"), no geo overlap
        assert "c1" in result.campaigns_safe

    def test_no_geo_in_signal_skips_audience_match(self):
        matcher = SafetyImpactMatcher()
        ad_set = SimpleNamespace(targeting={"geo_locations": {"countries": ["US"]}})
        campaign = _make_campaign("US Campaign", "conversions", ad_sets=[ad_set])
        signal = _make_unsafe_signal(
            topic="controversy",
            safety_flags=[{"category": "social_unrest"}],
        )

        result = matcher.evaluate([signal], [campaign])

        # No geo in signal → audience match returns 0
        # No category keywords in "US Campaign conversions" → no category match
        assert "c1" in result.campaigns_safe


class TestSemanticMatching:
    """Test CLIP-based semantic matching."""

    def test_semantic_match_above_threshold(self):
        mock_encoder = MagicMock()
        mock_encoder.encode_text.return_value = MagicMock()
        mock_encoder.cosine_similarity.return_value = MagicMock(
            max=MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.85)))
        )

        mock_safety = MagicMock()
        mock_safety._encoder = mock_encoder

        matcher = SafetyImpactMatcher(brand_safety=mock_safety, semantic_threshold=0.6)
        campaign = _make_campaign("Beach Vacation Packages", "awareness")
        signal = _make_unsafe_signal(
            topic="tsunami warning",
            safety_flags=[{"category": "natural_disaster"}],
        )

        result = matcher.evaluate([signal], [campaign])

        assert "c1" in result.campaigns_to_pause
        assert any("semantic_match" in r for r in result.match_details["c1"].match_reasons)

    def test_semantic_match_below_threshold(self):
        mock_encoder = MagicMock()
        mock_encoder.encode_text.return_value = MagicMock()
        mock_encoder.cosine_similarity.return_value = MagicMock(
            max=MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.3)))
        )

        mock_safety = MagicMock()
        mock_safety._encoder = mock_encoder

        matcher = SafetyImpactMatcher(brand_safety=mock_safety, semantic_threshold=0.6)
        campaign = _make_campaign("Electronics Sale", "conversions")
        signal = _make_unsafe_signal(
            topic="tsunami warning",
            safety_flags=[{"category": "natural_disaster"}],
        )

        result = matcher.evaluate([signal], [campaign])

        assert "c1" in result.campaigns_safe

    def test_no_brand_safety_skips_semantic(self):
        matcher = SafetyImpactMatcher(brand_safety=None)
        campaign = _make_campaign("Electronics Sale", "conversions")
        signal = _make_unsafe_signal(
            topic="tsunami warning",
            safety_flags=[{"category": "natural_disaster"}],
        )

        result = matcher.evaluate([signal], [campaign])

        # No semantic matcher, no category match → safe
        assert "c1" in result.campaigns_safe


class TestEdgeCases:
    """Test edge cases and empty inputs."""

    def test_no_unsafe_signals_all_safe(self):
        matcher = SafetyImpactMatcher()
        campaigns = [_make_campaign("Campaign A", "awareness")]

        result = matcher.evaluate([], campaigns)

        assert len(result.campaigns_safe) == 1
        assert len(result.campaigns_to_pause) == 0

    def test_no_live_campaigns(self):
        matcher = SafetyImpactMatcher()
        signal = _make_unsafe_signal(
            topic="crisis",
            safety_flags=[{"category": "violence"}],
        )

        result = matcher.evaluate([signal], [])

        assert len(result.campaigns_safe) == 0
        assert len(result.campaigns_to_pause) == 0

    def test_multiple_campaigns_mixed_results(self):
        matcher = SafetyImpactMatcher()
        travel_campaign = _make_campaign("Airline Deals", "awareness", campaign_id="c-travel")
        electronics_campaign = _make_campaign("Laptop Sale", "conversions", campaign_id="c-elec")
        hotel_campaign = _make_campaign("Hotel Bookings", "awareness", campaign_id="c-hotel")

        signal = _make_unsafe_signal(
            topic="earthquake",
            safety_flags=[{"category": "natural_disaster"}],
        )

        result = matcher.evaluate([signal], [travel_campaign, electronics_campaign, hotel_campaign])

        # airline + hotel match "natural_disaster" keywords; laptop doesn't
        assert "c-travel" in result.campaigns_to_pause
        assert "c-hotel" in result.campaigns_to_pause
        assert "c-elec" in result.campaigns_safe

    def test_multiple_signals_compound_matching(self):
        matcher = SafetyImpactMatcher()
        campaign = _make_campaign("Community Festival Event", "awareness")

        signals = [
            _make_unsafe_signal(
                topic="riot",
                safety_flags=[{"category": "social_unrest"}],
            ),
            _make_unsafe_signal(
                topic="shooting",
                safety_flags=[{"category": "violence"}],
            ),
        ]

        result = matcher.evaluate(signals, [campaign])

        # "Community Festival Event" matches social_unrest keywords (community, event, festival)
        assert "c1" in result.campaigns_to_pause

    def test_signal_without_safety_flags(self):
        matcher = SafetyImpactMatcher()
        campaign = _make_campaign("Travel Deals", "awareness")
        signal = {"topic": "unknown", "is_safe": False, "safety_flags": []}

        result = matcher.evaluate([signal], [campaign])

        # No risk categories extracted → no category match
        assert "c1" in result.campaigns_safe
