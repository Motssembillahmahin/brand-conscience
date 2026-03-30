"""Safety impact matcher — match unsafe signals to specific campaigns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced

if TYPE_CHECKING:
    from brand_conscience.models.safety.brand_safety import BrandSafetyClassifier

logger = get_logger(__name__)

# Keywords associated with each risk category for campaign matching
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "political_controversy": ["political", "politics", "election", "government", "policy"],
    "natural_disaster": [
        "travel",
        "tourism",
        "outdoor",
        "adventure",
        "airline",
        "hotel",
        "resort",
        "flight",
    ],
    "social_unrest": ["social", "community", "public", "event", "gathering", "festival"],
    "violence": ["weapons", "combat", "tactical", "military", "hunting", "defense"],
}


@dataclass
class MatchResult:
    """Result of matching a single campaign against unsafe signals."""

    campaign_id: str
    campaign_name: str
    is_affected: bool
    match_reasons: list[str] = field(default_factory=list)
    match_score: float = 0.0


@dataclass
class SafetyImpactResult:
    """Aggregated result of matching all live campaigns against unsafe signals."""

    campaigns_to_pause: list[str] = field(default_factory=list)
    campaigns_safe: list[str] = field(default_factory=list)
    match_details: dict[str, MatchResult] = field(default_factory=dict)


class SafetyImpactMatcher:
    """Match unsafe cultural signals to specific campaigns for targeted pausing.

    Uses three matching strategies:
    1. Category matching — signal risk categories vs campaign objective/name keywords
    2. Semantic matching — CLIP similarity between signal topic and campaign creatives
    3. Audience matching — geographic/demographic overlap
    """

    def __init__(
        self,
        brand_safety: BrandSafetyClassifier | None = None,
        semantic_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self._brand_safety = brand_safety
        self._semantic_threshold = (
            semantic_threshold or settings.safety.brand_safety.similarity_threshold
        )

    @traced(name="safety_impact_evaluate", tags=["safety", "impact_matcher"])
    def evaluate(
        self,
        unsafe_signals: list[dict],
        live_campaigns: list,
    ) -> SafetyImpactResult:
        """Evaluate which live campaigns are affected by unsafe cultural signals.

        Args:
            unsafe_signals: Cultural signals where is_safe=False. Each has
                'topic', 'safety_flags', and optionally 'raw_data' with geo info.
            live_campaigns: Campaign ORM objects with status=LIVE.

        Returns:
            SafetyImpactResult with campaigns to pause and safe campaigns.
        """
        result = SafetyImpactResult()

        if not unsafe_signals or not live_campaigns:
            result.campaigns_safe = [str(c.id) for c in live_campaigns]
            return result

        # Extract risk categories from all unsafe signals
        risk_categories = self._extract_risk_categories(unsafe_signals)
        signal_topics = [s.get("topic", "") for s in unsafe_signals]

        for campaign in live_campaigns:
            match = self._match_campaign(
                campaign=campaign,
                risk_categories=risk_categories,
                signal_topics=signal_topics,
                unsafe_signals=unsafe_signals,
            )
            result.match_details[match.campaign_id] = match

            if match.is_affected:
                result.campaigns_to_pause.append(match.campaign_id)
            else:
                result.campaigns_safe.append(match.campaign_id)

        logger.info(
            "safety_impact_evaluated",
            total_campaigns=len(live_campaigns),
            to_pause=len(result.campaigns_to_pause),
            safe=len(result.campaigns_safe),
            risk_categories=risk_categories,
        )
        return result

    def _match_campaign(
        self,
        campaign,
        risk_categories: list[str],
        signal_topics: list[str],
        unsafe_signals: list[dict],
    ) -> MatchResult:
        """Match a single campaign against all unsafe signals."""
        campaign_id = str(campaign.id)
        campaign_name = campaign.name
        reasons: list[str] = []
        scores: list[float] = []

        # Strategy 1: Category matching
        cat_score = self._category_match(campaign, risk_categories)
        if cat_score > 0:
            reasons.append(f"category_match(score={cat_score:.2f})")
            scores.append(cat_score)

        # Strategy 2: Semantic matching (CLIP-based)
        sem_score = self._semantic_match(campaign, signal_topics)
        if sem_score > self._semantic_threshold:
            reasons.append(f"semantic_match(score={sem_score:.2f})")
            scores.append(sem_score)

        # Strategy 3: Audience/geo matching
        geo_score = self._audience_match(campaign, unsafe_signals)
        if geo_score > 0:
            reasons.append(f"audience_match(score={geo_score:.2f})")
            scores.append(geo_score)

        total_score = max(scores) if scores else 0.0
        is_affected = len(reasons) > 0

        if is_affected:
            logger.info(
                "campaign_safety_matched",
                campaign_id=campaign_id,
                campaign_name=campaign_name,
                reasons=reasons,
                score=total_score,
            )

        return MatchResult(
            campaign_id=campaign_id,
            campaign_name=campaign_name,
            is_affected=is_affected,
            match_reasons=reasons,
            match_score=total_score,
        )

    def _category_match(self, campaign, risk_categories: list[str]) -> float:
        """Match campaign objective/name against risk category keywords.

        Returns score 0.0-1.0 based on keyword overlap.
        """
        # Build searchable text from campaign fields
        campaign_text = f"{campaign.name} {campaign.objective}".lower()
        config = campaign.config or {}
        if config.get("category"):
            campaign_text += f" {config['category']}"
        if config.get("keywords"):
            campaign_text += f" {' '.join(config['keywords'])}"

        matches = 0
        total_keywords = 0

        for category in risk_categories:
            keywords = CATEGORY_KEYWORDS.get(category, [])
            total_keywords += len(keywords)
            for keyword in keywords:
                if keyword in campaign_text:
                    matches += 1

        if total_keywords == 0:
            return 0.0

        return min(matches / max(total_keywords * 0.3, 1), 1.0)

    def _semantic_match(self, campaign, signal_topics: list[str]) -> float:
        """Use CLIP to compute similarity between signal topics and campaign text.

        Returns max cosine similarity score.
        """
        if not self._brand_safety or not signal_topics:
            return 0.0

        # Build campaign text from name + objective + creative prompts if available
        campaign_text = f"{campaign.name} {campaign.objective}"

        try:
            encoder = self._brand_safety._encoder
            topic_embeddings = encoder.encode_text(signal_topics)
            campaign_embedding = encoder.encode_text([campaign_text])
            similarities = encoder.cosine_similarity(campaign_embedding, topic_embeddings)

            max_sim = float(similarities.max().item())
            return max_sim
        except Exception as exc:
            logger.warning("semantic_match_failed", error=str(exc))
            return 0.0

    def _audience_match(self, campaign, unsafe_signals: list[dict]) -> float:
        """Match campaign targeting against signal geographic/demographic context.

        Returns score 0.0-1.0 based on audience overlap.
        """
        # Extract geo context from signals
        signal_geos: set[str] = set()
        for signal in unsafe_signals:
            raw = signal.get("raw_data") or {}
            if geo := raw.get("geo"):
                if isinstance(geo, list):
                    signal_geos.update(g.lower() for g in geo)
                elif isinstance(geo, str):
                    signal_geos.add(geo.lower())

        if not signal_geos:
            return 0.0

        # Check campaign ad_sets for targeting overlap
        campaign_geos: set[str] = set()
        for ad_set in getattr(campaign, "ad_sets", []):
            targeting = ad_set.targeting or {}
            geo_locations = targeting.get("geo_locations", {})
            countries = geo_locations.get("countries", [])
            regions = geo_locations.get("regions", [])
            cities = geo_locations.get("cities", [])
            campaign_geos.update(c.lower() for c in countries)
            campaign_geos.update(r.lower() for r in regions)
            campaign_geos.update(c.lower() for c in cities)

        if not campaign_geos:
            return 0.0

        overlap = signal_geos & campaign_geos
        if overlap:
            return len(overlap) / len(signal_geos)

        return 0.0

    def _extract_risk_categories(self, unsafe_signals: list[dict]) -> list[str]:
        """Extract unique risk category names from unsafe signals."""
        categories: set[str] = set()
        for signal in unsafe_signals:
            for flag in signal.get("safety_flags", []):
                if cat := flag.get("category"):
                    categories.add(cat)
        return sorted(categories)
