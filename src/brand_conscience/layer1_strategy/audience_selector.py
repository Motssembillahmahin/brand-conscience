"""Audience segment catalog and selection."""

from __future__ import annotations

from dataclasses import dataclass, field

from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced

logger = get_logger(__name__)


@dataclass
class AudienceSegment:
    """An audience segment with targeting parameters."""

    name: str
    description: str = ""
    estimated_size: int = 0
    targeting: dict = field(default_factory=dict)
    historical_ctr: float = 0.0
    historical_roas: float = 0.0


class AudienceSelector:
    """Select and configure audience segments for campaigns."""

    DEFAULT_SEGMENTS: dict[str, AudienceSegment] = {
        "broad_interest": AudienceSegment(
            name="broad_interest",
            description="Broad interest-based targeting",
            estimated_size=1_000_000,
            targeting={"interest_categories": ["general"]},
        ),
        "retargeting": AudienceSegment(
            name="retargeting",
            description="Website visitors and past customers",
            estimated_size=50_000,
            targeting={"custom_audience_type": "website_visitors"},
        ),
        "lookalike": AudienceSegment(
            name="lookalike",
            description="Lookalike of best customers",
            estimated_size=500_000,
            targeting={"lookalike_source": "high_value_customers", "lookalike_pct": 1},
        ),
        "custom_audience": AudienceSegment(
            name="custom_audience",
            description="Custom audience from CRM data",
            estimated_size=100_000,
            targeting={"custom_audience_type": "crm_upload"},
        ),
    }

    def __init__(self) -> None:
        self._segments = dict(self.DEFAULT_SEGMENTS)

    @traced(name="select_audience", tags=["layer1", "audience"])
    def select(self, segment_name: str) -> AudienceSegment:
        """Select an audience segment by name.

        Args:
            segment_name: Segment identifier.

        Returns:
            AudienceSegment instance.
        """
        segment = self._segments.get(segment_name)
        if segment is None:
            logger.warning("unknown_segment", name=segment_name, fallback="broad_interest")
            segment = self._segments["broad_interest"]

        logger.info(
            "audience_selected",
            segment=segment.name,
            estimated_size=segment.estimated_size,
        )
        return segment

    def get_available_segments(self) -> list[str]:
        """Return list of available segment names."""
        return list(self._segments.keys())

    def get_meta_targeting(self, segment: AudienceSegment) -> dict:
        """Convert segment to Meta API targeting spec."""
        # TODO: translate to actual Meta targeting format
        return {
            "geo_locations": {"countries": ["US"]},
            "age_min": 18,
            "age_max": 65,
            **segment.targeting,
        }
