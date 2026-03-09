"""Test factories using factory-boy."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import factory

from brand_conscience.common.types import ActionType, CampaignStatus, QualityTier
from brand_conscience.db.tables import (
    AdSet,
    Campaign,
    Creative,
    MomentSnapshot,
    PerformanceMetric,
)


class MomentSnapshotFactory(factory.Factory):
    class Meta:
        model = MomentSnapshot

    id = factory.LazyFunction(uuid.uuid4)
    timestamp = factory.LazyFunction(lambda: datetime.now(UTC))
    urgency_score = factory.Faker("pyfloat", min_value=0.0, max_value=1.0)
    recommended_action = ActionType.LAUNCH.value
    business_signals = factory.LazyFunction(lambda: [])
    cultural_signals = factory.LazyFunction(lambda: [])
    creative_signals = factory.LazyFunction(lambda: [])
    affected_categories = factory.LazyFunction(lambda: ["general"])
    affected_audiences = factory.LazyFunction(lambda: ["broad_interest"])
    context_summary = factory.Faker("sentence")


class CampaignFactory(factory.Factory):
    class Meta:
        model = Campaign

    id = factory.LazyFunction(uuid.uuid4)
    name = factory.Sequence(lambda n: f"Campaign {n}")
    status = CampaignStatus.DRAFT
    objective = "conversions"
    daily_budget = 500.0
    total_spend = 0.0
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))
    updated_at = factory.LazyFunction(lambda: datetime.now(UTC))


class AdSetFactory(factory.Factory):
    class Meta:
        model = AdSet

    id = factory.LazyFunction(uuid.uuid4)
    name = factory.Sequence(lambda n: f"AdSet {n}")
    audience_segment = "broad_interest"
    daily_budget = 100.0
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))


class CreativeFactory(factory.Factory):
    class Meta:
        model = Creative

    id = factory.LazyFunction(uuid.uuid4)
    prompt_text = factory.Faker("sentence")
    prompt_score = 0.85
    image_path = factory.Sequence(lambda n: f"/creatives/image-{n}.png")
    quality_tier = QualityTier.EXCELLENT
    brand_alignment_score = 0.75
    originality_score = 0.6
    predicted_ctr = 0.03
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))


class PerformanceMetricFactory(factory.Factory):
    class Meta:
        model = PerformanceMetric

    id = factory.LazyFunction(uuid.uuid4)
    timestamp = factory.LazyFunction(lambda: datetime.now(UTC))
    impressions = factory.Faker("random_int", min=100, max=10000)
    clicks = factory.Faker("random_int", min=1, max=500)
    spend = factory.Faker("pyfloat", min_value=10.0, max_value=500.0)
    conversions = factory.Faker("random_int", min=0, max=50)
    revenue = factory.Faker("pyfloat", min_value=0.0, max_value=2000.0)
