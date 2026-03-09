"""Tests for creative evaluation gates (unit tests without CLIP model)."""

from __future__ import annotations

from brand_conscience.common.types import GateResult, QualityTier
from brand_conscience.models.quality_classifier.inference import QualityClassifier


def test_quality_tier_passes_gate():
    classifier = QualityClassifier()
    assert classifier.passes_gate(QualityTier.EXCELLENT)
    assert classifier.passes_gate(QualityTier.GOOD)
    assert not classifier.passes_gate(QualityTier.ACCEPTABLE)
    assert not classifier.passes_gate(QualityTier.REJECT)


def test_gate_result_enum():
    assert GateResult.PASSED.value == "passed"
    assert GateResult.REJECTED.value == "rejected"


def test_quality_tier_enum():
    assert QualityTier.EXCELLENT.value == "excellent"
    assert QualityTier.REJECT.value == "reject"
