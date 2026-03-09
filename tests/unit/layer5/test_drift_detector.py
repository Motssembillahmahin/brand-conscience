"""Tests for drift detector."""

from __future__ import annotations

import numpy as np

from brand_conscience.common.types import DriftSeverity
from brand_conscience.layer5_feedback.drift_detector import DriftDetector


def test_no_drift_same_distribution():
    detector = DriftDetector()
    reference = np.random.normal(0, 1, 1000)
    current = np.random.normal(0, 1, 1000)
    severity, psi = detector.check_drift(reference, current)
    assert severity in (DriftSeverity.NONE, DriftSeverity.LOW)
    assert psi < 0.2


def test_drift_different_distribution():
    detector = DriftDetector()
    reference = np.random.normal(0, 1, 1000)
    current = np.random.normal(3, 1, 1000)  # shifted mean
    severity, psi = detector.check_drift(reference, current)
    assert severity in (DriftSeverity.MEDIUM, DriftSeverity.HIGH)
    assert psi > 0.2


def test_should_retrain():
    detector = DriftDetector(psi_threshold=0.2)
    assert detector.should_retrain(0.3)
    assert not detector.should_retrain(0.1)


def test_psi_symmetric_ish():
    detector = DriftDetector()
    a = np.random.normal(0, 1, 1000)
    b = np.random.normal(2, 1, 1000)
    psi_ab = detector.compute_psi(a, b)
    psi_ba = detector.compute_psi(b, a)
    # PSI is approximately symmetric
    assert abs(psi_ab - psi_ba) < psi_ab * 0.5
