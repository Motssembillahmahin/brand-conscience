"""Exception hierarchy for Brand Conscience."""

from __future__ import annotations


class BrandConscienceError(Exception):
    """Base exception for all Brand Conscience errors."""


# Configuration
class ConfigurationError(BrandConscienceError):
    """Invalid or missing configuration."""


# Database
class DatabaseError(BrandConscienceError):
    """Database operation failed."""


class RecordNotFoundError(DatabaseError):
    """Requested record does not exist."""


# Campaign
class CampaignError(BrandConscienceError):
    """Campaign operation failed."""


class InvalidTransitionError(CampaignError):
    """Invalid campaign state transition."""

    def __init__(self, current: str, target: str) -> None:
        super().__init__(f"Cannot transition from {current} to {target}")
        self.current = current
        self.target = target


class ApprovalRequiredError(CampaignError):
    """Campaign requires human approval before proceeding."""


# Safety
class SafetyError(BrandConscienceError):
    """Safety constraint violation."""


class CircuitBreakerTrippedError(SafetyError):
    """Circuit breaker has been triggered."""

    def __init__(self, reason: str, campaigns_paused: list[str] | None = None) -> None:
        super().__init__(f"Circuit breaker tripped: {reason}")
        self.reason = reason
        self.campaigns_paused = campaigns_paused or []


class BrandSafetyViolationError(SafetyError):
    """Content flagged by brand safety classifier."""


class BidCapExceededError(SafetyError):
    """Bid exceeds maximum allowed cap."""


# Creative
class CreativeError(BrandConscienceError):
    """Creative generation or evaluation failed."""


class GateRejectionError(CreativeError):
    """Creative rejected by evaluation gate."""

    def __init__(self, gate_name: str, score: float, threshold: float) -> None:
        super().__init__(
            f"Rejected by {gate_name}: score={score:.3f}, threshold={threshold:.3f}"
        )
        self.gate_name = gate_name
        self.score = score
        self.threshold = threshold


class DiversityViolationError(CreativeError):
    """Creative too similar to existing active creatives."""


# External API
class ExternalAPIError(BrandConscienceError):
    """External API call failed."""


class MetaAPIError(ExternalAPIError):
    """Meta Marketing API error."""


class GeminiAPIError(ExternalAPIError):
    """Google Gemini API error."""


class SlackAPIError(ExternalAPIError):
    """Slack API error."""


# ML Models
class ModelError(BrandConscienceError):
    """ML model error."""


class ModelNotLoadedError(ModelError):
    """Model checkpoint not found or not loaded."""


class DriftDetectedError(ModelError):
    """Model drift detected above threshold."""

    def __init__(self, model_name: str, psi_score: float, threshold: float) -> None:
        super().__init__(
            f"Drift detected in {model_name}: PSI={psi_score:.3f} > {threshold:.3f}"
        )
        self.model_name = model_name
        self.psi_score = psi_score
        self.threshold = threshold
