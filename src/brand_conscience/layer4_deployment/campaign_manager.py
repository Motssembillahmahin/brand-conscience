"""Campaign state machine and lifecycle management."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from brand_conscience.common.config import get_settings
from brand_conscience.common.database import get_session
from brand_conscience.common.exceptions import ApprovalRequiredError, InvalidTransitionError
from brand_conscience.common.logging import get_logger
from brand_conscience.common.notifications import SlackNotifier
from brand_conscience.common.tracing import traced
from brand_conscience.common.types import CampaignStatus, validate_campaign_transition
from brand_conscience.db.tables import Campaign

logger = get_logger(__name__)


class CampaignManager:
    """Manage campaign lifecycle state machine.

    DRAFT → PENDING_APPROVAL → LIVE → PAUSED → COMPLETED
    """

    def __init__(self, notifier: SlackNotifier | None = None) -> None:
        self._notifier = notifier or SlackNotifier()

    @traced(name="create_campaign", tags=["layer4", "campaign"])
    def create(
        self,
        name: str,
        objective: str,
        daily_budget: float,
        moment_profile_id: str | None = None,
        config: dict | None = None,
    ) -> str:
        """Create a new campaign in DRAFT status.

        Returns:
            Campaign ID.
        """
        campaign_id = str(uuid.uuid4())

        with get_session() as session:
            campaign = Campaign(
                id=uuid.UUID(campaign_id),
                name=name,
                status=CampaignStatus.DRAFT,
                objective=objective,
                daily_budget=daily_budget,
                moment_profile_id=(
                    uuid.UUID(moment_profile_id) if moment_profile_id else None
                ),
                config=config,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            session.add(campaign)

        logger.info("campaign_created", campaign_id=campaign_id, name=name)
        return campaign_id

    @traced(name="transition_campaign", tags=["layer4", "campaign"])
    def transition(self, campaign_id: str, target_status: CampaignStatus) -> None:
        """Transition a campaign to a new status.

        Raises:
            InvalidTransitionError: If the transition is not valid.
            ApprovalRequiredError: If the campaign needs approval.
        """
        with get_session() as session:
            campaign = session.get(Campaign, uuid.UUID(campaign_id))
            if campaign is None:
                raise ValueError(f"Campaign {campaign_id} not found")

            if not validate_campaign_transition(campaign.status, target_status):
                raise InvalidTransitionError(
                    campaign.status.value, target_status.value
                )

            # Check approval requirement
            if target_status == CampaignStatus.LIVE:
                settings = get_settings()
                if campaign.daily_budget > settings.deployment.spend_approval_threshold:
                    if campaign.status != CampaignStatus.PENDING_APPROVAL:
                        campaign.status = CampaignStatus.PENDING_APPROVAL
                        campaign.updated_at = datetime.now(UTC)
                        self._notifier.send_approval_request(
                            campaign_id=campaign_id,
                            budget=campaign.daily_budget,
                            rationale=f"Objective: {campaign.objective}",
                        )
                        raise ApprovalRequiredError(
                            f"Campaign {campaign_id} requires approval (budget=${campaign.daily_budget:,.2f})"
                        )

            campaign.status = target_status
            campaign.updated_at = datetime.now(UTC)

        logger.info(
            "campaign_transitioned",
            campaign_id=campaign_id,
            new_status=target_status.value,
        )

    @traced(name="submit_for_deployment", tags=["layer4", "campaign"])
    def submit_for_deployment(self, campaign_id: str) -> CampaignStatus:
        """Submit a DRAFT campaign for deployment.

        Auto-approves if below spend threshold, otherwise sends approval request.

        Returns:
            Resulting campaign status.
        """
        settings = get_settings()

        with get_session() as session:
            campaign = session.get(Campaign, uuid.UUID(campaign_id))
            if campaign is None:
                raise ValueError(f"Campaign {campaign_id} not found")

            if campaign.daily_budget <= settings.deployment.spend_approval_threshold:
                # Auto-approve
                self.transition(campaign_id, CampaignStatus.PENDING_APPROVAL)
                self.transition(campaign_id, CampaignStatus.LIVE)
                return CampaignStatus.LIVE
            else:
                self.transition(campaign_id, CampaignStatus.PENDING_APPROVAL)
                return CampaignStatus.PENDING_APPROVAL

    def approve(self, campaign_id: str) -> None:
        """Approve a pending campaign (from Slack command or CLI)."""
        self.transition(campaign_id, CampaignStatus.LIVE)
        logger.info("campaign_approved", campaign_id=campaign_id)

    def pause(self, campaign_id: str) -> None:
        """Pause a live campaign."""
        self.transition(campaign_id, CampaignStatus.PAUSED)

    def complete(self, campaign_id: str) -> None:
        """Mark a campaign as completed."""
        self.transition(campaign_id, CampaignStatus.COMPLETED)
