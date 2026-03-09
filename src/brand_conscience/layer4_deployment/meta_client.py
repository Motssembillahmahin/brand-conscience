"""Meta Marketing API client for campaign CRUD operations."""

from __future__ import annotations

from typing import Any

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced

logger = get_logger(__name__)


class MetaClient:
    """CRUD operations against the Meta Marketing API."""

    def __init__(
        self,
        access_token: str | None = None,
        ad_account_id: str | None = None,
    ) -> None:
        settings = get_settings()
        self._token = access_token or settings.meta.access_token
        self._account_id = ad_account_id or settings.meta.ad_account_id
        self._api: Any = None

    def _init_api(self) -> None:
        if self._api is not None:
            return
        from facebook_business.api import FacebookAdsApi
        from facebook_business.adobjects.adaccount import AdAccount

        FacebookAdsApi.init(access_token=self._token)
        self._api = AdAccount(f"act_{self._account_id}")
        logger.info("meta_api_initialized", account=self._account_id)

    @traced(name="meta_create_campaign", tags=["layer4", "meta"])
    def create_campaign(
        self,
        name: str,
        objective: str = "OUTCOME_SALES",
        daily_budget: float = 0.0,
        status: str = "PAUSED",
    ) -> str:
        """Create a campaign on Meta.

        Returns:
            Meta campaign ID.
        """
        self._init_api()
        params = {
            "name": name,
            "objective": objective,
            "status": status,
            "special_ad_categories": [],
        }
        if daily_budget > 0:
            params["daily_budget"] = int(daily_budget * 100)  # cents

        result = self._api.create_campaign(params=params)
        campaign_id = result["id"]
        logger.info("meta_campaign_created", meta_id=campaign_id, name=name)
        return campaign_id

    @traced(name="meta_create_adset", tags=["layer4", "meta"])
    def create_adset(
        self,
        campaign_id: str,
        name: str,
        daily_budget: float,
        targeting: dict,
        bid_amount: float | None = None,
    ) -> str:
        """Create an ad set under a campaign.

        Returns:
            Meta ad set ID.
        """
        self._init_api()
        params = {
            "campaign_id": campaign_id,
            "name": name,
            "daily_budget": int(daily_budget * 100),
            "targeting": targeting,
            "billing_event": "IMPRESSIONS",
            "optimization_goal": "OFFSITE_CONVERSIONS",
            "status": "PAUSED",
        }
        if bid_amount:
            params["bid_amount"] = int(bid_amount * 100)

        result = self._api.create_ad_set(params=params)
        adset_id = result["id"]
        logger.info("meta_adset_created", meta_id=adset_id, campaign=campaign_id)
        return adset_id

    @traced(name="meta_create_ad", tags=["layer4", "meta"])
    def create_ad(
        self,
        adset_id: str,
        name: str,
        creative_id: str,
    ) -> str:
        """Create an ad with a creative.

        Returns:
            Meta ad ID.
        """
        self._init_api()
        params = {
            "adset_id": adset_id,
            "name": name,
            "creative": {"creative_id": creative_id},
            "status": "PAUSED",
        }
        result = self._api.create_ad(params=params)
        ad_id = result["id"]
        logger.info("meta_ad_created", meta_id=ad_id, adset=adset_id)
        return ad_id

    @traced(name="meta_update_status", tags=["layer4", "meta"])
    def update_campaign_status(self, campaign_id: str, status: str) -> None:
        """Update a campaign's status (ACTIVE, PAUSED, etc.)."""
        self._init_api()
        from facebook_business.adobjects.campaign import Campaign

        campaign = Campaign(campaign_id)
        campaign.api_update(params={"status": status})
        logger.info("meta_status_updated", campaign=campaign_id, status=status)

    @traced(name="meta_update_bid", tags=["layer4", "meta"])
    def update_adset_bid(self, adset_id: str, bid_amount: float) -> None:
        """Update an ad set's bid amount."""
        self._init_api()
        from facebook_business.adobjects.adset import AdSet

        adset = AdSet(adset_id)
        adset.api_update(params={"bid_amount": int(bid_amount * 100)})
        logger.info("meta_bid_updated", adset=adset_id, bid=bid_amount)

    @traced(name="meta_get_insights", tags=["layer4", "meta"])
    def get_campaign_insights(
        self, campaign_id: str, fields: list[str] | None = None
    ) -> dict:
        """Get campaign performance insights."""
        self._init_api()
        from facebook_business.adobjects.campaign import Campaign

        fields = fields or [
            "impressions", "clicks", "spend", "conversions", "actions",
        ]
        campaign = Campaign(campaign_id)
        insights = campaign.get_insights(fields=fields)
        return dict(insights[0]) if insights else {}

    @traced(name="meta_pause_all", tags=["layer4", "meta"])
    def pause_all_campaigns(self) -> list[str]:
        """Pause all active campaigns in the ad account.

        Returns:
            List of paused campaign IDs.
        """
        self._init_api()
        campaigns = self._api.get_campaigns(
            fields=["id", "status"],
            params={"effective_status": ["ACTIVE"]},
        )
        paused = []
        for c in campaigns:
            self.update_campaign_status(c["id"], "PAUSED")
            paused.append(c["id"])
        logger.warning("meta_all_campaigns_paused", count=len(paused))
        return paused
