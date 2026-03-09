"""Slack notification sender."""

from __future__ import annotations

from typing import Any

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger

logger = get_logger(__name__)


class SlackNotifier:
    """Send notifications via Slack SDK."""

    def __init__(self, bot_token: str | None = None) -> None:
        settings = get_settings()
        self._token = bot_token or settings.slack.bot_token
        self._channel_ops = settings.slack.channel_ops
        self._channel_approvals = settings.slack.channel_approvals
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from slack_sdk import WebClient

            self._client = WebClient(token=self._token)
        return self._client

    def send_ops_alert(self, message: str, *, blocks: list[dict[str, Any]] | None = None) -> None:
        """Send an alert to the ops channel."""
        self._send(self._channel_ops, message, blocks=blocks)

    def send_approval_request(self, campaign_id: str, budget: float, rationale: str) -> None:
        """Send a campaign approval request to the approvals channel."""
        message = (
            f"Campaign `{campaign_id}` requires approval.\n"
            f"Daily budget: ${budget:,.2f}\n"
            f"Rationale: {rationale}\n"
            f"Reply with `/approve {campaign_id}` to approve."
        )
        self._send(self._channel_approvals, message)

    def send_daily_summary(self, summary: str) -> None:
        """Send a daily performance summary to the ops channel."""
        self._send(self._channel_ops, summary)

    def _send(
        self,
        channel: str,
        text: str,
        *,
        blocks: list[dict[str, Any]] | None = None,
    ) -> None:
        """Send a message to a Slack channel."""
        if not self._token:
            logger.warning("slack_not_configured", msg="No bot token, skipping notification")
            return
        try:
            client = self._get_client()
            client.chat_postMessage(channel=channel, text=text, blocks=blocks)
            logger.info("slack_message_sent", channel=channel)
        except Exception as exc:
            logger.error("slack_send_failed", channel=channel, error=str(exc))
