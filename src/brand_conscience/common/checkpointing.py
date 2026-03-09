"""LangGraph PostgreSQL checkpoint management."""

from __future__ import annotations

from typing import Any

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger

logger = get_logger(__name__)


def get_checkpoint_saver() -> Any:
    """Create a LangGraph PostgreSQL checkpoint saver.

    Returns a PostgresSaver instance configured with the application database.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        settings = get_settings()
        saver = PostgresSaver.from_conn_string(settings.database.url)
        saver.setup()
        logger.info("checkpoint_saver_initialized")
        return saver
    except ImportError:
        logger.warning(
            "langgraph_checkpoint_not_installed",
            msg="LangGraph checkpointing disabled",
        )
        return None
    except Exception as exc:
        logger.error("checkpoint_saver_failed", error=str(exc))
        raise
