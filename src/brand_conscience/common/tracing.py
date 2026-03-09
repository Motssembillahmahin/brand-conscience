"""OPIK tracing integration."""

from __future__ import annotations

import functools
from typing import Any, TypeVar

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F")


def init_opik() -> None:
    """Initialize OPIK tracing client."""
    try:
        import opik

        settings = get_settings()
        opik.configure(
            api_key=settings.opik.api_key or None,
            url=settings.opik.url,
            project_name=settings.opik.project_name,
        )
        logger.info("opik_initialized", url=settings.opik.url)
    except ImportError:
        logger.warning("opik_not_installed", msg="OPIK tracing disabled")
    except Exception as exc:
        logger.warning("opik_init_failed", error=str(exc))


def traced(
    name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Decorator to trace a function with OPIK.

    Falls back to a no-op if OPIK is not available.
    """
    def decorator(func: Any) -> Any:
        try:
            import opik

            return opik.track(
                name=name or func.__name__,
                tags=tags,
                metadata=metadata,
            )(func)
        except ImportError:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)
            return wrapper

    return decorator


def get_trace_headers() -> dict[str, str]:
    """Get distributed trace headers for cross-service propagation."""
    try:
        from opik import opik_context

        return opik_context.get_distributed_trace_headers()
    except (ImportError, Exception):
        return {}
