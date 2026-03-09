"""SQLAlchemy database engine and session management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

if TYPE_CHECKING:
    from collections.abc import Generator

    from brand_conscience.common.config import Settings


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""


_engine: Any = None
_session_factory: sessionmaker[Session] | None = None


def init_database(settings: Settings) -> None:
    """Initialize the database engine and session factory."""
    global _engine, _session_factory
    _engine = create_engine(
        settings.database.url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        echo=settings.database.echo,
    )
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)


def get_engine() -> Any:
    """Return the current SQLAlchemy engine."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the current session factory."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _session_factory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional database session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
