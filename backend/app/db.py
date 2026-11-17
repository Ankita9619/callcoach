"""Database engine, session factory and the FastAPI session dependency."""

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str) -> Engine:
    if database_url.startswith("sqlite"):
        kwargs: dict[str, object] = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in database_url:
            # One shared connection, otherwise each connection gets its own empty in-memory DB.
            kwargs["poolclass"] = StaticPool
        return create_engine(database_url, **kwargs)
    return create_engine(database_url, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db(request: Request) -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed afterwards.

    The session factory lives on `app.state` (set in `create_app`), so tests can build
    an app against a throwaway database without monkey-patching globals.
    """
    factory: sessionmaker[Session] = request.app.state.session_factory
    db = factory()
    try:
        yield db
    finally:
        db.close()
