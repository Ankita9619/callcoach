"""FastAPI application factory.

`create_app()` wires settings -> database -> scorer -> routes. Tests call it with their own
Settings (in-memory SQLite, fake scorers) instead of patching globals. The module-level
`app` that uvicorn serves lives in `app/main.py`, so importing this module has no side effects.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import calls, meta
from app.config import Settings, get_settings
from app.db import Base, make_engine, make_session_factory
from app.scoring.base import Scorer
from app.scoring.mock import MockScorer
from app.scoring.service import ScoringService, build_scorer
from app.seed import seed_if_empty


def create_app(settings: Settings | None = None, scorer: Scorer | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="CallCoach API",
        version="0.1.0",
        description="LLM-based QA scoring for customer-service calls.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    engine = make_engine(settings.database_url)
    # Fine for a demo; a production service would use Alembic migrations instead.
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(engine)
    scoring_service = ScoringService(
        scorer or build_scorer(settings), max_retries=settings.llm_max_retries
    )

    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.scoring_service = scoring_service

    if settings.seed_sample_data:
        # Score samples on startup only in mock mode: it's instant and free. With a real
        # LLM, samples are stored unscored and can be scored from the UI/API on demand.
        seed_scoring = scoring_service if isinstance(scoring_service.primary, MockScorer) else None
        with session_factory() as db:
            seed_if_empty(db, seed_scoring)

    app.include_router(meta.router)
    app.include_router(calls.router)
    return app
