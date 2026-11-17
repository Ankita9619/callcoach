"""Load the sample transcripts in `sample_data/` into an empty database.

Each sample file starts with optional header lines:
    # Title: Billing dispute – refund
    # Agent: Sarah
followed by the transcript itself.

Run manually with `python -m app.seed`; it also runs on startup when SEED_SAMPLE_DATA=true.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import crud
from app.models import Call, CallSource
from app.scoring.service import ScoringService

logger = logging.getLogger(__name__)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


@dataclass
class Sample:
    title: str
    agent_name: str | None
    transcript: str


def parse_sample(path: Path) -> Sample:
    title, agent = path.stem.replace("_", " "), None
    body: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("# title:"):
            title = stripped.split(":", 1)[1].strip()
        elif stripped.lower().startswith("# agent:"):
            agent = stripped.split(":", 1)[1].strip() or None
        else:
            body.append(line)
    return Sample(title=title, agent_name=agent, transcript="\n".join(body).strip())


def load_samples(directory: Path = SAMPLE_DIR) -> list[Sample]:
    return [parse_sample(p) for p in sorted(directory.glob("*.txt"))]


def seed_if_empty(db: Session, scoring: ScoringService | None = None) -> int:
    """Insert samples if the calls table is empty. Scores them too if `scoring` is given.

    Returns the number of calls inserted.
    """
    if db.scalar(select(func.count()).select_from(Call)):
        return 0
    samples = load_samples()
    for sample in samples:
        call = crud.create_call(
            db,
            title=sample.title,
            transcript=sample.transcript,
            agent_name=sample.agent_name,
            source=CallSource.SAMPLE,
        )
        if scoring is not None:
            crud.score_call(db, call, scoring)
    logger.info("Seeded %d sample calls", len(samples))
    return len(samples)


if __name__ == "__main__":
    from app.config import get_settings
    from app.db import Base, make_engine, make_session_factory
    from app.scoring.service import build_scorer

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    service = ScoringService(build_scorer(settings), settings.llm_max_retries)
    with make_session_factory(engine)() as session:
        print(f"Inserted {seed_if_empty(session, service)} sample calls")
