"""Health check and rubric endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_scoring_service
from app.db import get_db
from app.schemas import HealthOut, RubricCategoryOut
from app.scoring.rubric import RUBRIC
from app.scoring.service import ScoringService

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthOut)
def health(
    db: Session = Depends(get_db),
    scoring: ScoringService = Depends(get_scoring_service),
) -> HealthOut:
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except Exception:
        database = "error"
    return HealthOut(
        status="ok" if database == "ok" else "degraded",
        database=database,
        llm_provider=scoring.primary.name,
        llm_model=scoring.primary.model,
    )


@router.get("/api/rubric", response_model=list[RubricCategoryOut])
def rubric() -> list[RubricCategoryOut]:
    return [
        RubricCategoryOut(key=c.key, name=c.name, weight=c.weight, description=c.description)
        for c in RUBRIC
    ]
