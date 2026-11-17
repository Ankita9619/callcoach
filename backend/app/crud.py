"""Database operations for calls, kept out of the route handlers so they're reusable
(seeding, background jobs later) and easy to test."""

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Call, CallSource, CallStatus
from app.scoring.service import ScoringService


def create_call(
    db: Session,
    *,
    title: str,
    transcript: str,
    agent_name: str | None = None,
    source: CallSource = CallSource.PASTE,
) -> Call:
    call = Call(title=title, transcript=transcript, agent_name=agent_name, source=source)
    db.add(call)
    db.commit()
    db.refresh(call)
    return call


def list_calls(
    db: Session,
    *,
    agent: str | None = None,
    sentiment: str | None = None,
    status: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    search: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[Sequence[Call], int]:
    query = select(Call)
    if agent:
        query = query.where(Call.agent_name == agent)
    if sentiment:
        query = query.where(Call.sentiment == sentiment)
    if status:
        query = query.where(Call.status == status)
    if min_score is not None:
        query = query.where(Call.overall_score >= min_score)
    if max_score is not None:
        query = query.where(Call.overall_score <= max_score)
    if search:
        query = query.where(Call.title.ilike(f"%{search}%"))

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = db.scalars(
        query.order_by(Call.created_at.desc(), Call.id.desc()).limit(limit).offset(offset)
    ).all()
    return rows, total


def score_call(db: Session, call: Call, service: ScoringService) -> Call:
    try:
        outcome = service.score(call.transcript)
    except Exception as exc:  # even the fallback failed: record it, don't lose the call
        call.status = CallStatus.FAILED
        call.error = f"{type(exc).__name__}: {exc}"
        db.commit()
        db.refresh(call)
        return call

    result = outcome.result
    call.status = CallStatus.SCORED
    call.overall_score = outcome.overall_score
    call.category_scores = [s.model_dump() for s in result.category_scores]
    call.summary = result.summary
    call.sentiment = result.sentiment
    call.coaching_tips = result.coaching_tips
    call.scored_by = outcome.scored_by
    call.model = outcome.model
    call.input_tokens = outcome.input_tokens
    call.output_tokens = outcome.output_tokens
    call.latency_ms = outcome.latency_ms
    call.scored_at = datetime.now(UTC)
    call.error = outcome.fallback_reason
    db.commit()
    db.refresh(call)
    return call
