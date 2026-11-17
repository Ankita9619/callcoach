"""SQLAlchemy ORM models (database tables)."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class CallStatus(StrEnum):
    PENDING = "pending"  # stored, not scored yet
    SCORED = "scored"
    FAILED = "failed"  # scoring failed even after the fallback


class CallSource(StrEnum):
    PASTE = "paste"
    UPLOAD = "upload"
    SAMPLE = "sample"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    agent_name: Mapped[str | None] = mapped_column(String(100), index=True)
    transcript: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(20), default=CallSource.PASTE)
    status: Mapped[str] = mapped_column(String(20), default=CallStatus.PENDING, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # Scoring results. Nullable until the call has been scored.
    overall_score: Mapped[float | None] = mapped_column(Float, index=True)
    category_scores: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    summary: Mapped[str | None] = mapped_column(Text)
    sentiment: Mapped[str | None] = mapped_column(String(20), index=True)
    coaching_tips: Mapped[list[str] | None] = mapped_column(JSON)

    # Which scorer produced the result, so the UI can flag fallback results.
    scored_by: Mapped[str | None] = mapped_column(String(50))
    model: Mapped[str | None] = mapped_column(String(100))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    scored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
