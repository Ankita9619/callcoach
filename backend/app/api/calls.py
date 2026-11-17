"""/api/calls endpoints."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app import crud
from app.api.deps import get_scoring_service
from app.db import get_db
from app.models import Call, CallSource
from app.schemas import MAX_TRANSCRIPT_CHARS, CallCreate, CallDetail, CallList, CallSummary
from app.scoring.schemas import Sentiment
from app.scoring.service import ScoringService

router = APIRouter(prefix="/api/calls", tags=["calls"])

DbSession = Annotated[Session, Depends(get_db)]
Scoring = Annotated[ScoringService, Depends(get_scoring_service)]

TEXT_EXTENSIONS = {".txt", ".md"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".webm"}


def _get_or_404(db: Session, call_id: int) -> Call:
    call = db.get(Call, call_id)
    if call is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Call {call_id} not found")
    return call


@router.get("", response_model=CallList)
def list_calls(
    db: DbSession,
    agent: str | None = None,
    sentiment: Sentiment | None = None,
    call_status: Annotated[str | None, Query(alias="status")] = None,
    min_score: Annotated[float | None, Query(ge=0, le=100)] = None,
    max_score: Annotated[float | None, Query(ge=0, le=100)] = None,
    q: Annotated[str | None, Query(max_length=100, description="Search in titles")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CallList:
    rows, total = crud.list_calls(
        db,
        agent=agent,
        sentiment=sentiment,
        status=call_status,
        min_score=min_score,
        max_score=max_score,
        search=q,
        limit=limit,
        offset=offset,
    )
    return CallList(
        items=[CallSummary.model_validate(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("", response_model=CallDetail, status_code=status.HTTP_201_CREATED)
def create_call(
    body: CallCreate,
    db: DbSession,
    scoring: Scoring,
    score: Annotated[bool, Query(description="Score immediately")] = True,
) -> Call:
    call = crud.create_call(
        db, title=body.title, transcript=body.transcript, agent_name=body.agent_name
    )
    return crud.score_call(db, call, scoring) if score else call


@router.post("/upload", response_model=CallDetail, status_code=status.HTTP_201_CREATED)
async def upload_transcript(
    db: DbSession,
    scoring: Scoring,
    file: Annotated[UploadFile, File(description="A .txt or .md transcript")],
    title: Annotated[str | None, Form(max_length=200)] = None,
    agent_name: Annotated[str | None, Form(max_length=100)] = None,
    score: Annotated[bool, Query(description="Score immediately")] = True,
) -> Call:
    filename = file.filename or "transcript.txt"
    suffix = Path(filename).suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "Audio transcription is not enabled yet. Upload a .txt transcript.",
        )
    if suffix not in TEXT_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported file type '{suffix}'"
        )

    raw = await file.read(MAX_TRANSCRIPT_CHARS * 4 + 1)  # UTF-8 is at most 4 bytes/char
    try:
        transcript = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise HTTPException(422, "File must be UTF-8 text") from exc

    # Reuse CallCreate's validation (length limits) for uploads too.
    try:
        body = CallCreate(
            title=title or Path(filename).stem.replace("_", " ").strip() or "Uploaded call",
            agent_name=agent_name or None,
            transcript=transcript,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    call = crud.create_call(
        db,
        title=body.title,
        transcript=body.transcript,
        agent_name=body.agent_name,
        source=CallSource.UPLOAD,
    )
    return crud.score_call(db, call, scoring) if score else call


@router.get("/{call_id}", response_model=CallDetail)
def get_call(call_id: int, db: DbSession) -> Call:
    return _get_or_404(db, call_id)


@router.post("/{call_id}/score", response_model=CallDetail)
def score_call(call_id: int, db: DbSession, scoring: Scoring) -> Call:
    """(Re-)score a call, e.g. after switching from mock mode to a real LLM."""
    return crud.score_call(db, _get_or_404(db, call_id), scoring)


@router.delete("/{call_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_call(call_id: int, db: DbSession) -> None:
    db.delete(_get_or_404(db, call_id))
    db.commit()
