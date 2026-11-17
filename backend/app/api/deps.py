"""Shared FastAPI dependencies."""

from fastapi import Request

from app.scoring.service import ScoringService


def get_scoring_service(request: Request) -> ScoringService:
    service: ScoringService = request.app.state.scoring_service
    return service
