"""Pydantic models for API requests and responses.

Kept separate from the ORM models in `models.py`: the database shape and the public API
shape can then evolve independently (e.g. the list endpoint omits the full transcript).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.scoring.schemas import CategoryScore

MAX_TRANSCRIPT_CHARS = 50_000


class CallCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200, examples=["Billing dispute – refund"])
    agent_name: str | None = Field(default=None, max_length=100, examples=["Sarah"])
    transcript: str = Field(
        min_length=20,
        max_length=MAX_TRANSCRIPT_CHARS,
        examples=["Agent: Thank you for calling, my name is Sarah...\nCustomer: Hi, ..."],
    )


class CallSummary(BaseModel):
    """A row in the call list: everything except the transcript and detailed scores."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    agent_name: str | None
    source: str
    status: str
    created_at: datetime
    overall_score: float | None
    sentiment: str | None
    scored_by: str | None


class CallDetail(CallSummary):
    transcript: str
    category_scores: list[CategoryScore] | None
    summary: str | None
    coaching_tips: list[str] | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int | None
    scored_at: datetime | None
    error: str | None


class CallList(BaseModel):
    items: list[CallSummary]
    total: int
    limit: int
    offset: int


class RubricCategoryOut(BaseModel):
    key: str
    name: str
    weight: float
    description: str


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ok", "error"]
    llm_provider: str
    llm_model: str
