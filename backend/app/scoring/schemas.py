"""The structured output every scorer must return.

The same Pydantic model does three jobs:
1. Its JSON schema is sent to the LLM (as a tool schema / in the prompt), so the model
   knows exactly what shape to return.
2. It validates whatever the LLM sends back. Invalid output raises `ValidationError`,
   which triggers the fallback scorer.
3. It is the type the rest of the backend works with.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.scoring.rubric import RUBRIC, CategoryKey

Sentiment = Literal["positive", "neutral", "negative"]


class CategoryScore(BaseModel):
    category: CategoryKey
    score: int = Field(ge=0, le=10, description="0 = very poor, 10 = excellent")
    justification: str = Field(
        min_length=1, max_length=500, description="One or two sentences citing the transcript"
    )


class ScoreResult(BaseModel):
    category_scores: list[CategoryScore] = Field(
        description="Exactly one entry per rubric category"
    )
    summary: str = Field(min_length=1, max_length=1000, description="2-3 sentence call summary")
    sentiment: Sentiment = Field(description="The customer's sentiment by the end of the call")
    coaching_tips: list[str] = Field(
        min_length=1, max_length=5, description="Concrete, actionable tips for the agent"
    )

    @model_validator(mode="after")
    def one_score_per_category(self) -> "ScoreResult":
        got = sorted(s.category for s in self.category_scores)
        expected = sorted(c.key for c in RUBRIC)
        if got != expected:
            raise ValueError(f"expected one score for each of {expected}, got {got}")
        return self

    def scores_by_category(self) -> dict[str, int]:
        return {s.category: s.score for s in self.category_scores}


@dataclass
class ScorerResponse:
    """What a single scorer call returns: the validated result plus usage metadata."""

    result: ScoreResult
    input_tokens: int | None = None
    output_tokens: int | None = None


class ScoringError(Exception):
    """Raised by a scorer when the provider fails or returns unusable output."""
