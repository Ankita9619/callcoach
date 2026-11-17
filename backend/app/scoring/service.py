"""Scoring orchestration: pick the scorer, retry, and fall back if the LLM fails.

Flow for one call:
    primary scorer (LLM) --fails--> retry (llm_max_retries times) --still fails--> MockScorer
The result records which scorer produced it, so a fallback score is never mistaken for an
LLM score.
"""

import logging
import time
from dataclasses import dataclass

from app.config import Settings
from app.scoring.base import Scorer
from app.scoring.mock import MockScorer
from app.scoring.rubric import overall_score
from app.scoring.schemas import ScoreResult

logger = logging.getLogger(__name__)


@dataclass
class ScoringOutcome:
    result: ScoreResult
    overall_score: float
    scored_by: str  # "anthropic", "openai", "mock" or "mock-fallback"
    model: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    fallback_reason: str | None = None  # why the primary scorer was abandoned

    @property
    def used_fallback(self) -> bool:
        return self.fallback_reason is not None


class ConfigError(Exception):
    pass


def build_scorer(settings: Settings) -> Scorer:
    """Create the primary scorer from settings. Imports are local so mock mode needs no SDKs."""
    if settings.llm_provider == "mock":
        return MockScorer()
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ConfigError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        from app.scoring.anthropic_scorer import AnthropicScorer

        return AnthropicScorer(
            settings.anthropic_api_key, settings.llm_model, settings.llm_timeout_seconds
        )
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ConfigError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        from app.scoring.openai_scorer import OpenAIScorer

        return OpenAIScorer(
            settings.openai_api_key, settings.llm_model, settings.llm_timeout_seconds
        )
    raise ConfigError(f"Unknown LLM provider: {settings.llm_provider}")


class ScoringService:
    def __init__(self, primary: Scorer, max_retries: int = 1) -> None:
        self.primary = primary
        self.max_retries = max_retries
        self.fallback = MockScorer()

    def score(self, transcript: str) -> ScoringOutcome:
        start = time.perf_counter()
        last_error: str | None = None

        for attempt in range(1 + self.max_retries):
            try:
                response = self.primary.score(transcript)
            # Deliberately broad: *any* failure of the primary scorer (API error, timeout,
            # invalid JSON, a bug in a provider adapter) should degrade to the fallback
            # rather than fail the user's request.
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Scorer %s failed (attempt %d/%d): %s",
                    self.primary.name,
                    attempt + 1,
                    1 + self.max_retries,
                    last_error,
                )
                continue
            return ScoringOutcome(
                result=response.result,
                overall_score=overall_score(response.result.scores_by_category()),
                scored_by=self.primary.name,
                model=self.primary.model,
                latency_ms=_elapsed_ms(start),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )

        logger.error("Falling back to rule-based scorer after: %s", last_error)
        response = self.fallback.score(transcript)
        return ScoringOutcome(
            result=response.result,
            overall_score=overall_score(response.result.scores_by_category()),
            scored_by="mock-fallback",
            model=self.fallback.model,
            latency_ms=_elapsed_ms(start),
            fallback_reason=last_error,
        )


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
