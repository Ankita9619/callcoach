"""The interface every scorer implements.

Using a Protocol (structural typing) means the scoring service and the API never import
a concrete provider. Swapping Anthropic for OpenAI, or for a fake in tests, is a config
change, not a code change.
"""

from typing import Protocol

from app.scoring.schemas import ScorerResponse


class Scorer(Protocol):
    name: str  # e.g. "mock", "anthropic", "openai"
    model: str  # model identifier, for display and metrics

    def score(self, transcript: str) -> ScorerResponse:
        """Score a transcript. Raise `ScoringError` if the result can't be produced."""
        ...
