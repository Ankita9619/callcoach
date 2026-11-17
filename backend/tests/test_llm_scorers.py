"""Tests for the Anthropic and OpenAI adapters, using fake SDK clients (no network).

These check the parts we own: the request we send, how we parse the reply, and that bad
replies become ScoringError (which the service turns into a fallback).
"""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.scoring.anthropic_scorer import AnthropicScorer
from app.scoring.openai_scorer import OpenAIScorer
from app.scoring.prompts import TOOL_NAME
from app.scoring.rubric import RUBRIC
from app.scoring.schemas import ScoringError

VALID = {
    "category_scores": [{"category": c.key, "score": 8, "justification": "Ok."} for c in RUBRIC],
    "summary": "Customer was refunded.",
    "sentiment": "positive",
    "coaching_tips": ["Offer a reference number earlier."],
}


class FakeAnthropicMessages:
    def __init__(self, content: list[Any]) -> None:
        self.content = content
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(
            content=self.content, usage=SimpleNamespace(input_tokens=321, output_tokens=123)
        )


def anthropic_with(content: list[Any]) -> tuple[AnthropicScorer, FakeAnthropicMessages]:
    messages = FakeAnthropicMessages(content)
    client = SimpleNamespace(messages=messages)
    return AnthropicScorer(api_key="test", client=client), messages


def test_anthropic_forces_tool_use_and_parses_result() -> None:
    scorer, messages = anthropic_with([SimpleNamespace(type="tool_use", input=VALID)])
    response = scorer.score("Agent: hi")

    assert response.result.sentiment == "positive"
    assert (response.input_tokens, response.output_tokens) == (321, 123)
    assert messages.kwargs["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
    assert messages.kwargs["tools"][0]["input_schema"]["title"] == "ScoreResult"
    assert "<transcript>" in messages.kwargs["messages"][0]["content"]


def test_anthropic_without_tool_use_block_raises() -> None:
    scorer, _ = anthropic_with([SimpleNamespace(type="text", text="Sure! Here's my review…")])
    with pytest.raises(ScoringError, match="no tool_use"):
        scorer.score("Agent: hi")


def test_anthropic_invalid_tool_input_raises() -> None:
    scorer, _ = anthropic_with([SimpleNamespace(type="tool_use", input={"summary": "x"})])
    with pytest.raises(ScoringError, match="invalid score JSON"):
        scorer.score("Agent: hi")


class FakeOpenAICompletions:
    def __init__(self, content: str | None) -> None:
        self.content = content
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=200, completion_tokens=80),
        )


def openai_with(content: str | None) -> tuple[OpenAIScorer, FakeOpenAICompletions]:
    completions = FakeOpenAICompletions(content)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAIScorer(api_key="test", client=client), completions


def test_openai_uses_json_mode_and_parses_result() -> None:
    scorer, completions = openai_with(json.dumps(VALID))
    response = scorer.score("Agent: hi")

    assert response.result.summary == "Customer was refunded."
    assert (response.input_tokens, response.output_tokens) == (200, 80)
    assert completions.kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("content", [None, "not json at all", json.dumps({"summary": "x"})])
def test_openai_bad_output_raises(content: str | None) -> None:
    scorer, _ = openai_with(content)
    with pytest.raises(ScoringError):
        scorer.score("Agent: hi")
