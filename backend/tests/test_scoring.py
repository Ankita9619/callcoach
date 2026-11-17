"""Unit tests for the rubric, the output schema, the mock scorer and the fallback service."""

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.scoring.mock import MockScorer, parse_turns
from app.scoring.rubric import RUBRIC, overall_score
from app.scoring.schemas import ScoreResult
from app.scoring.service import ConfigError, ScoringService, build_scorer
from app.seed import load_samples
from tests.conftest import BAD_TRANSCRIPT, GOOD_TRANSCRIPT, FailingScorer, FlakyScorer


def valid_payload() -> dict[str, object]:
    return {
        "category_scores": [
            {"category": c.key, "score": 7, "justification": "Fine."} for c in RUBRIC
        ],
        "summary": "A call.",
        "sentiment": "neutral",
        "coaching_tips": ["Do better."],
    }


# --- rubric ---------------------------------------------------------------------------


def test_rubric_weights_sum_to_one() -> None:
    assert sum(c.weight for c in RUBRIC) == pytest.approx(1.0)


def test_overall_score_is_weighted_and_scaled_to_100() -> None:
    assert overall_score({c.key: 10 for c in RUBRIC}) == 100.0
    assert overall_score({c.key: 0 for c in RUBRIC}) == 0.0
    # Only resolution (weight 0.35) at 10 -> 35.
    scores: dict[str, int] = {c.key: 0 for c in RUBRIC}
    scores["resolution"] = 10
    assert overall_score(scores) == 35.0


# --- output schema --------------------------------------------------------------------


def test_score_result_accepts_valid_payload() -> None:
    assert ScoreResult.model_validate(valid_payload()).sentiment == "neutral"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["category_scores"].pop(),  # missing category
        lambda p: p["category_scores"].append(p["category_scores"][0]),  # duplicate
        lambda p: p["category_scores"][0].update(score=11),  # out of range
        lambda p: p.update(sentiment="furious"),  # not an allowed value
        lambda p: p.update(coaching_tips=[]),  # at least one tip
    ],
)
def test_score_result_rejects_invalid_payload(mutate) -> None:  # type: ignore[no-untyped-def]
    payload = valid_payload()
    mutate(payload)
    with pytest.raises(ValidationError):
        ScoreResult.model_validate(payload)


# --- mock scorer ----------------------------------------------------------------------


def test_parse_turns_handles_labels_and_continuation_lines() -> None:
    turns = parse_turns("Agent: Hello\nthere\nCustomer: Hi\n\n# comment\nCaller: Bye")
    assert [(t.speaker, t.text) for t in turns] == [
        ("agent", "Hello there"),
        ("customer", "Hi"),
        ("customer", "Bye"),
    ]


def test_parse_turns_without_labels_treats_text_as_agent() -> None:
    assert parse_turns("just some text")[0].speaker == "agent"


def test_mock_scores_good_call_higher_than_bad_call() -> None:
    scorer = MockScorer()
    good = scorer.score(GOOD_TRANSCRIPT).result
    bad = scorer.score(BAD_TRANSCRIPT).result
    assert overall_score(good.scores_by_category()) > 80
    assert overall_score(bad.scores_by_category()) < 30
    assert good.sentiment == "positive"
    assert bad.sentiment == "negative"


def test_mock_penalises_asking_for_card_details_but_not_warnings() -> None:
    scorer = MockScorer()
    breach = scorer.score("Agent: Please read me your full card number and the CVV.")
    warning = scorer.score("Agent: Please never tell anyone your password.")
    assert breach.result.scores_by_category()["compliance"] == 0
    assert warning.result.scores_by_category()["compliance"] > 0


def test_mock_is_deterministic() -> None:
    scorer = MockScorer()
    assert scorer.score(GOOD_TRANSCRIPT) == scorer.score(GOOD_TRANSCRIPT)


def test_sample_transcripts_load_and_score() -> None:
    samples = load_samples()
    assert 5 <= len(samples) <= 10
    scores = [
        overall_score(MockScorer().score(s.transcript).result.scores_by_category()) for s in samples
    ]
    assert max(scores) - min(scores) > 50, "samples should cover good and bad calls"
    assert all(s.agent_name for s in samples)


# --- scoring service: retries and fallback --------------------------------------------


def test_service_uses_primary_scorer_when_it_succeeds() -> None:
    outcome = ScoringService(FlakyScorer(failures=0)).score(GOOD_TRANSCRIPT)
    assert outcome.scored_by == "flaky"
    assert not outcome.used_fallback
    assert (outcome.input_tokens, outcome.output_tokens) == (100, 50)


def test_service_retries_then_succeeds() -> None:
    scorer = FlakyScorer(failures=1)
    outcome = ScoringService(scorer, max_retries=1).score(GOOD_TRANSCRIPT)
    assert scorer.calls == 2
    assert outcome.scored_by == "flaky"


def test_service_falls_back_after_retries_are_exhausted() -> None:
    scorer = FailingScorer()
    outcome = ScoringService(scorer, max_retries=2).score(GOOD_TRANSCRIPT)
    assert scorer.calls == 3
    assert outcome.scored_by == "mock-fallback"
    assert outcome.used_fallback
    assert "provider unavailable" in (outcome.fallback_reason or "")
    assert outcome.overall_score > 0


def test_build_scorer_defaults_to_mock() -> None:
    assert build_scorer(Settings(llm_provider="mock")).name == "mock"


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_build_scorer_requires_api_key(provider: str) -> None:
    settings = Settings(llm_provider=provider, anthropic_api_key=None, openai_api_key=None)
    with pytest.raises(ConfigError):
        build_scorer(settings)


def test_build_scorer_creates_llm_scorers_with_key() -> None:
    a = build_scorer(Settings(llm_provider="anthropic", anthropic_api_key="sk-test"))
    o = build_scorer(Settings(llm_provider="openai", openai_api_key="sk-test", llm_model="x"))
    assert (a.name, o.name, o.model) == ("anthropic", "openai", "x")
