from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.factory import create_app
from app.scoring.base import Scorer
from app.scoring.mock import MockScorer
from app.scoring.schemas import ScorerResponse, ScoringError

GOOD_TRANSCRIPT = """\
Agent: Good morning, thank you for calling Acme. My name is Sam. This call may be recorded.
How can I help you today?
Customer: Hi, I was charged twice and I'm frustrated.
Agent: I'm sorry to hear that, I completely understand. Could you confirm your date of birth?
Customer: 1 May 1990.
Agent: Thanks. I've processed a refund, your reference number is R-1. Is there anything else?
Customer: No, that's great, thank you!
Agent: Thank you for calling, have a great day."""

BAD_TRANSCRIPT = """\
Agent: Yeah?
Customer: My internet is down, this is ridiculous.
Agent: Calm down. There's nothing I can do, call back later.
Customer: Useless. I want to make a complaint."""


class FailingScorer:
    """Always fails, like an LLM provider that's down."""

    name = "failing"
    model = "fail-1"

    def __init__(self) -> None:
        self.calls = 0

    def score(self, transcript: str) -> ScorerResponse:
        self.calls += 1
        raise ScoringError("provider unavailable")


class FlakyScorer:
    """Fails the first `failures` times, then delegates to the mock scorer."""

    name = "flaky"
    model = "flaky-1"

    def __init__(self, failures: int = 1) -> None:
        self.failures = failures
        self.calls = 0

    def score(self, transcript: str) -> ScorerResponse:
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError("timed out")
        response = MockScorer().score(transcript)
        response.input_tokens, response.output_tokens = 100, 50
        return response


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "sqlite:///:memory:",
        "seed_sample_data": False,
        "llm_provider": "mock",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


ClientFactory = Callable[..., TestClient]


@pytest.fixture
def make_client() -> Iterator[ClientFactory]:
    clients: list[TestClient] = []

    def factory(scorer: Scorer | None = None, **settings: object) -> TestClient:
        client = TestClient(create_app(make_settings(**settings), scorer=scorer))
        clients.append(client)
        return client

    yield factory
    for c in clients:
        c.close()


@pytest.fixture
def client(make_client: ClientFactory) -> TestClient:
    return make_client()
