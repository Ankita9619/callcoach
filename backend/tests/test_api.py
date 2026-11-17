"""End-to-end API tests through FastAPI's TestClient, against in-memory SQLite."""

from fastapi.testclient import TestClient

from tests.conftest import BAD_TRANSCRIPT, GOOD_TRANSCRIPT, ClientFactory, FailingScorer


def create(client: TestClient, transcript: str = GOOD_TRANSCRIPT, **extra: object) -> dict:  # type: ignore[type-arg]
    body = {"title": "Test call", "agent_name": "Sam", "transcript": transcript} | extra
    response = client.post("/api/calls", json=body)
    assert response.status_code == 201, response.text
    return response.json()  # type: ignore[no-any-return]


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {
        "status": "ok",
        "database": "ok",
        "llm_provider": "mock",
        "llm_model": "rule-based-v1",
    }


def test_rubric(client: TestClient) -> None:
    keys = [c["key"] for c in client.get("/api/rubric").json()]
    assert keys == ["greeting", "empathy", "resolution", "compliance"]


def test_create_scores_call_immediately(client: TestClient) -> None:
    call = create(client)
    assert call["status"] == "scored"
    assert call["scored_by"] == "mock"
    assert call["overall_score"] > 80
    assert {s["category"] for s in call["category_scores"]} == {
        "greeting",
        "empathy",
        "resolution",
        "compliance",
    }
    assert call["coaching_tips"]
    assert call["error"] is None


def test_create_without_scoring_then_score_later(client: TestClient) -> None:
    response = client.post(
        "/api/calls?score=false", json={"title": "Later", "transcript": GOOD_TRANSCRIPT}
    )
    call = response.json()
    assert call["status"] == "pending"
    assert call["overall_score"] is None

    scored = client.post(f"/api/calls/{call['id']}/score").json()
    assert scored["status"] == "scored"
    assert client.get(f"/api/calls/{call['id']}").json()["overall_score"] == scored["overall_score"]


def test_create_validates_input(client: TestClient) -> None:
    assert (
        client.post("/api/calls", json={"title": "x", "transcript": "too short"}).status_code == 422
    )
    assert client.post("/api/calls", json={"transcript": GOOD_TRANSCRIPT}).status_code == 422


def test_get_and_delete_missing_call_return_404(client: TestClient) -> None:
    assert client.get("/api/calls/999").status_code == 404
    assert client.delete("/api/calls/999").status_code == 404
    assert client.post("/api/calls/999/score").status_code == 404


def test_delete(client: TestClient) -> None:
    call = create(client)
    assert client.delete(f"/api/calls/{call['id']}").status_code == 204
    assert client.get(f"/api/calls/{call['id']}").status_code == 404


def test_list_filters_and_pagination(client: TestClient) -> None:
    create(client, title="Refund call", agent_name="Sam")
    create(client, transcript=BAD_TRANSCRIPT, title="Outage call", agent_name="Mark")
    create(client, title="Another refund", agent_name="Sam")

    def ids(query: str) -> list[str]:
        return [c["title"] for c in client.get(f"/api/calls{query}").json()["items"]]

    listing = client.get("/api/calls").json()
    assert listing["total"] == 3
    assert "transcript" not in listing["items"][0]  # list rows are lightweight
    assert ids("") == ["Another refund", "Outage call", "Refund call"]  # newest first
    assert ids("?agent=Mark") == ["Outage call"]
    assert ids("?sentiment=negative") == ["Outage call"]
    assert ids("?min_score=80") == ["Another refund", "Refund call"]
    assert ids("?max_score=30") == ["Outage call"]
    assert ids("?q=refund") == ["Another refund", "Refund call"]

    page = client.get("/api/calls?limit=2&offset=2").json()
    assert (page["total"], len(page["items"])) == (3, 1)


def test_list_rejects_invalid_filters(client: TestClient) -> None:
    assert client.get("/api/calls?sentiment=furious").status_code == 422
    assert client.get("/api/calls?limit=1000").status_code == 422


def test_upload_text_transcript(client: TestClient) -> None:
    response = client.post(
        "/api/calls/upload",
        files={"file": ("billing_refund.txt", GOOD_TRANSCRIPT.encode(), "text/plain")},
        data={"agent_name": "Sam"},
    )
    assert response.status_code == 201, response.text
    call = response.json()
    assert (call["title"], call["source"], call["status"]) == ("billing refund", "upload", "scored")


def test_upload_rejects_audio_and_unknown_types(client: TestClient) -> None:
    audio = client.post("/api/calls/upload", files={"file": ("call.mp3", b"ID3", "audio/mpeg")})
    pdf = client.post("/api/calls/upload", files={"file": ("call.pdf", b"%PDF", "application/pdf")})
    assert audio.status_code == 415
    assert pdf.status_code == 415


def test_upload_rejects_non_utf8_and_too_short(client: TestClient) -> None:
    binary = client.post(
        "/api/calls/upload", files={"file": ("x.txt", b"\xff\xfe\x00", "text/plain")}
    )
    short = client.post("/api/calls/upload", files={"file": ("x.txt", b"hi", "text/plain")})
    assert binary.status_code == 422
    assert short.status_code == 422


def test_llm_failure_falls_back_and_is_visible_in_api(make_client: ClientFactory) -> None:
    scorer = FailingScorer()
    client = make_client(scorer=scorer, llm_max_retries=1)

    call = create(client)
    assert scorer.calls == 2  # first attempt + one retry
    assert call["status"] == "scored"
    assert call["scored_by"] == "mock-fallback"
    assert "provider unavailable" in call["error"]
    assert client.get("/health").json()["llm_provider"] == "failing"


def test_sample_data_is_seeded_and_scored_on_startup(make_client: ClientFactory) -> None:
    client = make_client(seed_sample_data=True)
    listing = client.get("/api/calls?limit=100").json()
    assert listing["total"] >= 5
    assert all(c["status"] == "scored" and c["source"] == "sample" for c in listing["items"])


def test_seeding_with_llm_provider_leaves_samples_unscored(make_client: ClientFactory) -> None:
    client = make_client(scorer=FailingScorer(), seed_sample_data=True)
    statuses = {c["status"] for c in client.get("/api/calls?limit=100").json()["items"]}
    assert statuses == {"pending"}
