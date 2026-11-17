# CallCoach backend

FastAPI service that stores call transcripts and scores them against a QA rubric with an LLM.

## Quick start (no API key needed)

```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive API docs. On first start, 8 sample calls from `sample_data/` are loaded and scored in **mock mode**.

## Using a real LLM

```bash
cp .env.example .env
# then set, e.g.:
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

`LLM_PROVIDER=openai` with `OPENAI_API_KEY` works the same way. `LLM_MODEL` overrides the default model. Re-score any call with `POST /api/calls/{id}/score`.

Set `DATABASE_URL=postgresql+psycopg://user:pass@host:5432/callcoach` to use PostgreSQL instead of SQLite.

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | DB status and active LLM provider |
| GET | `/api/rubric` | Rubric categories and weights |
| GET | `/api/calls` | List calls. Filters: `agent`, `sentiment`, `status`, `min_score`, `max_score`, `q`, plus `limit`/`offset` |
| POST | `/api/calls` | Create from pasted transcript (`?score=false` to skip scoring) |
| POST | `/api/calls/upload` | Create from a `.txt`/`.md` file (multipart) |
| GET | `/api/calls/{id}` | Full call with transcript and scores |
| POST | `/api/calls/{id}/score` | (Re-)score a call |
| DELETE | `/api/calls/{id}` | Delete a call |

## How scoring works

```
transcript ─▶ ScoringService ─▶ primary scorer (Anthropic / OpenAI / mock)
                   │                 │ fails? (API error, timeout, invalid JSON)
                   │                 ▼
                   │            retry LLM_MAX_RETRIES times
                   │                 │ still failing?
                   │                 ▼
                   └──────────▶ rule-based MockScorer  (scored_by = "mock-fallback")
```

- **Structured output.** `ScoreResult` (Pydantic) defines the JSON the LLM must return. Anthropic gets its JSON schema as a forced tool (`tool_choice`). OpenAI uses JSON mode with the schema in the prompt. Either way, the reply is validated with Pydantic, and anything invalid raises `ScoringError`.
- **The overall score is computed in code** from the 0–10 category scores and the rubric weights (`app/scoring/rubric.py`). The LLM never does the arithmetic.
- **Fallback is visible.** `scored_by`, `model`, token counts, latency and the fallback reason (`error`) are stored on each call.
- **Prompt-injection guard.** The transcript goes inside `<transcript>` tags, and the system prompt tells the model to treat it as data.

## Project layout

```
app/
  main.py          ASGI entry point (uvicorn app.main:app)
  factory.py       create_app(): wires settings, DB, scorer and routes
  config.py        settings from env vars (pydantic-settings)
  db.py, models.py SQLAlchemy 2.0 engine, session dependency and Call table
  schemas.py       API request/response models
  crud.py          DB operations used by the routes and the seeder
  api/             routers: calls.py, meta.py (health, rubric)
  scoring/         rubric, output schema, mock/Anthropic/OpenAI scorers, fallback service
  seed.py          loads sample_data/*.txt
sample_data/       8 sample transcripts, good and bad
tests/             pytest: unit tests + API tests on in-memory SQLite
```

## Checks

```bash
pytest           # 43 tests, no network or API key needed
ruff check . && ruff format --check .
mypy app tests   # strict mode
```
