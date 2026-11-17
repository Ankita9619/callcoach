# CallCoach – project brief

**Why:** A GitHub portfolio project that shows FastAPI, React Router 7 and LLM API skills for Full Stack Engineer role. The posting asks for React + TypeScript, Python backends (FastAPI/Django), LLM/voice/NLP in production, PostgreSQL, Docker, CI/CD, Linux, and Prometheus/Grafana. 

**What it does**
1. Upload a call recording or paste a transcript. Audio is transcribed with a speech-to-text API (e.g. Whisper).
2. An LLM scores the call against a QA rubric (greeting, empathy, problem resolution, compliance). It returns structured JSON with scores, a summary, sentiment and coaching tips.
3. A React dashboard shows the call list with scores and filters, plus a detail page. The LLM output streams into the UI.

**Stack**
- Frontend: React Router 7 (framework mode: loaders/actions), TypeScript, Vitest
- Backend: FastAPI, Python 3.11, Pydantic, type hints, pre-commit (ruff, mypy), pytest
- LLM: OpenAI or Anthropic API with structured output / function calling, plus a fallback when the LLM call fails or returns invalid JSON
- DB: PostgreSQL
- Ops: Docker Compose, GitHub Actions CI, Prometheus `/metrics` endpoint + one Grafana dashboard (latency, LLM errors, token usage)

**Scope:** 1–2 weekends. Include 5–10 sample transcripts and a mock-LLM mode so reviewers can run it without API keys. `docker compose up` runs everything. Write a strong README with an architecture diagram and screenshots.

**Ankita's background:** 6+ years experience (Amazon SDE, Accenture). React + TypeScript frontend, Python Lambdas behind an AWS Bedrock agent, Java backends. New to FastAPI and React Router 7.
