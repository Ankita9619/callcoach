# CallCoach

AI-assisted QA scoring for customer-service calls. Upload or paste a call transcript, and CallCoach scores it against a QA rubric (greeting, empathy, problem resolution, compliance) using an LLM, then shows the results in a dashboard.

> 🚧 Work in progress. See [`docs/project-brief.md`](docs/project-brief.md) for the plan.

## Status

| Stage | What | State |
|---|---|---|
| 1 | FastAPI backend, scoring (mock + LLM with fallback), tests, backend CI | in review |
| 2 | React Router 7 dashboard | planned |
| 3 | Docker Compose, CI, Prometheus/Grafana | planned |

## Run the backend locally

See [`backend/README.md`](backend/README.md).
