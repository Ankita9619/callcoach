"""Application settings, read from environment variables (or a local .env file).

pydantic-settings maps each field to an env var of the same name, case-insensitive,
e.g. `LLM_PROVIDER=anthropic` sets `settings.llm_provider`.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["mock", "anthropic", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SQLite keeps local dev dependency-free; Docker Compose points this at PostgreSQL.
    database_url: str = "sqlite:///./callcoach.db"

    # "mock" needs no API key, so anyone can run the project.
    llm_provider: LLMProvider = "mock"
    llm_model: str | None = None  # None -> provider's default model
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 1  # extra attempts before falling back to the mock scorer

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    seed_sample_data: bool = True
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
