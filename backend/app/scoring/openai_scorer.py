"""Scorer backed by OpenAI's Chat Completions API.

Uses JSON mode (`response_format={"type": "json_object"}`) with the schema described in
the prompt, then validates the reply with Pydantic. Anything that doesn't parse or doesn't
match the schema becomes a `ScoringError`, which triggers the fallback.
"""

from typing import Any

import openai
from pydantic import ValidationError

from app.scoring.prompts import SYSTEM_PROMPT, json_instructions, user_prompt
from app.scoring.schemas import ScoreResult, ScorerResponse, ScoringError

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIScorer:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        timeout: float = 30.0,
        client: Any = None,  # injectable for tests
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self._client: Any = client or openai.OpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    def score(self, transcript: str) -> ScorerResponse:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{json_instructions()}"},
                    {"role": "user", "content": user_prompt(transcript)},
                ],
            )
        except openai.OpenAIError as exc:
            raise ScoringError(f"OpenAI API error: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise ScoringError("OpenAI returned an empty response")

        try:
            result = ScoreResult.model_validate_json(content)
        except ValidationError as exc:
            raise ScoringError(f"OpenAI returned invalid score JSON: {exc}") from exc

        usage = response.usage
        return ScorerResponse(
            result=result,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
