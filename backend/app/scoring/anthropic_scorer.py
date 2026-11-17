"""Scorer backed by Anthropic's Messages API.

Structured output via *forced tool use*: we declare one tool whose input schema is the
`ScoreResult` JSON schema and set `tool_choice` to that tool. Claude must then reply with
a `tool_use` block whose `input` is the JSON we want, which we validate with Pydantic.
"""

from typing import Any

import anthropic
from pydantic import ValidationError

from app.scoring.prompts import SYSTEM_PROMPT, TOOL_DESCRIPTION, TOOL_NAME, user_prompt
from app.scoring.schemas import ScoreResult, ScorerResponse, ScoringError

DEFAULT_MODEL = "claude-haiku-5-5"


class AnthropicScorer:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        timeout: float = 30.0,
        client: Any = None,  # injectable for tests
    ) -> None:
        self.model = model or DEFAULT_MODEL
        # max_retries=0: retries and fallback are handled by our scoring service.
        self._client: Any = client or anthropic.Anthropic(
            api_key=api_key, timeout=timeout, max_retries=0
        )

    def score(self, transcript: str) -> ScorerResponse:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                tools=[
                    {
                        "name": TOOL_NAME,
                        "description": TOOL_DESCRIPTION,
                        "input_schema": ScoreResult.model_json_schema(),
                    }
                ],
                tool_choice={"type": "tool", "name": TOOL_NAME},
                messages=[{"role": "user", "content": user_prompt(transcript)}],
            )
        except anthropic.APIError as exc:
            raise ScoringError(f"Anthropic API error: {exc}") from exc

        tool_input = next(
            (b.input for b in response.content if getattr(b, "type", None) == "tool_use"), None
        )
        if tool_input is None:
            raise ScoringError("Anthropic response contained no tool_use block")

        try:
            result = ScoreResult.model_validate(tool_input)
        except ValidationError as exc:
            raise ScoringError(f"Anthropic returned invalid score JSON: {exc}") from exc

        return ScorerResponse(
            result=result,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
