"""The QA rubric: what a good call looks like and how much each part counts.

Each category is scored 0-10 by the scorer (LLM or mock). The overall 0-100 score is
computed here in code, not by the LLM, so the maths is deterministic and auditable.
"""

from dataclasses import dataclass
from typing import Literal

CategoryKey = Literal["greeting", "empathy", "resolution", "compliance"]


@dataclass(frozen=True)
class RubricCategory:
    key: CategoryKey
    name: str
    weight: float
    description: str


RUBRIC: tuple[RubricCategory, ...] = (
    RubricCategory(
        key="greeting",
        name="Greeting & opening",
        weight=0.20,
        description=(
            "Agent greets the customer warmly, introduces themselves by name and "
            "invites the customer to explain their issue."
        ),
    ),
    RubricCategory(
        key="empathy",
        name="Empathy & tone",
        weight=0.25,
        description=(
            "Agent acknowledges the customer's feelings, apologises where appropriate, "
            "stays polite and patient, and never sounds dismissive."
        ),
    ),
    RubricCategory(
        key="resolution",
        name="Problem resolution",
        weight=0.35,
        description=(
            "Agent understands the issue, resolves it or sets a clear next step "
            "(reference number, timeline), and confirms nothing else is needed."
        ),
    ),
    RubricCategory(
        key="compliance",
        name="Compliance",
        weight=0.20,
        description=(
            "Agent mentions call recording where required, verifies identity before "
            "discussing the account, never asks for full card numbers, PINs or passwords, "
            "and closes the call properly."
        ),
    ),
)

RUBRIC_BY_KEY: dict[str, RubricCategory] = {c.key: c for c in RUBRIC}

assert abs(sum(c.weight for c in RUBRIC) - 1.0) < 1e-9, "rubric weights must sum to 1"


def overall_score(category_scores: dict[str, int]) -> float:
    """Weighted average of 0-10 category scores, scaled to 0-100 and rounded to 1 decimal."""
    total = sum(RUBRIC_BY_KEY[key].weight * score for key, score in category_scores.items())
    return round(total * 10, 1)


def rubric_as_prompt_text() -> str:
    lines = [
        f"- {c.key} ({c.name}, weight {int(c.weight * 100)}%): {c.description}" for c in RUBRIC
    ]
    return "\n".join(lines)
