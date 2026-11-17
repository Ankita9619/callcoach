"""A deterministic, rule-based scorer.

It looks for typical phrases in the transcript. That's crude compared with an LLM, but it
has two real jobs:
- **Mock mode**: reviewers can run the whole app without an API key, and scores still
  differ between good and bad calls.
- **Fallback**: if the LLM is down or returns invalid JSON, the call still gets a score,
  flagged as a fallback result.
"""

import re
from dataclasses import dataclass

from app.scoring.rubric import RUBRIC_BY_KEY
from app.scoring.schemas import CategoryScore, ScoreResult, ScorerResponse, Sentiment

GREETING_OPENERS = (
    "hello", "hi ", "hi,", "good morning", "good afternoon", "good evening",
    "thank you for calling", "thanks for calling", "welcome",
)  # fmt: skip
INTRODUCTIONS = ("my name is", "this is", "speaking", "you're speaking with", "you are through to")
INVITATIONS = ("how can i help", "how may i help", "what can i do for you", "how can i assist")

EMPATHY = (
    "i understand", "i'm sorry", "i am sorry", "i apologise", "i apologize", "apologies",
    "i can imagine", "that must be", "i appreciate", "thank you for your patience",
    "i completely understand", "sorry to hear", "that sounds frustrating",
)  # fmt: skip
DISMISSIVE = (
    "calm down", "not my problem", "nothing i can do", "you should have", "that's not my job",
    "i already told you", "you need to listen",
)  # fmt: skip

RESOLUTION = (
    "resolved", "i've processed", "i have processed", "refund", "replacement", "i've updated",
    "i have updated", "fixed", "reference number", "ticket number", "confirmation email",
    "is there anything else", "anything else i can help", "within 24 hours",
    "within 3-5 business days", "i've arranged", "i have arranged", "i've booked",
)  # fmt: skip
UNRESOLVED = ("call back later", "can't help", "cannot help", "try again later", "no idea")

RECORDING = ("call may be recorded", "call is being recorded", "recorded for quality")
VERIFICATION = (
    "verify", "date of birth", "security question", "postcode", "zip code",
    "last four digits", "account number", "confirm your",
)  # fmt: skip
CLOSING = ("have a great day", "have a nice day", "thank you for calling", "goodbye", "take care")
# Asking for these over the phone is a breach. Phrased as requests, so that an agent
# *warning* "don't share your password" isn't penalised.
NON_COMPLIANT = (
    "full card number", "the cvv", "your cvv", "tell me your password", "read me your password",
    "what's your password", "what is your password", "your pin number", "tell me your pin",
)  # fmt: skip

POSITIVE = (
    "thank", "great", "perfect", "appreciate", "helpful", "happy", "awesome", "brilliant",
    "excellent", "wonderful", "lovely", "that works", "sorted",
)  # fmt: skip
NEGATIVE = (
    "angry", "frustrated", "ridiculous", "terrible", "unacceptable", "worst", "annoyed",
    "annoying", "disappointed", "disappointing", "useless", "complaint", "waste of time",
    "fed up",
)  # fmt: skip

TIPS: dict[str, str] = {
    "greeting": ("Open with a warm greeting, introduce yourself by name and ask how you can help."),
    "empathy": (
        "Acknowledge the customer's frustration explicitly (e.g. \"I understand how "
        'annoying that is") before moving to the fix, and avoid dismissive phrases.'
    ),
    "resolution": (
        "Close the loop: state the fix or next step, give a reference number and "
        "timeline, and ask whether there's anything else you can help with."
    ),
    "compliance": (
        "Mention call recording, verify identity before discussing the account, never "
        "ask for full card numbers or PINs, and close the call politely."
    ),
}

_SPEAKER_RE = re.compile(r"^\s*([A-Za-z][\w .'-]{0,30}):\s*(.*)$")


@dataclass
class Turn:
    speaker: str  # "agent" or "customer"
    text: str


def parse_turns(transcript: str) -> list[Turn]:
    """Split `Agent: ...` / `Customer: ...` lines into turns.

    Unlabelled lines continue the previous turn. If nothing is labelled at all, the whole
    transcript is treated as agent speech, so scoring still works.
    """
    turns: list[Turn] = []
    for raw in transcript.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SPEAKER_RE.match(line)
        if match:
            label, text = match.group(1).lower(), match.group(2)
            speaker = "customer" if label.startswith(("customer", "caller", "client")) else "agent"
            turns.append(Turn(speaker, text))
        elif turns:
            turns[-1].text += " " + line
        else:
            turns.append(Turn("agent", line))
    return turns


def _normalise(text: str) -> str:
    return text.lower().replace("’", "'")


def _found(text: str, phrases: tuple[str, ...]) -> list[str]:
    return [p.strip() for p in phrases if p in text]


def _clamp(value: int) -> int:
    return max(0, min(10, value))


def _justify(found: list[str], bad: list[str], empty: str) -> str:
    parts = []
    if found:
        parts.append("Detected: " + ", ".join(f"'{p}'" for p in found[:4]) + ".")
    if bad:
        parts.append("Problems: " + ", ".join(f"'{p}'" for p in bad[:3]) + ".")
    return " ".join(parts) or empty


class MockScorer:
    name = "mock"
    model = "rule-based-v1"

    def score(self, transcript: str) -> ScorerResponse:
        turns = parse_turns(transcript)
        agent_text = _normalise(" ".join(t.text for t in turns if t.speaker == "agent"))
        first_agent = next((_normalise(t.text) for t in turns if t.speaker == "agent"), "")

        # Greeting: judged on the agent's first turn only.
        opener = _found(first_agent, GREETING_OPENERS)
        intro = _found(first_agent, INTRODUCTIONS)
        invite = _found(agent_text, INVITATIONS)
        greeting = _clamp(2 + 3 * bool(opener) + 3 * bool(intro) + 2 * bool(invite))

        empathy_hits = _found(agent_text, EMPATHY)
        dismissive = _found(agent_text, DISMISSIVE)
        empathy = _clamp(2 + 2 * len(empathy_hits) - 3 * len(dismissive))

        resolution_hits = _found(agent_text, RESOLUTION)
        unresolved = _found(agent_text, UNRESOLVED)
        resolution = _clamp(2 + 2 * len(resolution_hits) - 3 * len(unresolved))

        compliance_hits = (
            _found(agent_text, RECORDING)[:1]
            + _found(agent_text, VERIFICATION)[:1]
            + _found(agent_text, CLOSING)[:1]
        )
        non_compliant = _found(agent_text, NON_COMPLIANT)
        compliance = _clamp(1 + 3 * len(compliance_hits) - 4 * len(non_compliant))

        category_scores = [
            CategoryScore(
                category="greeting",
                score=greeting,
                justification=_justify(opener + intro + invite, [], "No greeting detected."),
            ),
            CategoryScore(
                category="empathy",
                score=empathy,
                justification=_justify(empathy_hits, dismissive, "No empathy phrases detected."),
            ),
            CategoryScore(
                category="resolution",
                score=resolution,
                justification=_justify(
                    resolution_hits, unresolved, "No clear resolution or next step detected."
                ),
            ),
            CategoryScore(
                category="compliance",
                score=compliance,
                justification=_justify(
                    compliance_hits, non_compliant, "No compliance steps detected."
                ),
            ),
        ]

        result = ScoreResult(
            category_scores=category_scores,
            summary=self._summary(turns, category_scores),
            sentiment=self._sentiment(turns),
            coaching_tips=self._tips(category_scores),
        )
        return ScorerResponse(result=result)

    @staticmethod
    def _sentiment(turns: list[Turn]) -> Sentiment:
        customer = [_normalise(t.text) for t in turns if t.speaker == "customer"]
        if not customer:
            return "neutral"
        # The end of the call matters most: weight the last two customer turns double.
        weighted = customer + customer[-2:]
        balance = sum(len(_found(t, POSITIVE)) - len(_found(t, NEGATIVE)) for t in weighted)
        if balance > 0:
            return "positive"
        if balance < 0:
            return "negative"
        return "neutral"

    @staticmethod
    def _summary(turns: list[Turn], scores: list[CategoryScore]) -> str:
        first_customer = next((t.text for t in turns if t.speaker == "customer"), "")
        issue = (first_customer[:160] + "…") if len(first_customer) > 160 else first_customer
        best = max(scores, key=lambda s: s.score)
        worst = min(scores, key=lambda s: s.score)
        parts = [f"{len(turns)}-turn call."]
        if issue:
            parts.append(f'The customer opened with: "{issue}"')
        parts.append(
            f"Strongest area: {RUBRIC_BY_KEY[best.category].name.lower()} ({best.score}/10); "
            f"weakest: {RUBRIC_BY_KEY[worst.category].name.lower()} ({worst.score}/10)."
        )
        return " ".join(parts)

    @staticmethod
    def _tips(scores: list[CategoryScore]) -> list[str]:
        weak = sorted((s for s in scores if s.score < 7), key=lambda s: s.score)
        tips = [TIPS[s.category] for s in weak]
        return tips or ["Strong call. Keep using the same structure and tone."]
