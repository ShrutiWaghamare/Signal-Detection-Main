"""Pydantic schema for the agent's final structured answer, with a
causal-language guardrail baked directly into validation.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

BANNED_CAUSAL_PHRASES = [
    "proves",
    "confirms",
    "causes",
    "definitively",
    "guaranteed",
    "certainly caused",
]


class SignalAnswer(BaseModel):
    relevant_signals: str
    evidence: str
    interpretation: str
    limitations: str
    recommended_review: str

    @field_validator("interpretation")
    @classmethod
    def no_causal_claims(cls, value: str) -> str:
        lowered = value.lower()
        hits = [phrase for phrase in BANNED_CAUSAL_PHRASES if phrase in lowered]
        if hits:
            raise ValueError(f"causal language detected in interpretation: {hits}")
        return value
