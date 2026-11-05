"""Data model and four-span (de)serialization for Kardia-R1.

A KardiaBench-style dialogue is anchored to a :class:`UserProfile` and a
situation, and evolves over multiple turns.  Each assistant turn is stored as
four explicit components -- understanding, reasoning, emotion, response --
which serialize to the tagged four-span format described in Section 4 of the
paper (see :mod:`kardia.tags`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .tags import (
    SPANS,
    UNDERSTANDING_BEGIN,
    UNDERSTANDING_END,
    REASONING_BEGIN,
    REASONING_END,
    EMOTION_BEGIN,
    EMOTION_END,
    RESPONSE_BEGIN,
    RESPONSE_END,
)


@dataclass
class UserProfile:
    """A (anonymized) real-world user identity grounding the dialogue."""

    user_id: str
    mbti: str = ""
    gender: str = ""
    relationship: str = ""
    occupation: str = ""
    about: str = ""
    recent_activities: str = ""
    extra: Dict[str, str] = field(default_factory=dict)

    def to_text(self) -> str:
        """Render the profile as a compact, prompt-ready text block."""
        lines: List[str] = [f"Username: {self.user_id}"]
        for label, value in (
            ("MBTI", self.mbti),
            ("Gender", self.gender),
            ("Relationship", self.relationship),
            ("Occupation", self.occupation),
            ("About", self.about),
            ("Recent Activities", self.recent_activities),
        ):
            if value:
                lines.append(f"{label}: {value}")
        for k, v in self.extra.items():
            if v:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)


@dataclass
class Turn:
    """A single assistant turn with its four progressive components.

    ``emotion`` is the gold affect label (used by the emotion-matching
    reward); ``query`` is the user utterance that this turn responds to.
    """

    query: str = ""
    understanding: str = ""
    reasoning: str = ""
    emotion: str = ""
    response: str = ""

    def to_spans(self) -> str:
        """Serialize to the tagged four-span format."""
        return spans_to_text(
            understanding=self.understanding,
            reasoning=self.reasoning,
            emotion=self.emotion,
            response=self.response,
        )


@dataclass
class Dialogue:
    """A multi-turn persona-grounded empathetic dialogue.

    ``p_fail`` / ``p_pass`` / ``p_solved`` are the empirical rubric-decision
    rates (FAIL / PASS / SOLVED) used for difficulty-aware partitioning
    (Eq. 4).  A trajectory is *easy* if failure is not dominant.
    """

    dialogue_id: str
    profile: UserProfile
    situation: str = ""
    turns: List[Turn] = field(default_factory=list)
    p_fail: float = 0.0
    p_pass: float = 0.0
    p_solved: float = 0.0


def spans_to_text(
    understanding: str,
    reasoning: str,
    emotion: str,
    response: str,
) -> str:
    """Encode the four components into the canonical tagged format."""
    parts = [
        f"{UNDERSTANDING_BEGIN}{understanding}{UNDERSTANDING_END}",
        f"{REASONING_BEGIN}{reasoning}{REASONING_END}",
        f"{EMOTION_BEGIN}{emotion}{EMOTION_END}",
        f"{RESPONSE_BEGIN}{response}{RESPONSE_END}",
    ]
    return " ".join(parts)


def extract_span(text: str, begin: str, end: str) -> Optional[str]:
    """Return the content between ``begin`` and ``end`` tags, or ``None``."""
    m = re.search(re.escape(begin) + r"(.*?)" + re.escape(end), text, re.DOTALL)
    if m is None:
        return None
    return m.group(1)


def parse_spans(text: str) -> Dict[str, Optional[str]]:
    """Parse a four-span output into a dict of components.

    Missing spans yield ``None``.  The returned dict is keyed by the span
    names ``understanding`` / ``reasoning`` / ``emotion`` / ``response``.
    """
    result: Dict[str, Optional[str]] = {}
    for name, begin, end in SPANS:
        content = extract_span(text, begin, end)
        result[name] = content.strip() if content is not None else None
    return result


def turn_from_spans(query: str, text: str) -> Turn:
    """Reconstruct a :class:`Turn` from a query and a four-span string."""
    spans = parse_spans(text)
    return Turn(
        query=query,
        understanding=spans["understanding"] or "",
        reasoning=spans["reasoning"] or "",
        emotion=spans["emotion"] or "",
        response=spans["response"] or "",
    )
