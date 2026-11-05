"""Reward functions for Rubric-as-Judge Empathetic RL (Rubric-ERL).

The unified per-candidate reward (Eq. 9) is a weighted sum of three terms:

    r_j = λ_f · r_format_j  +  λ_e · r_emo_j  +  λ_r · r_rubric_j

with equal weighting λ_f = λ_e = λ_r = 1/3 (the paper's default).

* ``r_format``  (Eq. 10): structural compliance with the four-span format.
* ``r_emo``     (Eq. 11): deterministic emotion-matching correctness.
* ``r_rubric``  (Eq. 12): the rubric-as-judge score, normalized to [0, 1].
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .tags import SPANS
from .data import extract_span
from .emotions import normalize_emotion

DEFAULT_LAMBDAS = {
    "format": 1 / 3,
    "emotion": 1 / 3,
    "rubric": 1 / 3,
}

# ---------------------------------------------------------------------------
# 1. Format reward (Eq. 10)
# ---------------------------------------------------------------------------
# The paper checks (i) presence of all begin/end tags, (ii) ordering of spans,
# and (iii) non-empty content within each segment.  We enumerate each
# checkable violation; K is the total number of possible violations.
_N_TAG_CHECKS = 2 * len(SPANS)  # begin + end for each span
_N_ORDER_CHECKS = 1
_N_CONTENT_CHECKS = len(SPANS)
K = _N_TAG_CHECKS + _N_ORDER_CHECKS + _N_CONTENT_CHECKS  # 8 + 1 + 4 = 13


def format_violations(output: str) -> List[str]:
    """Return the list of format violations for a candidate ``output``.

    Each violation corresponds to one of the ``K`` constraints checked by the
    format reward.  An empty list means full compliance (reward = 1.0).
    """
    output = output or ""
    violations: List[str] = []

    # (i) presence of every begin/end tag
    for name, begin, end in SPANS:
        if begin not in output:
            violations.append(f"missing_{name}_begin")
        if end not in output:
            violations.append(f"missing_{name}_end")

    # (ii) correct ordering of the four spans
    positions = [output.find(begin) for _, begin, _ in SPANS]
    if any(p == -1 for p in positions) or positions != sorted(positions):
        violations.append("wrong_order")

    # (iii) non-empty content within each segment
    for name, begin, end in SPANS:
        content = extract_span(output, begin, end)
        if content is None or not content.strip():
            violations.append(f"empty_{name}")

    return violations


def format_reward(output: str) -> float:
    """Compute r_format ∈ [0, 1] (Eq. 10)."""
    violations = format_violations(output)
    return 1.0 - len(violations) / K


# ---------------------------------------------------------------------------
# 2. Emotion-matching reward (Eq. 11)
# ---------------------------------------------------------------------------

def emotion_matching_reward(predicted_emotion: str, gold_emotion: str) -> float:
    """Deterministic affect-identification check: 1.0 if correct else 0.0."""
    pred = normalize_emotion(predicted_emotion)
    gold = normalize_emotion(gold_emotion)
    return 1.0 if (pred == gold and pred != "") else 0.0


# ---------------------------------------------------------------------------
# 3. Rubric-as-judge reward (Eq. 12)
# ---------------------------------------------------------------------------

def normalize_rubric(
    scores: Dict[str, float],
    *,
    score_min: int = 1,
    score_max: int = 5,
) -> float:
    """Aggregate five dimension scores and normalize to [0, 1].

    The paper aggregates the five dimension-specific scores and applies a
    ``Norm(·)`` function scaling the result to [0, 1].  We use the mean score
    normalized by the score range (min-max), which equals ``sum / (5·max)``
    when ``score_min = 0``-anchored.  Missing dimensions are treated as the
    minimum score.
    """
    if not scores:
        return 0.0
    dims = list(scores.values())
    n = len(dims)
    total = sum(float(s) for s in dims)
    # Normalize mean into [0, 1] using the Likert range.
    mean = total / n
    return max(0.0, min(1.0, (mean - score_min) / (score_max - score_min)))


def rubric_reward(
    scores: Dict[str, float],
    *,
    score_min: int = 1,
    score_max: int = 5,
) -> float:
    """Compute r_rubric ∈ [0, 1] from judge dimension scores (Eq. 12)."""
    return normalize_rubric(scores, score_min=score_min, score_max=score_max)


# ---------------------------------------------------------------------------
# Unified reward (Eq. 9)
# ---------------------------------------------------------------------------

def unified_reward(
    format_r: float,
    emotion_r: float,
    rubric_r: float,
    *,
    lambdas: Optional[Dict[str, float]] = None,
) -> float:
    """Combine the three reward components (Eq. 9)."""
    lam = lambdas or DEFAULT_LAMBDAS
    return (
        lam["format"] * format_r
        + lam["emotion"] * emotion_r
        + lam["rubric"] * rubric_r
    )
