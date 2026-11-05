"""Kardia-R1: reference implementation.

Kardia-R1 is a progressive, reasoning-centric framework for empathetic
dialogue generation.  It models empathy as a perspective-taking process with
three explicit steps -- (1) *understanding* the user, (2) *reasoning* about
the emotional context, and (3) *emotion perception* -- before generating a
supportive *response*.  The model is trained with:

  * Stage 1: cold-start supervised fine-tuning (SFT) on "easy" trajectories,
    using a four-span structured output format.
  * Stage 2: Rubric-as-Judge Empathetic Reinforcement Learning (Rubric-ERL),
    a GRPO-based method whose reward is a balanced combination of a
    *format reward*, an *emotion-matching reward*, and a *rubric-as-judge
    reward* produced by an external judge LLM over five interpretable
    psychological criteria.

This package exposes the pure-Python building blocks so they can be reused
and unit-tested independently of the training infrastructure.
"""

from .tags import (
    SPANS,
    SPAN_NAMES,
    UNDERSTANDING_BEGIN,
    UNDERSTANDING_END,
    REASONING_BEGIN,
    REASONING_END,
    EMOTION_BEGIN,
    EMOTION_END,
    RESPONSE_BEGIN,
    RESPONSE_END,
)
from .emotions import EMOTIONS, EMOTION_TO_ID, ID_TO_EMOTION, normalize_emotion
from .data import (
    UserProfile,
    Turn,
    Dialogue,
    spans_to_text,
    parse_spans,
    extract_span,
)
from .partition import is_easy, partition_dialogues
from .rewards import (
    format_violations,
    format_reward,
    emotion_matching_reward,
    normalize_rubric,
    rubric_reward,
    unified_reward,
    DEFAULT_LAMBDAS,
)
from .grpo import compute_advantages, grpo_surrogate_loss, grpo_ratio, k3_kl

__all__ = [
    "SPANS",
    "SPAN_NAMES",
    "UNDERSTANDING_BEGIN",
    "UNDERSTANDING_END",
    "REASONING_BEGIN",
    "REASONING_END",
    "EMOTION_BEGIN",
    "EMOTION_END",
    "RESPONSE_BEGIN",
    "RESPONSE_END",
    "EMOTIONS",
    "EMOTION_TO_ID",
    "ID_TO_EMOTION",
    "normalize_emotion",
    "UserProfile",
    "Turn",
    "Dialogue",
    "spans_to_text",
    "parse_spans",
    "extract_span",
    "is_easy",
    "partition_dialogues",
    "format_violations",
    "format_reward",
    "emotion_matching_reward",
    "normalize_rubric",
    "rubric_reward",
    "unified_reward",
    "DEFAULT_LAMBDAS",
    "compute_advantages",
    "grpo_surrogate_loss",
    "grpo_ratio",
    "k3_kl",
]

__version__ = "0.1.0"
