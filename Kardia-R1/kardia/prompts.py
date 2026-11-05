"""Prompt templates for Kardia-R1.

Two families of prompts are defined:

1. **SFT / rollout prompt** -- the context ``c`` fed to the policy, which
   bundles the user profile, the emotional situation, and the dialogue
   history, and asks the model to produce the four-span structured output.

2. **Rubric-as-judge prompt** -- the instruction given to the external judge
   LLM (``LLM_rubric``, Qwen3-8B in the paper) to score a candidate response
   along five interpretable psychological criteria (relevance, fluency,
   empathy, persona consistency, safety).
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from .data import Dialogue, Turn, UserProfile
from .tags import (
    UNDERSTANDING_BEGIN,
    UNDERSTANDING_END,
    REASONING_BEGIN,
    REASONING_END,
    EMOTION_BEGIN,
    EMOTION_END,
    RESPONSE_BEGIN,
    RESPONSE_END,
)

# ---------------------------------------------------------------------------
# SFT / rollout prompt
# ---------------------------------------------------------------------------

SFT_SYSTEM_PROMPT = (
    "You are an empathetic conversational agent. Before replying, you follow a "
    "progressive perspective-taking process: first, summarize the user's "
    "intent and emotional cues (understanding); then articulate your internal "
    "appraisal of their situation and personality (reasoning); then predict "
    "their current emotional state (emotion); finally, produce an attuned and "
    "validating reply (response). Always output all four spans in order, "
    "delimited by the following tags:\n"
    f"{UNDERSTANDING_BEGIN}...{UNDERSTANDING_END} "
    f"{REASONING_BEGIN}...{REASONING_END} "
    f"{EMOTION_BEGIN}...{EMOTION_END} "
    f"{RESPONSE_BEGIN}...{RESPONSE_END}"
)


def render_history(turns: List[Turn], max_turns: Optional[int] = None) -> str:
    """Render the dialogue history as a readable transcript.

    Past assistant turns are shown in their full four-span form so the model
    can track the evolving affective state; the final turn contains only the
    user's query (the assistant is expected to complete it).
    """
    history: List[str] = []
    turns_to_show = turns if max_turns is None else turns[-max_turns:]
    for t in turns_to_show:
        if t.query:
            history.append(f"User: {t.query}")
        if t.understanding or t.reasoning or t.emotion or t.response:
            history.append(f"Assistant: {t.to_spans()}")
    return "\n".join(history)


def build_sft_prompt(
    dialogue: Dialogue,
    *,
    include_last_assistant: bool = False,
    max_history_turns: Optional[int] = None,
) -> str:
    """Build the context ``c`` for SFT training / RL rollout.

    The prompt includes the user profile and situation, followed by the
    dialogue history.  By default the final assistant turn (the one being
    predicted) is represented only by its user query.
    """
    profile_text = dialogue.profile.to_text()

    # History excludes the final turn's assistant spans (the target).
    history_turns = list(dialogue.turns)
    if history_turns and not include_last_assistant:
        last = history_turns[-1]
        history_turns = history_turns[:-1]
        if last.query:
            history_turns.append(Turn(query=last.query))

    history_text = render_history(history_turns, max_turns=max_history_turns)

    parts = [
        SFT_SYSTEM_PROMPT,
        "",
        "### User Profile",
        profile_text,
        "",
        "### Situation",
        dialogue.situation or "(not provided)",
        "",
        "### Conversation",
        history_text,
        "",
        "### Assistant",
    ]
    return "\n".join(parts)



# ---------------------------------------------------------------------------
# Rubric-as-judge prompt
# ---------------------------------------------------------------------------

RUBRIC_JUDGE_SYSTEM = (
    "You are an expert psychological evaluator for empathetic dialogue. You "
    "assess the quality of a supportive response along five criteria, each "
    "scored on an integer scale from 1 (very poor) to 5 (excellent). "
    "Respond ONLY with a JSON object of the form "
    '{"relevance": int, "fluency": int, "empathy": int, '
    '"persona_consistency": int, "safety": int}.'
)

RUBRIC_DIMENSION_DESCRIPTIONS = {
    "relevance": (
        "Relevance to user intent and emotional need: whether the response "
        "accurately identifies the user's goal and addresses both explicit "
        "and implicit emotional cues."
    ),
    "fluency": (
        "Fluency and clarity: linguistic coherence, readability, and the "
        "logical flow of the reasoning and response."
    ),
    "empathy": (
        "Empathy: the degree to which the response reflects the user's affect "
        "and provides appropriate acknowledgment and emotional support."
    ),
    "persona_consistency": (
        "Persona consistency: alignment with the target stylistic identity, "
        "including tone, voice, and role-specific behavioral norms."
    ),
    "safety": (
        "Safety: the absence of harmful content, ethical violations, or "
        "advice that may lead to psychological or physical risk."
    ),
}


def build_rubric_prompt(
    profile: UserProfile,
    context: str,
    response: str,
    *,
    dimensions: Optional[List[str]] = None,
) -> str:
    """Build the rubric-as-judge prompt for scoring a candidate response.

    ``context`` is the user's latest utterance / situation; ``response`` is the
    extracted ``<|response_begin|>...<|response_end|>`` span to be judged.
    """
    dims = dimensions or list(RUBRIC_DIMENSION_DESCRIPTIONS.keys())
    criteria_lines = []
    for i, dim in enumerate(dims, 1):
        desc = RUBRIC_DIMENSION_DESCRIPTIONS.get(dim, dim)
        criteria_lines.append(f"{i}. {dim}: {desc}")

    parts = [
        RUBRIC_JUDGE_SYSTEM,
        "",
        "### Evaluation criteria",
        *criteria_lines,
        "",
        "### User profile",
        profile.to_text(),
        "",
        "### Conversation context",
        context,
        "",
        "### Candidate response (to evaluate)",
        response,
        "",
        "### Your scores (JSON only)",
    ]
    return "\n".join(parts)


def parse_rubric_output(text: str) -> dict:
    """Robustly parse the judge LLM's JSON output into a score dict."""
    text = (text or "").strip()
    # Extract the first JSON object, tolerating surrounding prose / fences.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = {}
        for key in RUBRIC_DIMENSION_DESCRIPTIONS:
            m = re.search(rf'["\']?{key}["\']?\s*[:=]\s*(-?\d+(?:\.\d+)?)', text)
            if m:
                obj[key] = float(m.group(1))
    return obj
