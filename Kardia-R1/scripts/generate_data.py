"""Generate a small synthetic KardiaBench-style dataset.

The real KardiaBench (178,080 QA pairs across 22,080 dialogues anchored to 671
real PersonalityCafe profiles) is released under a restricted, research-only
agreement and is not bundled here.  This script synthesizes a *toy* corpus with
the exact same schema -- user profile + situation + multi-turn four-span
trajectories + difficulty stats (p_fail / p_pass / p_solved) -- so the
two-stage pipeline (SFT -> GRPO) can be exercised end-to-end offline.

Usage:
    python scripts/generate_data.py --num-dialogues 100 --out data/synthetic.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kardia import EMOTIONS, Dialogue, Turn, UserProfile
from kardia import partition_dialogues

# A tiny pool of synthetic (but psychologically-plausible) profiles.
_PROFILES = [
    ("INFP", "Female", "Single", "writer", "loves quiet mornings and poetry"),
    ("ISFJ", "Female", "Dating", "nurse", "values loyalty and helping others"),
    ("ENTJ", "Male", "Married", "engineer", "goal-driven and analytical"),
    ("ENFP", "Female", "Single", "student", "curious, warm, easily excited"),
    ("ISTJ", "Male", "Married", "accountant", "reliable, reserved, orderly"),
    ("INTP", "Male", "Single", "researcher", "introspective and abstract"),
    ("ESFJ", "Female", "Married", "teacher", "sociable and conscientious"),
    ("INFJ", "Female", "Single", "counselor", "insightful and idealistic"),
]

_SITUATIONS = [
    "Waiting for good news about a job application",
    "A close friend moved to another city",
    "Received unexpected praise at work",
    "An argument with a family member",
    "Worried about an upcoming exam",
    "Feeling lonely on a weekend",
    "Excited about a new hobby",
    "Overwhelmed by too many commitments",
]

_UNDERSTANDING = [
    "You're sharing a {emotion} experience and seeking connection",
    "You're reflecting on a situation that feels {emotion}",
    "You want to be heard about something {emotion}",
]

_REASONING = [
    "Your {mbti} nature shapes how you process this; you value being understood",
    "Given your {mbti} traits, you likely need validation before advice",
    "As an {mbti}, you may appreciate gentle acknowledgment over quick fixes",
]

_RESPONSE = [
    "That sounds {emotion} to carry. It makes sense you'd feel that way.",
    "Thank you for sharing that. It's completely okay to feel {emotion}.",
    "I can hear how {emotion} this is for you. You're not alone in it.",
]


def _fmt(template: str, **kw) -> str:
    return template.format(**kw)


def make_profile(i: int) -> UserProfile:
    mbti, gender, rel, occ, about = _PROFILES[i % len(_PROFILES)]
    return UserProfile(
        user_id=f"user_{i:06d}",
        mbti=mbti,
        gender=gender,
        relationship=rel,
        occupation=occ,
        about=about,
    )


def make_dialogue(i: int, rng: random.Random, n_turns: int) -> Dialogue:
    profile = make_profile(i)
    situation = rng.choice(_SITUATIONS)
    turns: List[Turn] = []
    emotion = rng.choice(EMOTIONS)

    for t in range(n_turns):
        query = (
            f"{situation} (turn {t + 1}). "
            f"Do you think it's normal to feel {emotion} about this?"
        )
        turn = Turn(
            query=query,
            understanding=_fmt(rng.choice(_UNDERSTANDING), emotion=emotion),
            reasoning=_fmt(rng.choice(_REASONING), mbti=profile.mbti),
            emotion=emotion,
            response=_fmt(rng.choice(_RESPONSE), emotion=emotion),
        )
        turns.append(turn)
        emotion = rng.choice(EMOTIONS)  # drift emotion across turns

    if i % 2 == 0:
        p_fail, p_pass, p_solved = 0.15, 0.35, 0.50  # easy
    else:
        p_fail, p_pass, p_solved = 0.60, 0.25, 0.15  # hard

    return Dialogue(
        dialogue_id=f"d{i:06d}",
        profile=profile,
        situation=situation,
        turns=turns,
        p_fail=p_fail,
        p_pass=p_pass,
        p_solved=p_solved,
    )



def dialogue_to_dict(d: Dialogue) -> Dict:
    return {
        "dialogue_id": d.dialogue_id,
        "profile": {
            "user_id": d.profile.user_id,
            "mbti": d.profile.mbti,
            "gender": d.profile.gender,
            "relationship": d.profile.relationship,
            "occupation": d.profile.occupation,
            "about": d.profile.about,
        },
        "situation": d.situation,
        "p_fail": d.p_fail,
        "p_pass": d.p_pass,
        "p_solved": d.p_solved,
        "turns": [
            {
                "query": t.query,
                "understanding": t.understanding,
                "reasoning": t.reasoning,
                "emotion": t.emotion,
                "response": t.response,
                "spans": t.to_spans(),
            }
            for t in d.turns
        ],
    }


def dialogue_from_dict(obj: Dict) -> Dialogue:
    p = obj["profile"]
    profile = UserProfile(
        user_id=p.get("user_id", ""),
        mbti=p.get("mbti", ""),
        gender=p.get("gender", ""),
        relationship=p.get("relationship", ""),
        occupation=p.get("occupation", ""),
        about=p.get("about", ""),
    )
    turns = [
        Turn(
            query=t.get("query", ""),
            understanding=t.get("understanding", ""),
            reasoning=t.get("reasoning", ""),
            emotion=t.get("emotion", ""),
            response=t.get("response", ""),
        )
        for t in obj.get("turns", [])
    ]
    return Dialogue(
        dialogue_id=obj.get("dialogue_id", ""),
        profile=profile,
        situation=obj.get("situation", ""),
        turns=turns,
        p_fail=float(obj.get("p_fail", 0.0)),
        p_pass=float(obj.get("p_pass", 0.0)),
        p_solved=float(obj.get("p_solved", 0.0)),
    )


def save_jsonl(dialogues: List[Dialogue], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps(dialogue_to_dict(d), ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dialogue]:
    dialogues = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dialogues.append(dialogue_from_dict(json.loads(line)))
    return dialogues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-dialogues", type=int, default=100)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/synthetic.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    dialogues = [
        make_dialogue(i, rng, args.turns) for i in range(args.num_dialogues)
    ]

    easy, hard = partition_dialogues(dialogues)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_jsonl(dialogues, args.out)

    print(f"Wrote {len(dialogues)} dialogues -> {args.out}")
    print(f"  easy (SFT):   {len(easy)}")
    print(f"  hard (GRPO):  {len(hard)}")
    print("\nExample four-span turn:\n")
    print(dialogues[0].turns[0].to_spans())


if __name__ == "__main__":
    main()
