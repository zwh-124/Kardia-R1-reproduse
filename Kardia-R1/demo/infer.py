"""End-to-end inference demo for Kardia-R1.

Demonstrates the four-span generation -> reward scoring pipeline.  By default
it uses a deterministic *mock policy* (no model download) so the full flow can
be exercised offline; pass ``--model`` to load a real trained checkpoint
(e.g. the SFT/R1 output) via ``transformers``.

Usage:
    python demo/infer.py                          # offline mock policy
    python demo/infer.py --model output/kardia-r1  # real checkpoint
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kardia import (
    Dialogue,
    Turn,
    UserProfile,
    parse_spans,
    format_reward,
    emotion_matching_reward,
    rubric_reward,
    unified_reward,
)
from kardia.judge import make_judge
from kardia.prompts import build_sft_prompt, build_rubric_prompt


def build_sample_dialogue() -> Dialogue:
    profile = UserProfile(
        user_id="user a008b489",
        mbti="ISFJ",
        gender="Female",
        relationship="Dating",
        occupation="none currently",
        about=("My name is user a008b489. I enjoy speaking in different voices, "
               "singing like Michael Jackson, listening to music, cinnamon "
               "dolce fraps, and volunteering."),
        recent_activities="Commented: '[ISFJ] ISFJ Pet Peeves' — Misspellings.",
    )
    turn = Turn(
        query=("It's so true! Having friends who respect boundaries makes all "
               "the difference. Have you ever had a friend who didn't respect "
               "your boundaries?"),
        understanding=("You're stressing how mutual respect strengthens "
                       "friendships and opening space for shared experiences."),
        reasoning=("Your ISFJ warmth shows in how you value emotional safety "
                   "and open communication."),
        emotion="faithful",
        response=("Absolutely—respect builds trust. Holding your ground in "
                  "those moments really protects what matters."),
    )
    return Dialogue(
        dialogue_id="demo",
        profile=profile,
        situation="A friend's boyfriend showed romantic interest in the user.",
        turns=[turn],
    )


class MockPolicy:
    """Deterministic policy that emits a plausible four-span completion."""

    def generate(self, prompt: str, **kwargs) -> str:
        from kardia import spans_to_text

        return spans_to_text(
            understanding="You value mutual respect and want to connect over "
                          "shared boundary experiences.",
            reasoning="Your ISFJ traits reflect warmth, loyalty, and a desire "
                      "for emotional safety.",
            emotion="faithful",
            response="Absolutely—respect builds trust, and it's okay to guard "
                     "your boundaries.",
        )


class TransformersPolicy:
    def __init__(self, model_name_or_path: str, temperature: float = 0.0):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto", torch_dtype="auto"
        )
        self.temperature = temperature

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            do_sample=self.temperature > 0.0,
            temperature=self.temperature if self.temperature > 0.0 else None,
        )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="Path/HF id of a trained checkpoint; omit for mock.")
    ap.add_argument("--judge-backend", default="mock")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    args = ap.parse_args()

    dialogue = build_sample_dialogue()
    turn = dialogue.turns[-1]

    policy = (TransformersPolicy(args.model) if args.model else MockPolicy())
    judge = make_judge(backend=args.judge_backend,
                       model_name_or_path=args.judge_model)

    prompt = build_sft_prompt(dialogue)
    print("=" * 70)
    print("PROMPT (context c)")
    print("=" * 70)
    print(prompt)
    print()

    completion = policy.generate(prompt)
    print("=" * 70)
    print("GENERATED COMPLETION (four spans)")
    print("=" * 70)
    print(completion)
    print()

    spans = parse_spans(completion)
    r_fmt = format_reward(completion)
    r_emo = emotion_matching_reward(spans["emotion"] or "", turn.emotion)
    scores = judge.score(dialogue.profile, turn.query, spans["response"] or "")
    r_rub = rubric_reward(scores)
    total = unified_reward(r_fmt, r_emo, r_rub)

    print("=" * 70)
    print("REWARDS (Eq. 9, λ = 1/3 each)")
    print("=" * 70)
    print(f"  format  reward : {r_fmt:.4f}")
    print(f"  emotion reward : {r_emo:.4f}  (predicted={spans['emotion']!r}, "
          f"gold={turn.emotion!r})")
    print(f"  rubric  reward : {r_rub:.4f}  scores={scores}")
    print(f"  TOTAL  reward  : {total:.4f}")
    print()

    print("=" * 70)
    print("RUBRIC JUDGE PROMPT (for reference)")
    print("=" * 70)
    print(build_rubric_prompt(dialogue.profile, turn.query,
                              spans["response"] or ""))


if __name__ == "__main__":
    main()
