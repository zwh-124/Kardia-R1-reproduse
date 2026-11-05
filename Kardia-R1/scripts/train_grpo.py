"""Stage 2: Rubric-as-Judge Empathetic RL (Rubric-ERL) via GRPO.

Trains the SFT policy on the *hard* subset D_hard.  For each context the policy
samples N candidates; raw rewards combine format adherence, emotion-matching
correctness, and a rubric-as-judge score (Eq. 9), and GRPO performs within-group
advantage normalization (Eq. 6) with a clipped surrogate + KL anchor to the SFT
model (Eq. 7-8).

Paper hyper-parameters (Section 5.1): lr = 1e-6, N = 8 samples/context,
global batch 32, 2 RL epochs, λ_f = λ_e = λ_r = 1/3, rubric judge = Qwen3-8B.

Usage (with a real judge):
    python scripts/train_grpo.py \
        --model output/kardia-sft \
        --data data/synthetic.jsonl \
        --judge-backend transformers --judge-model Qwen/Qwen3-8B \
        --output output/kardia-r1

Usage (offline smoke-test with the deterministic MockJudge):
    python scripts/train_grpo.py --model output/kardia-sft \
        --data data/synthetic.jsonl --judge-backend mock
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kardia import (
    format_reward,
    emotion_matching_reward,
    rubric_reward,
    unified_reward,
    parse_spans,
)
from kardia import partition_dialogues
from kardia.judge import RubricJudge, make_judge
from kardia.prompts import build_sft_prompt
from scripts.generate_data import load_jsonl


def compute_reward(
    completion: str,
    gold_emotion: str,
    judge: RubricJudge,
    profile,
    context: str,
    lambdas: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute the three reward components and the unified reward (Eq. 9)."""
    spans = parse_spans(completion)
    response = spans.get("response") or ""
    predicted_emotion = spans.get("emotion") or ""

    r_fmt = format_reward(completion)
    r_emo = emotion_matching_reward(predicted_emotion, gold_emotion)

    scores = judge.score(profile, context, response)
    r_rub = rubric_reward(scores)

    r = unified_reward(r_fmt, r_emo, r_rub, lambdas=lambdas)
    return {
        "format": r_fmt,
        "emotion": r_emo,
        "rubric": r_rub,
        "total": r,
        "scores": scores,
    }


def build_grpo_dataset(data_path: str):
    """Build GRPO rows for the hard subset.

    Each row stores the rollout prompt plus the metadata needed by the reward
    function: the gold emotion, the serialized profile, and the user context.
    """
    from kardia import UserProfile

    dialogues = load_jsonl(data_path)
    _easy, hard = partition_dialogues(dialogues)
    rows = []
    for d in hard:
        p = d.profile
        profile_dict = {
            "user_id": p.user_id,
            "mbti": p.mbti,
            "gender": p.gender,
            "relationship": p.relationship,
            "occupation": p.occupation,
            "about": p.about,
            "recent_activities": p.recent_activities,
        }
        for t in d.turns:
            sub = type(d)(
                dialogue_id=d.dialogue_id,
                profile=d.profile,
                situation=d.situation,
                turns=[t],
            )
            rows.append({
                "prompt": build_sft_prompt(sub),
                "gold_emotion": t.emotion,
                "profile": profile_dict,
                "context": t.query,
            })
    return rows


def _profile_from_dict(d: Dict) -> "UserProfile":
    from kardia import UserProfile

    return UserProfile(
        user_id=d.get("user_id", ""),
        mbti=d.get("mbti", ""),
        gender=d.get("gender", ""),
        relationship=d.get("relationship", ""),
        occupation=d.get("occupation", ""),
        about=d.get("about", ""),
        recent_activities=d.get("recent_activities", ""),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="output/kardia-sft")
    ap.add_argument("--data", default="data/synthetic.jsonl")
    ap.add_argument("--output", default="output/kardia-r1")
    ap.add_argument("--judge-backend", default="mock")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--clip-epsilon", type=float, default=0.2)
    args = ap.parse_args()

    rows = build_grpo_dataset(args.data)
    print(f"Loaded {len(rows)} GRPO examples (hard subset).")

    judge = make_judge(backend=args.judge_backend,
                       model_name_or_path=args.judge_model)

    def reward_func(prompts, completions, **kwargs):
        gold_emotions = kwargs.get("gold_emotion", [""] * len(completions))
        profiles = kwargs.get("profile", [{}] * len(completions))
        contexts = kwargs.get("context", [""] * len(completions))
        totals = []
        for comp, gem, prof, ctx in zip(
            completions, gold_emotions, profiles, contexts
        ):
            res = compute_reward(comp, gem, judge, _profile_from_dict(prof), ctx)
            totals.append(res["total"])
        return totals

    _train_trl(args, rows, reward_func)


def _train_trl(args, rows, reward_func) -> None:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    ds = Dataset.from_list(rows)
    grpo_config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        beta=args.beta,
        max_prompt_length=1536,
        max_completion_length=512,
        bf16=True,
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
