"""Stage 1: cold-start supervised fine-tuning (SFT) on the *easy* subset.

Trains a base LLM to produce the four-span structured output
(understanding -> reasoning -> emotion -> response) using context-response
pairs from D_easy.  Objective is the standard autoregressive log-likelihood
(Eq. 5):

    L_SFT(θ) = -E_{(c, y) ~ D_easy}[ log πθ(y | c) ]

Paper hyper-parameters (Section 5.1): AdamW, lr = 1e-4, global batch 128,
2 epochs, on 8 A100 GPUs.

Usage:
    python scripts/train_sft.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --data data/synthetic.jsonl \
        --output output/kardia-sft
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kardia import partition_dialogues
from kardia.prompts import build_sft_prompt
from scripts.generate_data import load_jsonl


def build_sft_examples(data_path: str):
    """Yield (prompt, completion) pairs for the easy subset.

    The prompt is the SFT context ``c`` (profile + situation + history ending
    with the user query); the completion is the target four-span turn.
    """
    dialogues = load_jsonl(data_path)
    easy, _hard = partition_dialogues(dialogues)
    examples = []
    for d in easy:
        for t in d.turns:
            # Build a single-turn context: history up to (not including) t.
            sub = type(d)(
                dialogue_id=d.dialogue_id,
                profile=d.profile,
                situation=d.situation,
                turns=[t],
                p_fail=d.p_fail,
                p_pass=d.p_pass,
                p_solved=d.p_solved,
            )
            prompt = build_sft_prompt(sub)
            completion = t.to_spans()
            examples.append({"prompt": prompt, "completion": completion})
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="data/synthetic.jsonl")
    ap.add_argument("--output", default="output/kardia-sft")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--use-trl", action="store_true",
                    help="Use TRL's SFTTrainer (requires trl); "
                         "otherwise run a plain HF training loop.")
    args = ap.parse_args()

    examples = build_sft_examples(args.data)
    print(f"Loaded {len(examples)} SFT examples (easy subset).")

    if not args.use_trl:
        _train_plain_hf(args, examples)
    else:
        _train_trl(args, examples)


def _train_plain_hf(args, examples) -> None:
    """Reference training loop using bare `transformers` (no TRL)."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    def tokenize(ex):
        full = ex["prompt"] + "\n" + ex["completion"]
        tok = tokenizer(
            full, truncation=True, max_length=args.max_length,
        )
        # Label only the completion tokens (mask the prompt).
        prompt_tok = tokenizer(
            ex["prompt"], truncation=True, max_length=args.max_length,
        )
        plen = len(prompt_tok["input_ids"])
        labels = tok["input_ids"].copy()
        labels[:plen] = [-100] * plen
        tok["labels"] = labels
        return tok

    ds = Dataset.from_list(examples).map(tokenize, remove_columns=["prompt", "completion"])

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.output)


def _train_trl(args, examples) -> None:
    """Reference training loop using TRL's SFTTrainer."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    def formatting(ex):
        return {
            "prompt": ex["prompt"],
            "completion": ex["completion"],
        }

    ds = Dataset.from_list(examples).map(formatting)

    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_length,
        bf16=True,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=args.model,
        args=sft_config,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
