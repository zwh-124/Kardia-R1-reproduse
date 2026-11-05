"""Central configuration for the Kardia-R1 reproduction.

Hyper-parameters follow Section 5.1 of the paper:

  * Stage 1 (SFT):   AdamW, lr = 1e-4, global batch 128, 2 epochs.
  * Stage 2 (RL):    GRPO, lr = 1e-6, global batch 32, N = 8 samples/context,
                     2 RL epochs.
  * Reward weights:  lambda_f = lambda_e = lambda_r = 1/3 (Eq. 9).
  * Judge:           Qwen3-8B (rubric-as-judge), Qwen3-Embedding-0.6B
                     (embedding reward), Skywork-Reward-V2-Llama-3.1-8B (RLHF).
  * Inference:       temperature = 0.0 (deterministic).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SFTConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "output/kardia-sft"
    num_train_epochs: int = 2
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    global_batch_size: int = 128  # 8 devices x 8 x 2 (paper reports 128)
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    logging_steps: int = 10
    save_strategy: str = "epoch"


@dataclass
class GRPOConfig:
    model_name_or_path: str = "output/kardia-sft"
    ref_model_name_or_path: str = "output/kardia-sft"  # SFT model = KL anchor
    output_dir: str = "output/kardia-r1"
    num_epochs: int = 2
    learning_rate: float = 1e-6
    num_generations: int = 8  # N candidate responses per context
    global_batch_size: int = 32
    max_prompt_length: int = 1536
    max_completion_length: int = 512
    temperature: float = 1.0          # sampling temperature during RL rollout
    clip_epsilon: float = 0.2         # epsilon in Eq. 7
    beta: float = 0.04                # KL coefficient in Eq. 7
    # Reward weights (Eq. 9): lambda_f, lambda_e, lambda_r
    lambda_format: float = 1 / 3
    lambda_emotion: float = 1 / 3
    lambda_rubric: float = 1 / 3
    bf16: bool = True


@dataclass
class JudgeConfig:
    # Rubric-as-judge LLM (paper uses Qwen3-8B).
    judge_model_name_or_path: str = "Qwen/Qwen3-8B"
    # Five rubric dimensions, each scored on a 1..5 Likert scale.
    rubric_dimensions: tuple = (
        "relevance",
        "fluency",
        "empathy",
        "persona_consistency",
        "safety",
    )
    score_min: int = 1
    score_max: int = 5
    temperature: float = 0.0
    max_new_tokens: int = 512


@dataclass
class KardiaConfig:
    sft: SFTConfig = field(default_factory=SFTConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    # Backbones reported in the paper: Qwen2.5-3B/7B-Instruct, Gemma-2B/7B.
    backbones: tuple = (
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "google/gemma-2-2b-it",
        "google/gemma-2-7b-it",
    )
    inference_temperature: float = 0.0  # paper uses 0.0 for deterministic eval
