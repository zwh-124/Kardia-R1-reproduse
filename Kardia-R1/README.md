# Kardia-R1

**Kardia-R1: Unleashing LLMs to Reason toward Understanding and Empathy for
Emotional Support via Rubric-as-Judge Reinforcement Learning**

> Jiahao Yuan, Zhiqing Cui, Hanqing Wang, Yuansheng Gao, Yucheng Zhou,
> Usman Naseem (2025)

This repository is a **faithful, self-contained reference implementation** of
the two-stage training framework described in the paper. It reproduces the
four-span structured output, the difficulty-aware data partitioning, the three
reward components of *Rubric-as-Judge Empathetic Reinforcement Learning*
(Rubric-ERL), and the GRPO training objective.

---

## 1. What the paper proposes

Kardia-R1 models **empathy as a progressive perspective-taking process** with
three explicit reasoning steps, produced as four tagged spans in a fixed order:

| Span        | Tags                                             | Role                          |
|-------------|--------------------------------------------------|-------------------------------|
| 1. understanding | `<|understanding_begin|>…<|understanding_end|>` | user intent + emotional cues  |
| 2. reasoning     | `<|reasoning_begin|>…<|reasoning_end|>`         | internal appraisal            |
| 3. emotion       | `<|emotion_begin|>…<|emotion_end|>`             | explicit affect label         |
| 4. response      | `<|response_begin|>…<|response_end|>`           | empathetic reply              |

Training is a **two-stage pipeline**:

1. **Cold-start SFT** on the *easy* subset `D_easy` (Eq. 4/5).
2. **Rubric-ERL** (GRPO) on the *hard* subset `D_hard` (Eq. 6–9), optimizing a
   unified reward that combines three signals with equal weighting
   `λ_f = λ_e = λ_r = 1/3`:

   - **Format reward** (Eq. 10) — four-span structural compliance.
   - **Emotion-matching reward** (Eq. 11) — deterministic affect-label check.
   - **Rubric-as-judge reward** (Eq. 12) — an external judge LLM (Qwen3-8B)
     scores the response along five interpretable criteria (relevance,
     fluency, empathy, persona consistency, safety) and the aggregate is
     normalized to `[0, 1]`.

---

## 2. Repository structure

```
Kardia-R1/
├── kardia/                    # pure-Python core (no heavy deps, unit-testable)
│   ├── tags.py                # four-span tag constants
│   ├── emotions.py            # 32-label emotion vocabulary
│   ├── data.py                # UserProfile / Turn / Dialogue + span (de)serialization
│   ├── prompts.py             # SFT prompt + rubric-as-judge prompt
│   ├── partition.py           # difficulty-aware partitioning (Eq. 4)
│   ├── rewards.py             # format / emotion / rubric / unified reward (Eq. 9–12)
│   ├── grpo.py                # GRPO advantages & clipped surrogate loss (Eq. 6–8)
│   ├── judge.py               # rubric judge (Mock / transformers / OpenAI-compatible)
│   └── config.py              # paper hyper-parameters
├── scripts/
│   ├── generate_data.py       # synthetic KardiaBench-style data (toy corpus)
│   ├── train_sft.py           # Stage 1: cold-start SFT
│   └── train_grpo.py          # Stage 2: Rubric-ERL (GRPO)
├── demo/
│   └── infer.py               # four-span generation + reward scoring demo
├── tests/                     # unit tests for the core modules
├── requirements.txt
└── README.md
```

---

## 3. Installation

The core algorithm modules only need `numpy`:

```bash
pip install numpy
```

To run the full training pipeline (SFT + GRPO), additionally install:

```bash
pip install torch transformers datasets accelerate trl
# optional: vllm, ms-swift  (the original paper uses Ms-Swift + vLLM)
```

---

## 4. Quick start (offline, no GPU required)

Everything below runs with the deterministic `MockJudge` and `MockPolicy`, so
you can validate the full flow without downloading a model.

```bash
# 1. Generate a toy KardiaBench-style corpus
python scripts/generate_data.py --num-dialogues 100 --out data/synthetic.jsonl

# 2. Run the unit tests (validates rewards, GRPO math, partitioning, prompts)
python -m unittest discover -s tests -v

# 3. Run the inference demo (four-span generation -> reward scoring)
python demo/infer.py
```


## 5. Training the full model

### Stage 1 — Cold-start SFT (Eq. 5)

```bash
python scripts/train_sft.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --data data/synthetic.jsonl \
    --output output/kardia-sft \
    --use-trl
```

Paper hyper-parameters (already the defaults in `config.py`): AdamW,
`lr = 1e-4`, global batch 128, 2 epochs.

### Stage 2 — Rubric-ERL / GRPO (Eq. 6–9)

```bash
# Offline smoke-test with the deterministic judge:
python scripts/train_grpo.py \
    --model output/kardia-sft \
    --data data/synthetic.jsonl \
    --judge-backend mock \
    --output output/kardia-r1

# Production: use the Qwen3-8B rubric judge (as in the paper):
python scripts/train_grpo.py \
    --model output/kardia-sft \
    --data data/synthetic.jsonl \
    --judge-backend transformers --judge-model Qwen/Qwen3-8B \
    --output output/kardia-r1
```

Paper hyper-parameters: `lr = 1e-6`, `N = 8` samples/context, global batch 32,
2 RL epochs, `β = 0.04`, `ε = 0.2`, `λ = 1/3` each.

---

## 6. Key equations implemented

| Equation | Location | Meaning |
|----------|----------|---------|
| Eq. 4  | `partition.py` | difficulty-aware split into `D_easy` / `D_hard` |
| Eq. 5  | `train_sft.py`  | autoregressive SFT objective |
| Eq. 6  | `grpo.compute_advantages` | group-wise z-score advantage |
| Eq. 7  | `grpo.grpo_surrogate_loss` | clipped surrogate + KL anchor |
| Eq. 8  | `grpo.grpo_ratio` | importance ratio `πθ/πθ0` |
| Eq. 9  | `rewards.unified_reward` | weighted reward combination |
| Eq. 10 | `rewards.format_reward` | format compliance `[0,1]` |
| Eq. 11 | `rewards.emotion_matching_reward` | deterministic affect check |
| Eq. 12 | `rewards.rubric_reward` | normalized rubric-as-judge score |

---

## 7. Important caveats

- **Data.** The real KardiaBench (178,080 QA pairs anchored to 671
  PersonalityCafe profiles) is released under a restricted, research-only
  agreement and is **not** bundled here. `scripts/generate_data.py` produces a
  *toy* corpus with the identical schema so the pipeline can be exercised;
  replace it with the official KardiaBench (or your own persona-grounded data)
  for real training.
- **Judge.** The paper uses **Qwen3-8B** as the rubric judge. The bundled
  `MockJudge` is a deterministic, model-free placeholder for offline testing.
  Point `--judge-model` at a real Qwen3-8B checkpoint for production.
- **Infrastructure.** The paper uses Ms-Swift + vLLM on 8×A100. The scripts
  here are written against `transformers` + `trl` for portability; the core
  math is identical.

---

## 8. Reference

Yuan, J., Cui, Z., Wang, H., Gao, Y., Zhou, Y., & Naseem, U. (2025).
*Kardia-R1: Unleashing LLMs to Reason toward Understanding and Empathy for
Emotional Support via Rubric-as-Judge Reinforcement Learning.*
arXiv preprint.


Expected demo output shows the four-span completion and the three reward
components plus the combined reward (Eq. 9).
