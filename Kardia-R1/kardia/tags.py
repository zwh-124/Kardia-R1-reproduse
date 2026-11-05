"""Four-span structural tags for Kardia-R1's progressive empathy format.

Each assistant turn is decomposed into four spans, always generated in this
fixed order:

  1. ``<|understanding_begin|> ... <|understanding_end|>``
       -- summarize the user's intent and emotional cues.
  2. ``<|reasoning_begin|> ... <|reasoning_end|>``
       -- articulate the internal appraisal / perspective-taking reasoning.
  3. ``<|emotion_begin|> ... <|emotion_end|>``
       -- predict an explicit affect label from the vocabulary.
  4. ``<|response_begin|> ... <|response_end|>``
       -- produce the final empathetic reply.

The tags act both as a *scaffold* during SFT (disentangling the empathetic
components) and as a *verifiable structure* during RL (the format reward
checks tag presence, ordering and non-empty content -- see :mod:`rewards`).
"""

UNDERSTANDING_BEGIN = "<|understanding_begin|>"
UNDERSTANDING_END = "<|understanding_end|>"
REASONING_BEGIN = "<|reasoning_begin|>"
REASONING_END = "<|reasoning_end|>"
EMOTION_BEGIN = "<|emotion_begin|>"
EMOTION_END = "<|emotion_end|>"
RESPONSE_BEGIN = "<|response_begin|>"
RESPONSE_END = "<|response_end|>"

# Ordered span definitions: (name, begin_tag, end_tag).  Order matters: the
# format reward verifies that spans appear in exactly this order.
SPANS = [
    ("understanding", UNDERSTANDING_BEGIN, UNDERSTANDING_END),
    ("reasoning", REASONING_BEGIN, REASONING_END),
    ("emotion", EMOTION_BEGIN, EMOTION_END),
    ("response", RESPONSE_BEGIN, RESPONSE_END),
]

SPAN_NAMES = [name for name, _, _ in SPANS]
