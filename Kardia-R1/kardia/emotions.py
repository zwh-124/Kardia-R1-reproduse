"""Emotion label vocabulary used by Kardia-R1.

KardiaBench preserves the empirical emotion distribution of
EmpatheticDialogues (Rashkin et al., 2019), which uses 32 affect categories.
This module defines that vocabulary plus a small normalisation helper.

NOTE: the exact label set is *configurable* -- if you train on your own
persona-grounded data you should replace :data:`EMOTIONS` with the label
vocabulary of your corpus.  The emotion-matching reward (Eq. 11) only
requires a deterministic ``predicted == gold`` comparison, so the vocabulary
only needs to be consistent between data and evaluation.
"""

# 32 emotion categories following EmpatheticDialogues.
EMOTIONS = [
    "afraid",
    "angry",
    "annoyed",
    "anxious",
    "ashamed",
    "caring",
    "confident",
    "content",
    "devastated",
    "disappointed",
    "disgusted",
    "embarrassed",
    "excited",
    "faithful",
    "frustrated",
    "grateful",
    "guilty",
    "hopeful",
    "impressed",
    "jealous",
    "joyful",
    "lonely",
    "nostalgic",
    "prepared",
    "proud",
    "sad",
    "sentimental",
    "surprised",
    "terrified",
    "trusting",
    "amused",
    "cheerful",
]

EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTIONS)}
ID_TO_EMOTION = {i: e for i, e in enumerate(EMOTIONS)}


def normalize_emotion(label: str) -> str:
    """Normalize a free-text emotion label to a canonical vocabulary entry.

    Performs trimming / lower-casing / punctuation stripping and returns the
    vocabulary entry on an exact match.  If no exact match is found the
    cleaned string is returned unchanged so the caller can decide how to
    handle out-of-vocabulary labels (e.g. treat them as a mismatch).
    """
    if label is None:
        return ""
    s = label.strip().lower().strip(".,;:!?'\"()[]{}")
    if s in EMOTION_TO_ID:
        return s
    return s
