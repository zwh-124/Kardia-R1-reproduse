"""Difficulty-aware data partitioning (Eq. 4 of the paper).

During KardiaBench synthesis every turn receives a rubric decision
``d_t ∈ {FAIL, PASS, SOLVED}``.  For each trajectory we estimate the empirical
resolution rates ``(p_fail, p_pass, p_solved)`` and assign it to the *easy*
set ``D_easy`` (routed to cold-start SFT) when failure is *not* dominant, or
to the *hard* set ``D_hard`` (routed to GRPO) otherwise.

    D_easy = { p | p_fail < p_pass + p_solved  ∨  p_fail + p_pass < p_solved }

Intuitively, lower failure dominance corresponds to more stable and coherent
resolution patterns, so those trajectories seed SFT, while harder ones -- where
the model must discover a good policy -- feed empathetic RL.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from .data import Dialogue


def is_easy(p_fail: float, p_pass: float, p_solved: float) -> bool:
    """Return True if a trajectory is classified as easy (Eq. 4)."""
    cond1 = p_fail < p_pass + p_solved
    cond2 = p_fail + p_pass < p_solved
    return cond1 or cond2


def partition_dialogues(
    dialogues: Iterable[Dialogue],
) -> Tuple[List[Dialogue], List[Dialogue]]:
    """Split dialogues into (easy, hard) subsets using Eq. 4.

    Dialogues missing difficulty statistics (all rates zero) are treated as
    *hard* by default, since routing unlabelled data to RL follows the
    paper's conservative choice of not biasing SFT.
    """
    easy: List[Dialogue] = []
    hard: List[Dialogue] = []
    for d in dialogues:
        if is_easy(d.p_fail, d.p_pass, d.p_solved):
            easy.append(d)
        else:
            hard.append(d)
    return easy, hard
