"""GRPO math (Eq. 6--8 of the paper).

Group Relative Policy Optimization performs *within-group* reward
normalization: for a context ``c`` the policy samples ``N`` candidates
``o_1..o_N``, each receiving a raw reward ``r_j``.  Raw rewards are turned
into advantages by z-scoring *within the group* (Eq. 6), and the policy is
updated with a clipped surrogate objective plus a KL anchor to the reference
(SFT) model (Eq. 7--8).

This module provides the pure mathematical pieces so they can be unit-tested
independently of any trainer; :mod:`scripts.train_grpo` wires them into a
full training loop.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

_EPS = 1e-8


def compute_advantages(rewards: List[float], eps: float = _EPS) -> np.ndarray:
    """Group-wise z-score normalization of raw rewards (Eq. 6).

    A_j = (r_j - μ_r) / (σ_r + ε)

    where μ_r and σ_r are the mean and (population) standard deviation of the
    ``N`` rewards within the group.
    """
    r = np.asarray(rewards, dtype=np.float64)
    if r.size == 0:
        return np.empty(0)
    mu = r.mean()
    sigma = r.std()  # population std, matching Eq. 6 (divide by N)
    return (r - mu) / (sigma + eps)


def grpo_ratio(logprobs: np.ndarray, ref_logprobs: np.ndarray) -> np.ndarray:
    """Importance ratio ρ_j = πθ(o_j|c) / πθ0(o_j|c) (Eq. 8).

    ``logprobs`` and ``ref_logprobs`` are *sequence-level* log-probabilities
    of each candidate under the current policy and the reference (SFT) model.
    """
    return np.exp(np.asarray(logprobs) - np.asarray(ref_logprobs))


def k3_kl(logprobs: np.ndarray, ref_logprobs: np.ndarray) -> np.ndarray:
    """Unbiased (k3) estimator of D_KL(πθ || πθ0) used for the KL penalty.

    kl = exp(ref - logp) - (ref - logp) - 1

    This is non-negative and matches the standard estimator used in
    R1-style GRPO implementations.  Set ``beta = 0`` in the loss to disable.
    """
    ref = np.asarray(ref_logprobs)
    logp = np.asarray(logprobs)
    diff = ref - logp
    return np.exp(diff) - diff - 1.0


def grpo_surrogate_loss(
    logprobs: np.ndarray,
    ref_logprobs: np.ndarray,
    advantages: np.ndarray,
    *,
    clip_epsilon: float = 0.2,
    beta: float = 0.04,
    use_k3_kl: bool = True,
) -> float:
    """Per-candidate GRPO loss (negative of Eq. 7), averaged over the group.

    Returns the mean loss ``L = -(1/N) Σ_j [ clipped surrogate - β·KL_j ]``,
    which is minimized by gradient descent.  All inputs are sequence-level
    ``numpy`` arrays of length ``N``.
    """
    logp = np.asarray(logprobs, dtype=np.float64)
    ref = np.asarray(ref_logprobs, dtype=np.float64)
    adv = np.asarray(advantages, dtype=np.float64)

    ratio = grpo_ratio(logp, ref)
    clipped = np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = np.minimum(ratio * adv, clipped * adv)  # Eq. 7 inner min

    if beta > 0.0:
        if use_k3_kl:
            kl = k3_kl(logp, ref)
        else:
            # Simple approximation: D_KL ≈ ref_logp - logp
            kl = ref - logp
        loss = -(surrogate - beta * kl)
    else:
        loss = -surrogate

    return float(np.mean(loss))
