"""Elo helpers for interpreting local match results.

The ladder uses an Elo-like rating; locally we approximate the rating *gap*
between two agents from their head-to-head score, which is what tells us whether
a new version is actually stronger.
"""
from __future__ import annotations

import math


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability A beats B given ratings."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update(rating: float, expected: float, score: float, k: float = 32.0) -> float:
    """One Elo update step. score in {0, 0.5, 1}."""
    return rating + k * (score - expected)


def elo_diff_from_score(score: float, eps: float = 1e-4) -> float:
    """Rating gap implied by a head-to-head score in (0, 1).

    score=0.5 -> 0 ; score=0.75 -> ~+191 ; clamped to avoid infinities.
    """
    p = min(1.0 - eps, max(eps, score))
    return -400.0 * math.log10(1.0 / p - 1.0)


def wilson_interval(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win-rate (draws counted as 0.5 win)."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))
