"""Random play-phase agent — sanity baseline and self-play sparring partner."""
from __future__ import annotations

import random

from ptcg import protocol as P


def play(obs: dict) -> list[int]:
    n = len(P.options(obs))
    k = min(P.max_count(obs), n)
    return random.sample(range(n), k)
