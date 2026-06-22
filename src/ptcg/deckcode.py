"""Deck conditioning: encode a 60-card deck as a vector over the legal card pool.

This is the input that lets ONE network pilot any deck in the roster. It's a pure
representation (counts of each legal card) — not strategy — so it's allowed under
the "no hand-coded strategy" rule.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ptcg import cards as C


@lru_cache(maxsize=1)
def pool_ids() -> tuple[int, ...]:
    """Sorted legal card ids — the fixed index space for deck/card vectors."""
    return tuple(sorted(C._cards().keys()))


@lru_cache(maxsize=1)
def _index() -> dict[int, int]:
    return {cid: i for i, cid in enumerate(pool_ids())}


def pool_size() -> int:
    return len(pool_ids())


def deck_vector(ids) -> np.ndarray:
    """Count vector over the pool: deck_vector[i] = copies of pool card i."""
    v = np.zeros(max(1, pool_size()), dtype=np.float32)
    idx = _index()
    for cid in ids:
        i = idx.get(int(cid))
        if i is not None:
            v[i] += 1.0
    return v
