"""Deck loading.

The deck is part of the submission: on the deck-selection step the agent returns
a list of 60 card IDs. We keep the list in ``deck.csv`` at the repo root (one card
ID per line) so the domain expert can swap decks without touching code.

To change deck: edit deck.csv, or set the PTCG_DECK env var to another csv path.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

DECK_SIZE = 60


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("PTCG_DECK")
    if env:
        paths.append(Path(env))
    # repo root (src/ptcg/deck.py -> parents[2]) and cwd, to work both in the
    # dev tree and inside the unpacked submission (deck.csv next to main.py).
    paths.append(Path(__file__).resolve().parents[2] / "deck.csv")
    paths.append(Path.cwd() / "deck.csv")
    return paths


@lru_cache(maxsize=4)
def load_deck(path: str | None = None) -> tuple[int, ...]:
    """Return the 60 card IDs as a tuple. Cached; tuple so it's hashable/immutable."""
    candidates = [Path(path)] if path else _candidate_paths()
    for p in candidates:
        if p.is_file():
            cards = [int(line) for line in p.read_text().splitlines() if line.strip()]
            if len(cards) != DECK_SIZE:
                raise ValueError(f"{p} has {len(cards)} cards, expected {DECK_SIZE}.")
            return tuple(cards)
    raise FileNotFoundError(f"No deck.csv found in: {[str(p) for p in candidates]}")
