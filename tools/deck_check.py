"""Validate a 60-card deck against the legal pool + PTCG construction rules.

Fast local pre-check so you get instant feedback while designing (the engine only
validates at battle start). Standard PTCG rules — confirm any tournament-specific
tweaks against the competition's rules doc.

    python tools/deck_check.py                 # checks deck.csv
    python tools/deck_check.py path/to/deck.csv
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ptcg import cards as C  # noqa: E402

BASIC_ENERGY = 5  # CardType.BASIC_ENERGY


def validate(ids: list[int]) -> tuple[bool, list[str]]:
    """Return (is_legal, problems). Reusable by the deck-builder server."""
    problems: list[str] = []
    if len(ids) != 60:
        problems.append(f"deck has {len(ids)} cards, must be exactly 60")

    if not C.is_loaded():
        problems.append("WARN: card DB not cached — only the 60-count check ran")
        return (len(ids) == 60, problems)

    unknown = [i for i in ids if C.card(i) is None]
    if unknown:
        problems.append(f"not in legal pool: {sorted(set(unknown))}")

    by_name = Counter()
    ace_specs = 0
    has_basic_pokemon = False
    for i in ids:
        c = C.card(i)
        if not c:
            continue
        if c["cardType"] != BASIC_ENERGY:
            by_name[c["name"]] += 1
        if c.get("aceSpec"):
            ace_specs += 1
        if c["cardType"] == 0 and c.get("basic"):
            has_basic_pokemon = True
    for nm, n in by_name.items():
        if n > 4:
            problems.append(f"{n}x '{nm}' exceeds the 4-copy limit")
    if ace_specs > 1:
        problems.append(f"{ace_specs} ACE SPEC cards — max 1 per deck")
    if not has_basic_pokemon:
        problems.append("no Basic Pokémon — deck cannot start a game")

    return (len(problems) == 0, problems)


def check(path: str) -> int:
    ids = [int(x) for x in Path(path).read_text().split() if x.strip()]
    ok, problems = validate(ids)
    if ok:
        print(f"OK: {path} is a legal 60-card deck.")
        return 0
    print(f"INVALID deck ({path}):")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "deck.csv"
    raise SystemExit(check(path))
