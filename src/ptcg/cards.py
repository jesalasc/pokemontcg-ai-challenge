"""Card database — id/attackId lookups for card-aware play.

The data comes from the official engine (`cg.api.all_card_data()` /
`all_attack()`), cached to JSON INSIDE the package (`_data/`) so it ships with
the submission and is usable on macOS without the engine. Regenerate with:

    # in Docker, with the official cg on the path:
    PYTHONPATH=/workspace/src:/workspace/data/engine python -m ptcg.cards

Degrades gracefully (returns None / defaults) if the cache is missing.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "_data"
CARDS_JSON = _DATA / "cards.json"
ATTACKS_JSON = _DATA / "attacks.json"


@lru_cache(maxsize=1)
def _cards() -> dict[int, dict]:
    if CARDS_JSON.is_file():
        return {int(c["cardId"]): c for c in json.loads(CARDS_JSON.read_text())}
    return {}


@lru_cache(maxsize=1)
def _attacks() -> dict[int, dict]:
    if ATTACKS_JSON.is_file():
        return {int(a["attackId"]): a for a in json.loads(ATTACKS_JSON.read_text())}
    return {}


def card(card_id) -> dict | None:
    return _cards().get(int(card_id)) if card_id is not None else None


def attack(attack_id) -> dict | None:
    return _attacks().get(int(attack_id)) if attack_id is not None else None


def name(card_id) -> str:
    c = card(card_id)
    return c["name"] if c else f"#{card_id}"


def attack_damage(attack_id) -> int:
    a = attack(attack_id)
    return int(a.get("damage", 0)) if a else 0


def prize_value(card_id) -> int:
    """Prizes the opponent takes when this Pokémon is KO'd (ex=2, megaEx=3)."""
    c = card(card_id)
    if not c:
        return 1
    if c.get("megaEx"):
        return 3
    if c.get("ex"):
        return 2
    return 1


def is_loaded() -> bool:
    return bool(_cards())


def build_cache() -> None:  # pragma: no cover - run in the engine container
    """Dump the card/attack DB to the package _data dir (needs cg.api)."""
    from dataclasses import asdict

    from cg.api import all_attack, all_card_data

    _DATA.mkdir(parents=True, exist_ok=True)
    cards = [asdict(c) for c in all_card_data()]
    attacks = [asdict(a) for a in all_attack()]
    CARDS_JSON.write_text(json.dumps(cards))
    ATTACKS_JSON.write_text(json.dumps(attacks))
    print(f"cached {len(cards)} cards -> {CARDS_JSON}")
    print(f"cached {len(attacks)} attacks -> {ATTACKS_JSON}")


if __name__ == "__main__":
    build_cache()
