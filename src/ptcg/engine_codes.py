"""cabt integer enums — the single place to calibrate engine codes.

Values below are derived empirically from a real game (artifacts/schema/*.json):
the only HIGH-CONFIDENCE fact is that attack options carry an ``attackId`` (and
appear as type 13 in the main menu). Zone/type names are best-effort guesses.

>>> CALIBRATE these against the sample kernel before trusting the named codes:
    kaggle kernels pull kiyotah/a-sample-rule-based-agent-dragapult-ex-deck
Fix them here once and every agent benefits.
"""
from __future__ import annotations


class Ctx:
    """select.context — the kind of decision being asked."""
    MAIN = 0          # main action menu: play / attach / evolve / retreat / attack / end
    # 1,2,3,4,7,8,22 -> target / card-selection contexts (option.type == 3)
    ENERGY = 30       # energy attach / distribution (option has count, energyIndex)
    NUMBER = 38       # choose a number (option has 'number')
    BINARY = 41       # simple A/B choice (option has only 'type')

    SELECT_CONTEXTS = frozenset({1, 2, 3, 4, 7, 8, 22})


# option.type inside Ctx.MAIN.
#   CONFIRMED 13 == attack (matched attackId count exactly).
#   CONFIRMED 14 == end turn (always present, always the last menu option).
#   {7,8,9,10,12} == develop actions (play/attach/evolve/retreat/ability) — exact
#                    names TODO via the sample kernel; treated uniformly for now.
# Key insight (measured): attacking ENDS the turn, so a good agent does all
# its develop actions FIRST and attacks LAST. See agents/rule_based.py.
ATTACK_TYPE = 13
END_TURN_TYPE = 14


class Area:
    """option.area — board zone. Guessed mapping; CONFIRM via sample kernel."""
    HAND = 1
    ACTIVE = 2
    BENCH = 3
    DISCARD = 4
    DECK = 5
    # 7 also observed (stadium/tool/lost-zone?) — unconfirmed

    ZONE_NAME = {1: "hand", 2: "active", 3: "bench", 4: "discard", 5: "deck"}


def is_attack(option: dict) -> bool:
    """High-confidence: an option is an attack iff it carries an attackId."""
    return option.get("attackId") is not None
