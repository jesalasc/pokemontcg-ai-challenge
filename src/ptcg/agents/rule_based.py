"""Layer 1 — rule-based baseline.

Philosophy: never crash, always make legal progress, and exploit the one
high-confidence signal we have (attacks carry ``attackId``) plus board-aware
targeting. This is intentionally simple and *parameterized* so the domain expert
can encode real Pokémon knowledge (attack selection, evolution timing, prize
math) by editing the scoring functions and ``engine_codes.py``.

Decision flow, per the verified schema:
  * context == MAIN   -> prefer attacking when offered, else develop the board
  * target/selection  -> prefer KOing the opponent's lowest-HP Pokémon
  * energy attach      -> put energy on the active (our attacker)
  * everything else    -> first legal option (engine's safe default)

This already beats random because random almost never assembles and uses
attacks; calibrate the type/area codes from the sample kernel to go further.
"""
from __future__ import annotations

from ptcg import engine_codes as E
from ptcg import heuristics as H
from ptcg import protocol as P

# MAIN-menu ordering (measured, not guessed): attacking ENDS the turn, so the
# winning order is develop-everything -> attack -> end turn. Per-type nudges on
# top of the "develop" tier let the domain expert sequence develop actions
# (e.g. attach before evolve) once the codes are named. CALIBRATE via sample.
MAIN_TYPE_PRIORITY: dict[int, float] = {
    # e.g. ATTACH_ENERGY: +3, EVOLVE: +2, SUPPORTER: +5, RETREAT: -5
}

DEVELOP_SCORE = 1000.0   # do all setup first
ATTACK_SCORE = 1.0       # attack only once setup is exhausted (ends the turn)
END_TURN_SCORE = -1e6    # end the turn only when nothing else is legal
TARGET_OPP_BASE = 500.0


def _card_at(obs: dict, player_index: int, area: int | None, index: int | None) -> dict | None:
    """Best-effort lookup of a card by (player, area, index) using the guessed
    zone map. Returns None if the mapping/indices don't resolve."""
    if area is None or index is None:
        return None
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if not (0 <= player_index < len(players)):
        return None
    zone = E.Area.ZONE_NAME.get(area)
    if zone is None:
        return None
    cards = players[player_index].get(zone)
    if isinstance(cards, list) and 0 <= index < len(cards) and isinstance(cards[index], dict):
        return cards[index]
    return None


def _score(option: dict, obs: dict, ctx: int | None) -> float:
    me = P.your_index(obs)

    # 1) Main menu: develop everything first, attack last, end turn only if forced.
    if ctx == E.Ctx.MAIN:
        if option.get("type") == E.END_TURN_TYPE:
            return END_TURN_SCORE
        if E.is_attack(option):
            return ATTACK_SCORE
        return DEVELOP_SCORE + MAIN_TYPE_PRIORITY.get(option.get("type"), 0.0)

    # 2) Targeting an opponent's Pokémon: prefer the lowest-HP (closest to KO).
    pidx = option.get("playerIndex")
    if pidx is not None and me is not None and pidx != me:
        card = _card_at(obs, pidx, option.get("area"), option.get("index"))
        hp = int(card.get("hp", 9999)) if card else 9999
        return TARGET_OPP_BASE - hp  # lower HP -> higher score

    # 3) Energy distribution: attach to our active attacker.
    if ctx == E.Ctx.ENERGY and option.get("area") == E.Area.ACTIVE:
        return 50.0

    # 4) Default: neutral — engine tends to list useful options first.
    return 0.0


def play(obs: dict) -> list[int]:
    sel = obs.get("select") or {}
    opts = sel.get("option", [])
    k = min(int(sel.get("maxCount", 1)), len(opts))
    if k <= 0:
        return []
    ctx = sel.get("context")
    ranked = sorted(range(len(opts)), key=lambda i: _score(opts[i], obs, ctx), reverse=True)
    return ranked[:k]


# Expose the position evaluation so MCTS / RL can reuse the same value sense.
evaluate = H.my_value_minus_opp
