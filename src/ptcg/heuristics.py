"""Reusable, domain-expert-tunable evaluation helpers.

These read the *verified* obs schema (see docs/ENGINE_API.md):
  current.players[i] = {active, bench, hand, discard, prize, deckCount,
                        handCount, benchMax, poisoned, burned, asleep,
                        paralyzed, confused}
  card = {id, serial, playerIndex, hp, maxHp, energies, energyCards,
          tools, preEvolution, appearThisTurn}

The win condition is taking all your prizes, so prize differential dominates
board value. Tune the weights here with the domain expert — this is exactly the
"translate card knowledge into reward shaping" lever the brief calls out.
"""
from __future__ import annotations

from typing import Any

from ptcg import protocol as P

# --- weights (tune with domain expert) ------------------------------------
W_PRIZE = 100.0       # each prize you've taken (opponent prizes remaining gap)
W_KO_RISK = 8.0       # being one hit from losing your active
W_ACTIVE_HP = 0.2     # keep your active healthy
W_BENCH = 3.0         # board development
W_HAND = 1.0          # card advantage
W_ENERGY = 2.0        # energy in play on your side
W_STATUS = 4.0        # special conditions on your active are bad


def _players(obs: dict) -> list[dict]:
    cur = obs.get("current") or {}
    return cur.get("players") or []


def _as_list(x: Any) -> list:
    return x if isinstance(x, list) else []


def prizes_remaining(obs: dict, player: int) -> int:
    ps = _players(obs)
    if player >= len(ps):
        return 6
    return len(_as_list(ps[player].get("prize")))


def active_card(obs: dict, player: int) -> dict | None:
    ps = _players(obs)
    if player >= len(ps):
        return None
    act = _as_list(ps[player].get("active"))
    return act[0] if act and isinstance(act[0], dict) else None


def card_damage(card: dict | None) -> int:
    if not card:
        return 0
    return max(0, int(card.get("maxHp", 0)) - int(card.get("hp", 0)))


def energy_count(card: dict | None) -> int:
    if not card:
        return 0
    return len(_as_list(card.get("energies"))) or len(_as_list(card.get("energyCards")))


def _status_count(player_state: dict) -> int:
    return sum(
        bool(player_state.get(s))
        for s in ("poisoned", "burned", "asleep", "paralyzed", "confused")
    )


def board_value(obs: dict, player: int) -> float:
    """Scalar estimate of how good the position is for ``player`` (higher better)."""
    ps = _players(obs)
    if player >= len(ps):
        return 0.0
    opp = 1 - player
    me, them = ps[player], ps[opp]

    # Prize race: you win by emptying YOUR prize pile; opponent wins by emptying theirs.
    prize_term = W_PRIZE * (prizes_remaining(obs, opp) - prizes_remaining(obs, player))

    my_active = active_card(obs, player)
    hp_term = W_ACTIVE_HP * (int(my_active.get("hp", 0)) if my_active else 0)
    ko_risk = -W_KO_RISK if (my_active and int(my_active.get("hp", 0)) <= 30) else 0.0

    bench_term = W_BENCH * len(_as_list(me.get("bench")))
    hand_term = W_HAND * int(me.get("handCount", len(_as_list(me.get("hand")))))
    energy_term = W_ENERGY * energy_count(my_active)
    status_term = -W_STATUS * _status_count(me)

    return prize_term + hp_term + ko_risk + bench_term + hand_term + energy_term + status_term


def my_value_minus_opp(obs: dict) -> float:
    """Value from the to-move player's perspective."""
    me = P.your_index(obs)
    if me is None:
        return 0.0
    return board_value(obs, me) - board_value(obs, 1 - me)
