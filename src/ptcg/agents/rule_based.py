"""Layer 1 — rule-based baseline (card-DB aware).

Measured principle: attacking ENDS the turn, so develop everything first and
attack last. Now calibrated against the official enums (engine_codes) and the
card database (cards), so it:
  * MAIN: develop -> attack(best/KO) -> end turn
  * picks the highest-damage / KO'ing attack when several are offered
  * targets the opponent's lowest-HP, highest-prize (ex/megaEx) Pokémon
  * never crashes (wrapped by make_safe)

Domain expert: tune via MAIN_TYPE_PRIORITY (sequence develop actions),
heuristics.py (board value), and engine_codes.py (codes). The sample agents in
data/samples/ show deck-specific lines worth porting (e.g. Dragapult AttackPlan).
"""
from __future__ import annotations

from ptcg import cards
from ptcg import engine_codes as E
from ptcg import heuristics as H
from ptcg import protocol as P

# Per-type nudges within the "develop" tier (keyed by OptionType). Raise to
# sequence develop actions (e.g. attach before evolve); keep RETREAT low.
MAIN_TYPE_PRIORITY: dict[int, float] = {
    E.OptionType.RETREAT: -5.0,   # retreating costs energy — only if useful
}

DEVELOP_SCORE = 1000.0   # do all setup first
ATTACK_BASE = 1.0        # attack only after develop (ends the turn); 1..~3 range
KO_BONUS = 1.0
END_TURN_SCORE = -1e6    # end turn only when forced
TARGET_OPP_BASE = 500.0


def _players(obs: dict) -> list[dict]:
    return (obs.get("current") or {}).get("players") or []


def _card_at(obs: dict, player_index: int, area, index) -> dict | None:
    if area is None or index is None:
        return None
    ps = _players(obs)
    if not (0 <= player_index < len(ps)):
        return None
    zone = E.ZONE_NAME.get(area)
    cards_ = ps[player_index].get(zone) if zone else None
    if isinstance(cards_, list) and 0 <= index < len(cards_) and isinstance(cards_[index], dict):
        return cards_[index]
    return None


def _attack_score(option: dict, obs: dict) -> float:
    """Order attacks among themselves: more damage, and KO > non-KO."""
    me = P.your_index(obs)
    dmg = cards.attack_damage(option.get("attackId"))
    score = ATTACK_BASE + min(0.9, dmg / 1000.0)
    if me is not None:
        opp_active = H.active_card(obs, 1 - me)
        if opp_active and dmg >= int(opp_active.get("hp", 9999)):
            score += KO_BONUS  # this attack KOs the defender
    return score


def _target_score(option: dict, obs: dict) -> float:
    """Damage targeting: prefer lowest HP, break ties toward ex/megaEx (more prizes)."""
    me = P.your_index(obs)
    pidx = option.get("playerIndex")
    card = _card_at(obs, pidx, option.get("area"), option.get("index"))
    hp = int(card.get("hp", 9999)) if card else 9999
    prize = cards.prize_value(card.get("id")) if card else 1
    base = TARGET_OPP_BASE if (pidx is not None and me is not None and pidx != me) else 0.0
    return base - hp + 0.5 * prize


def _score(option: dict, obs: dict, ctx) -> float:
    if ctx == E.SelectContext.MAIN:
        t = option.get("type")
        if t == E.OptionType.END:
            return END_TURN_SCORE
        if E.is_attack(option):
            return _attack_score(option, obs)
        return DEVELOP_SCORE + MAIN_TYPE_PRIORITY.get(t, 0.0)

    if ctx in E.DAMAGE_TARGET_CONTEXTS:
        return _target_score(option, obs)

    # Other selection contexts: neutral (engine lists sensible options first).
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


# Position evaluation reused by MCTS / RL.
evaluate = H.my_value_minus_opp
