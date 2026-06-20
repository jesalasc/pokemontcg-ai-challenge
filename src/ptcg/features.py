"""Observation -> tensors (Layer 3).

Design: an **option-scoring** policy. The cabt action set is dynamic (a variable
list of legal options each step), so instead of a fixed action head we encode
  * the global state  -> encode_state(obs)        [STATE_DIM]
  * each legal option -> encode_option(opt, obs)  [OPTION_DIM]
and the policy scores every (state, option) pair, then argmax/softmax over the
*legal* options. This mirrors the rule_based scorer, so imitation/distillation
from the baseline or MCTS is a direct fit.

Keep this in lockstep with the verified schema (docs/ENGINE_API.md). Extend the
vectors with domain features (specific card ids in hand, evolution readiness,
energy types) as the domain expert identifies what matters.
"""
from __future__ import annotations

import numpy as np

from ptcg import engine_codes as E
from ptcg import heuristics as H
from ptcg import protocol as P

# --- global state encoding -------------------------------------------------
# [prizes_me, prizes_opp, handCount_me, deckCount_me, bench_me, bench_opp,
#  active_hp_me, active_maxhp_me, active_energy_me, active_hp_opp,
#  status_me(5), turn_flags(4)]  -> 19 dims
STATE_DIM = 19

# --- per-option encoding ---------------------------------------------------
# [is_attack, type, area, index, playerIndex_is_opp, energyIndex, count, number]
OPTION_DIM = 8


def _players(obs: dict) -> list[dict]:
    return (obs.get("current") or {}).get("players") or []


def encode_state(obs: dict) -> np.ndarray:
    v = np.zeros(STATE_DIM, dtype=np.float32)
    me = P.your_index(obs)
    if me is None:
        return v
    opp = 1 - me
    ps = _players(obs)
    if len(ps) < 2:
        return v
    cur = obs.get("current") or {}
    mep, opp_p = ps[me], ps[opp]
    a_me = H.active_card(obs, me) or {}
    a_opp = H.active_card(obs, opp) or {}

    v[0] = H.prizes_remaining(obs, me)
    v[1] = H.prizes_remaining(obs, opp)
    v[2] = mep.get("handCount", 0)
    v[3] = mep.get("deckCount", 0)
    v[4] = len(mep.get("bench") or [])
    v[5] = len(opp_p.get("bench") or [])
    v[6] = a_me.get("hp", 0)
    v[7] = a_me.get("maxHp", 0)
    v[8] = H.energy_count(a_me)
    v[9] = a_opp.get("hp", 0)
    v[10:15] = [
        float(mep.get(s, False))
        for s in ("poisoned", "burned", "asleep", "paralyzed", "confused")
    ]
    v[15:19] = [
        float(cur.get(s, False))
        for s in ("supporterPlayed", "stadiumPlayed", "energyAttached", "retreated")
    ]
    return v


def encode_option(option: dict, obs: dict) -> np.ndarray:
    me = P.your_index(obs)
    pidx = option.get("playerIndex")
    return np.array(
        [
            float(E.is_attack(option)),
            float(option.get("type", 0)),
            float(option.get("area", 0)),
            float(option.get("index", 0)),
            float(pidx is not None and me is not None and pidx != me),
            float(option.get("energyIndex", 0)),
            float(option.get("count", 0)),
            float(option.get("number", 0)),
        ],
        dtype=np.float32,
    )


def encode_options(obs: dict) -> np.ndarray:
    """[n_options, OPTION_DIM] for the current step (empty array if none)."""
    opts = P.options(obs)
    if not opts:
        return np.zeros((0, OPTION_DIM), dtype=np.float32)
    return np.stack([encode_option(o, obs) for o in opts])
