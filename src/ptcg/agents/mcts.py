"""Layer 2 — determinized Monte-Carlo search.

For the current decision, we evaluate each legal option by: sample several
determinizations of the hidden state, play the option in each, roll out to the
end with the rule_based policy, and average the outcome. Pick the best-scoring
option. This is "flat Monte-Carlo with determinization" — the honest first step
toward full UCT, and it directly handles imperfect information by averaging over
sampled worlds.

It's strictly fallback-safe: if the engine search API isn't available, the move
is multi-select, or anything errors, it returns the rule_based action. Budgeted
by wall-clock (the ladder gives ~600s/game).

Upgrade path: replace flat-MC with UCT over the forward model (selection/
expansion via search_step children), and replace the placeholder opponent belief
in forward_model.py with one inferred from the public logs.
"""
from __future__ import annotations

import time

from ptcg import heuristics as H
from ptcg import protocol as P
from ptcg.agents import rule_based
from ptcg.deck import load_deck
from ptcg.forward_model import get_forward_model

TIME_BUDGET_S = 2.0
N_DETERMINIZATIONS = 8
MAX_ROLLOUT_DEPTH = 80
# Non-terminal rollout cutoffs fall back to a scaled board-value estimate.
VALUE_SCALE = 600.0


def _leaf_value(fm, state, me: int) -> float:
    if fm.is_terminal(state):
        return fm.reward(state, me)
    od = fm.obs_dict(state)
    v = H.board_value(od, me) - H.board_value(od, 1 - me)
    return max(-0.99, min(0.99, v / VALUE_SCALE))


def _rollout(fm, state, me: int, deadline: float) -> float:
    depth = 0
    while not fm.is_terminal(state) and depth < MAX_ROLLOUT_DEPTH and time.time() < deadline:
        od = fm.obs_dict(state)
        sel = od.get("select")
        if not sel or not sel.get("option"):
            break
        state = fm.step(state, rule_based.play(od))
        depth += 1
    return _leaf_value(fm, state, me)


def _flat_mc(fm, obs: dict, n_opts: int) -> list[float]:
    me = P.your_index(obs) or 0
    total = [0.0] * n_opts
    count = [0] * n_opts
    deadline = time.time() + TIME_BUDGET_S
    d = 0
    while d < N_DETERMINIZATIONS and time.time() < deadline:
        root = fm.root(obs)              # one sampled world
        for a in range(n_opts):
            if time.time() >= deadline:
                break
            try:
                child = fm.step(root, [a])
                total[a] += _rollout(fm, child, me, deadline)
                count[a] += 1
            except Exception:
                continue
        d += 1
    return [total[i] / count[i] if count[i] else float("-inf") for i in range(n_opts)]


def play(obs: dict) -> list[int]:
    sel = obs.get("select") or {}
    opts = sel.get("option", [])
    k = min(int(sel.get("maxCount", 1)), len(opts))
    if k <= 0:
        return []

    # Search only single-select decisions with a usable forward model.
    if k != 1 or obs.get("search_begin_input") is None:
        return rule_based.play(obs)
    fm = get_forward_model(load_deck())
    if fm is None:
        return rule_based.play(obs)

    try:
        scores = _flat_mc(fm, obs, len(opts))
        if all(s == float("-inf") for s in scores):
            return rule_based.play(obs)
        return [max(range(len(opts)), key=lambda i: scores[i])]
    except Exception:
        return rule_based.play(obs)
