"""Layer 2 — determinized MCTS over the engine forward model.

cabt is imperfect-information (hidden hands/decks) + stochastic. The standard
treatment, and what the brief prescribes, is **determinized MCTS**: each
iteration, ask the engine to sample a concrete world consistent with public
info (``forward_model.sample_determinization``), then run perfect-info MCTS on
that sample; average over many determinizations. The rule_based agent serves as
the rollout (playout) policy.

This module implements the full search loop against the ForwardModel interface.
Until the engine's search API is wired (see forward_model.py / fetch_engine.sh),
``get_forward_model()`` returns None and we transparently fall back to the
rule_based action — so "mcts" is always runnable, just not yet searching.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from ptcg import protocol as P
from ptcg.agents import rule_based
from ptcg.forward_model import get_forward_model

# Search budget. The ladder gives ~600s overage per game, so keep per-move cost
# modest and rely on iterative deepening by wall-clock.
TIME_BUDGET_S = 1.0
MAX_ITERS = 400
N_DETERMINIZATIONS = 16
UCT_C = 1.4


@dataclass
class Node:
    to_play: int
    parent: "Node | None" = None
    children: dict[int, "Node"] = field(default_factory=dict)
    untried: list[int] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0  # summed reward from the perspective of `to_play`

    def uct_child(self, c: float) -> "Node":
        log_n = math.log(self.visits + 1)
        return max(
            self.children.values(),
            key=lambda ch: (ch.value / ch.visits if ch.visits else 0.0)
            + c * math.sqrt(log_n / (ch.visits + 1e-9)),
        )


def _rollout(fm: Any, state: Any, max_depth: int = 60) -> float:
    """Play to the end with the rule_based policy; return reward for player 0."""
    depth = 0
    while not fm.is_terminal(state) and depth < max_depth:
        obs = fm.observation(state)  # forward model exposes obs for the policy
        action = rule_based.play(obs)
        state = fm.step(state, action)
        depth += 1
    return fm.reward(state, player=0)


def _search(fm: Any, obs: dict) -> int:
    """Run determinized MCTS and return the best option index."""
    root_player = P.your_index(obs) or 0
    stats: dict[int, list[float]] = {}  # option index -> [visits, value]

    deadline = time.time() + TIME_BUDGET_S
    iters = 0
    while iters < MAX_ITERS and time.time() < deadline:
        # Sample a concrete world consistent with what we can see.
        state = fm.sample_determinization(fm.root(obs))
        root = Node(to_play=root_player, untried=list(range(len(P.options(obs)))))

        # --- one perfect-info MCTS iteration on this determinization ---
        node, path = root, [root]
        while not node.untried and node.children:
            node = node.uct_child(UCT_C)
            path.append(node)
        if node.untried:
            a = node.untried.pop()
            child_state = fm.step(state, [a])
            child = Node(to_play=fm.to_play(child_state), parent=node)
            node.children[a] = child
            path.append(child)
            state = child_state
        reward0 = _rollout(fm, state)
        for nd in path:
            nd.visits += 1
            nd.value += reward0 if nd.to_play == 0 else -reward0

        for a, ch in root.children.items():
            s = stats.setdefault(a, [0.0, 0.0])
            s[0] += ch.visits
            s[1] += ch.value
        iters += 1

    if not stats:
        raise RuntimeError("MCTS produced no statistics")
    # Robust child: most-visited option index.
    return max(stats, key=lambda a: stats[a][0])


def play(obs: dict) -> list[int]:
    sel = obs.get("select") or {}
    k = min(int(sel.get("maxCount", 1)), len(sel.get("option", [])))
    fm = get_forward_model()
    if fm is None:
        # Engine search API not available -> use the strong baseline.
        return rule_based.play(obs)
    try:
        best = _search(fm, obs)
        # For multi-select, fill remaining picks with the baseline ranking.
        if k <= 1:
            return [best]
        rest = [i for i in rule_based.play({**obs, "select": {**sel, "maxCount": k}})
                if i != best]
        return [best, *rest][:k]
    except Exception:
        return rule_based.play(obs)
