"""Probe the engine's actual behavior to surface rule divergences.

The rules are "official PTCG, uniquely adjusted" — and those adjustments decide
games. This runs many games and aggregates what the engine actually does (from
the event logs), so we can compare against paper PTCG and pin findings into
tests/test_sim_differences.py.

    # in Docker:
    python tools/probe_rules.py -n 50

Extend with targeted probes (status timing, prize counts on ex/mega KOs, mulligan
handling) as questions arise.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# LogType (from cg.api): COIN=22, RESULT=23; RESULT.reason 1=0 prizes, 2=deck-out,
# 3=no active Pokémon, 4=card effect.
COIN, RESULT = 22, 23
REASON = {1: "took all prizes", 2: "deck-out", 3: "no active Pokemon", 4: "card effect"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--games", type=int, default=50)
    args = ap.parse_args()

    from kaggle_environments import make

    from ptcg.agents import random_agent
    from ptcg.agents.base import make_safe
    from ptcg.deck import load_deck

    deck = load_deck()
    coins = Counter()
    end_reasons = Counter()
    game_lengths = []

    seen_logs: set[int] = set()

    def recorder(inner):
        def f(obs):
            for log in obs.get("logs") or []:
                key = id(log)  # within a game, avoid recounting the same dict
                if key in seen_logs:
                    continue
                seen_logs.add(key)
                t = log.get("type")
                if t == COIN:
                    coins[bool(log.get("head"))] += 1
                elif t == RESULT:
                    end_reasons[REASON.get(log.get("reason"), log.get("reason"))] += 1
            return inner(obs)

        return f

    for _ in range(args.games):
        seen_logs.clear()
        a = recorder(make_safe(random_agent.play, deck))
        b = make_safe(random_agent.play, deck)
        env = make("cabt")
        env.run([a, b])
        game_lengths.append(len(env.steps))

    heads = coins.get(True, 0)
    tails = coins.get(False, 0)
    total = heads + tails
    print(f"games: {args.games}")
    print(f"coin flips: {total}  heads {heads} ({heads / total:.3f})  tails {tails}" if total else "no coin flips seen")
    print("game-end reasons:", dict(end_reasons))
    print(f"avg game length: {sum(game_lengths) / len(game_lengths):.1f} steps")
    print("\nCompare against paper PTCG; pin any divergence in tests/test_sim_differences.py")


if __name__ == "__main__":
    main()
