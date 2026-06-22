"""Record a readable, decision-by-decision replay of an agent-vs-agent game.

Unlike the engine's graphical render, this logs every decision in plain language
(via ptcg.describe) — the fastest way to spot bad lines for targeted retraining
(step 4 of the learning loop, docs/TRAINING.md). View with tools/serve_replay.py.

    python harness/record_replay.py --a dragapult --b rule_based \
        --a-deck dragapult-dusknoir --b-deck crustle
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ptcg import describe  # noqa: E402
from ptcg.agents import get_agent  # noqa: E402
from ptcg.agents.base import make_safe  # noqa: E402


def _load_deck(name: str) -> list[int]:
    p = _ROOT / "decks" / f"{name}.csv"
    return [int(x) for x in p.read_text().split() if x.strip()]


def _rec_agent(name: str, deck: list[int], steps: list[dict]):
    inner = make_safe(get_agent(name), deck)

    def agent(obs: dict):
        action = inner(obs)
        if obs.get("select") is not None and obs.get("current"):
            cur = obs["current"]
            me = cur["yourIndex"]
            opts = obs["select"].get("option", [])
            steps.append({
                "n": len(steps) + 1,
                "player": me,
                "turn": cur.get("turn"),
                "context": describe.context_label(obs),
                "you": describe.player_summary(obs, me),
                "opp": describe.player_summary(obs, 1 - me, hide_hand=True),
                "options": [{"i": i, "text": describe.describe_option(o, obs)}
                            for i, o in enumerate(opts)],
                "chosen": [int(i) for i in action],
                "chosen_text": "; ".join(describe.describe_option(opts[i], obs)
                                         for i in action if i < len(opts)),
            })
        return action

    return agent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="rule_based")
    ap.add_argument("--b", default="random")
    ap.add_argument("--a-deck", default="dragapult-ex")
    ap.add_argument("--b-deck", default="dragapult-ex")
    ap.add_argument("--out", default="artifacts/replays")
    args = ap.parse_args()

    from kaggle_environments import make

    deck_a, deck_b = _load_deck(args.a_deck), _load_deck(args.b_deck)
    steps: list[dict] = []
    env = make("cabt")
    env.run([_rec_agent(args.a, deck_a, steps), _rec_agent(args.b, deck_b, steps)])

    r = env.state[0]["reward"]
    winner = 0 if r == 1 else (1 if env.state[1]["reward"] == 1 else None)
    replay = {
        "a": args.a, "b": args.b, "a_deck": args.a_deck, "b_deck": args.b_deck,
        "winner": winner, "names": {"0": f"{args.a} ({args.a_deck})", "1": f"{args.b} ({args.b_deck})"},
        "n_steps": len(steps), "steps": steps,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{args.a}_vs_{args.b}_{int(time.time())}.json"
    f.write_text(json.dumps(replay))
    print(f"recorded {len(steps)} decisions, winner={replay['names'].get(str(winner), 'draw')}")
    print(f"-> {f}   (view with: make replay-view)")


if __name__ == "__main__":
    main()
