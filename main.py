"""Submission entrypoint for the PTCG AI Battle Challenge (cabt engine).

The Kaggle ladder imports this file and calls ``agent(obs)`` every step.
This file is the ONE thing that ships in the .tar.gz (with deck.csv + src/).

Design rule: this module must ALWAYS expose a working, crash-proof ``agent``.
A crash, timeout, or illegal move is an instant loss (reward -1), so even a
total import failure degrades gracefully to a self-contained legal agent.

Pick the brain via the PTCG_AGENT env var (default "rule_based"):
    PTCG_AGENT=mcts        # Layer 2
    PTCG_AGENT=rl          # Layer 3
"""
from __future__ import annotations

import os
import sys

# Make the bundled package importable both in the dev tree and in the unpacked
# submission (where src/ sits next to main.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
# _SRC: the ptcg package; _HERE: bundled top-level packages (cg/ for mcts, training/ for rl).
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _build_agent():
    """Return the configured, crash-safe agent — or a self-contained fallback."""
    try:
        from ptcg.agents import get_agent
        from ptcg.agents.base import make_safe
        from ptcg.deck import load_deck

        name = os.environ.get("PTCG_AGENT") or _agent_from_file() or "rule_based"
        try:
            play = get_agent(name)
        except Exception:
            play = get_agent("random")
        return make_safe(play, load_deck())
    except Exception:
        return _emergency_agent()


def _agent_from_file():
    """Agent name baked into the submission (tools/build_submission.py writes it).

    The ladder can't set env vars, so the build pins the agent via agent.txt.
    """
    try:
        p = os.path.join(_HERE, "agent.txt")
        if os.path.isfile(p):
            with open(p) as f:
                name = f.read().strip()
            return name or None
    except Exception:
        pass
    return None


def _emergency_agent():
    """Zero-dependency agent used only if the package itself fails to import.

    Returns a hardcoded legal deck on selection and the first legal options
    otherwise. Never references anything outside the stdlib.
    """
    deck = _read_deck_fallback()

    def agent(obs: dict) -> list:
        sel = obs.get("select")
        if sel is None:
            return deck
        n = len(sel.get("option", []))
        k = min(int(sel.get("maxCount", 0)), n)
        return list(range(k))

    return agent


def _read_deck_fallback() -> list:
    try:
        with open(os.path.join(_HERE, "deck.csv")) as f:
            cards = [int(line) for line in f if line.strip()]
        if len(cards) == 60:
            return cards
    except Exception:
        pass
    # Last resort: the engine's known-legal default deck.
    return (
        [721, 721] + [722] * 4 + [723] * 4 + [1092] + [1121] * 2 + [1145] * 2
        + [1163] * 2 + [1219] * 4 + [1227] * 4 + [1262] * 2 + [3] * 33
    )


agent = _build_agent()


if __name__ == "__main__":
    # Smoke test the contract without the engine (handy on macOS).
    print("deck-selection ->", agent({"select": None})[:5], "...(60 cards)")
    fake = {"select": {"option": [{"a": 1}, {"a": 2}, {"a": 3}], "maxCount": 1},
            "current": {"yourIndex": 0, "result": -1}, "logs": []}
    print("play-step ->", agent(fake))
