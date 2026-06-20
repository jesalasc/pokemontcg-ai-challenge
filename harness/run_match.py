"""Run a single cabt match between two agents.

Agents are specified by registry name ("random", "rule_based", ...) or passed as
raw callables. Returns a structured result; optionally returns the live env so
callers (replay) can render it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

# Allow running as a script: add repo's src/ to the path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ptcg.agents import get_agent  # noqa: E402
from ptcg.agents.base import make_safe  # noqa: E402
from ptcg.deck import load_deck  # noqa: E402

AgentSpec = str | Callable[[dict], list]


def build_agent(spec: AgentSpec, deck: Any = None) -> Callable[[dict], list]:
    if callable(spec):
        return spec
    return make_safe(get_agent(spec), deck if deck is not None else load_deck())


def _reward(state_entry: Any) -> Any:
    try:
        return state_entry["reward"]
    except (TypeError, KeyError):
        return getattr(state_entry, "reward", None)


def run_match(
    agent_a: AgentSpec,
    agent_b: AgentSpec,
    decks: tuple[Any, Any] | None = None,
    configuration: dict | None = None,
    return_env: bool = False,
) -> dict | tuple[dict, Any]:
    """Play one game. Returns {rewards, winner, steps, error}.

    winner: 0 (A), 1 (B), or None (draw). 'error' flags an engine/agent fault.
    """
    from kaggle_environments import make  # lazy: needs the Linux engine

    a = build_agent(agent_a, decks[0] if decks else None)
    b = build_agent(agent_b, decks[1] if decks else None)

    env = make("cabt", configuration=dict(configuration or {}))
    env.run([a, b])

    rewards = [_reward(env.state[0]), _reward(env.state[1])]
    if rewards[0] == 1:
        winner = 0
    elif rewards[1] == 1:
        winner = 1
    else:
        winner = None
    result = {
        "rewards": rewards,
        "winner": winner,
        "steps": len(env.steps),
        "error": any(
            getattr(s, "status", s.get("status") if isinstance(s, dict) else None)
            in ("ERROR", "INVALID", "TIMEOUT")
            for s in env.state
        ),
    }
    return (result, env) if return_env else result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run one cabt match.")
    ap.add_argument("--a", default="rule_based")
    ap.add_argument("--b", default="random")
    args = ap.parse_args()
    print(run_match(args.a, args.b))
