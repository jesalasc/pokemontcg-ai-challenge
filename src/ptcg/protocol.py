"""Verified cabt engine I/O contract.

Ground truth extracted from kaggle_environments==1.30.1
(envs/cabt/cabt.py + cg/game.py). See docs/ENGINE_API.md.

Each turn the engine calls ``agent(obs: dict) -> list[int]``:

  obs = {
    "select":  {"option": [...], "maxCount": int} | None,
    "logs":    [...],          # event history this turn / game
    "current": {...} | None,   # board state (None during deck selection)
    "search_begin_input": str  # serialized state -> drives the forward model
  }

Return: a list of *indices* into ``obs["select"]["option"]``.
When ``obs["select"] is None`` it is the DECK-SELECTION step: return a list of
60 card IDs instead.
"""
from __future__ import annotations

from typing import Any

# current["result"] values
RESULT_ONGOING = -1
# >= 0 means the game is over:
#   0 -> player 0 won, 1 -> player 1 won, anything else (e.g. 2) -> draw
RESULT_DRAW = 2


def is_deck_selection(obs: dict[str, Any]) -> bool:
    """True on the first call, where the agent must return a 60-card deck."""
    return obs.get("select") is None


def options(obs: dict[str, Any]) -> list[Any]:
    """The list of legal option objects the engine is offering this step."""
    sel = obs.get("select")
    return [] if sel is None else sel.get("option", [])


def max_count(obs: dict[str, Any]) -> int:
    """How many options must be selected this step."""
    sel = obs.get("select")
    return 0 if sel is None else int(sel.get("maxCount", 0))


def your_index(obs: dict[str, Any]) -> int | None:
    """Which player (0 or 1) the agent is, from the engine's perspective."""
    cur = obs.get("current")
    return None if cur is None else cur.get("yourIndex")


def is_terminal(obs: dict[str, Any]) -> bool:
    cur = obs.get("current")
    return cur is not None and cur.get("result", RESULT_ONGOING) >= 0


def legal_fallback(obs: dict[str, Any]) -> list[int]:
    """A guaranteed-legal selection: the first ``maxCount`` options.

    Mirrors the engine's own ``first_agent``. Used whenever a smarter agent
    errors or returns something invalid — a crash/illegal move is an instant
    loss (reward -1), so this is the floor we never fall below.
    """
    n = len(options(obs))
    k = min(max_count(obs), n)
    return list(range(k))
