"""Agent protocol + the crash-safety wrapper that every submission must use.

A play-phase agent is any callable ``(obs) -> list[int]`` that assumes
``obs["select"]`` is present (deck selection and crash safety are handled here,
once, so individual agents stay focused on strategy).

``make_safe`` is the contract enforcer:
  * deck-selection step  -> return the configured 60-card deck
  * any exception        -> legal fallback (never crash; a crash = instant loss)
  * invalid return       -> sanitized to a legal selection
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from ptcg import protocol as P

PlayAgent = Callable[[dict], list]


def sanitize(action: Any, obs: dict[str, Any]) -> list[int] | None:
    """Coerce an agent's return into a legal selection, or None if impossible.

    Legal = distinct ints in [0, n_options), preserving the agent's order,
    truncated/padded toward exactly ``maxCount`` selections.
    """
    n = len(P.options(obs))
    k = min(P.max_count(obs), n)
    if k <= 0:
        return []
    if not isinstance(action, Iterable) or isinstance(action, (str, bytes)):
        return None

    seen: set[int] = set()
    out: list[int] = []
    for x in action:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in seen:
            seen.add(i)
            out.append(i)
            if len(out) == k:
                break
    if len(out) < k:
        # Pad with the lowest unused indices so we always submit exactly k.
        for i in range(n):
            if i not in seen:
                out.append(i)
                seen.add(i)
                if len(out) == k:
                    break
    return out[:k] if len(out) == k else None


def make_safe(play_agent: PlayAgent, deck: Sequence[int]) -> Callable[[dict], list[int]]:
    """Wrap a play-phase agent into a full, crash-proof Kaggle agent."""
    deck_list = list(deck)

    def agent(obs: dict[str, Any]) -> list[int]:
        if P.is_deck_selection(obs):
            return deck_list
        try:
            action = play_agent(obs)
            safe = sanitize(action, obs)
            if safe is not None:
                return safe
        except Exception:  # noqa: BLE001 — never let the ladder see a crash
            pass
        return P.legal_fallback(obs)

    return agent
