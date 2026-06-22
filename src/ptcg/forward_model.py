"""Engine-backed forward model for search (Layer 2).

Uses the official `cg.api` search API (from the cg-lib dataset / bundled in mcts
submissions). `search_begin` takes the current observation plus a *prediction* of
all hidden information (our deck/prize order, the opponent's deck/prize/hand/
active) and returns a concrete, rollable clone — i.e. one **determinization**.
`search_step` advances it; `search_end`/`search_release` free memory.

Determinization recipe (from the official MCTS sample): sample our remaining
deck/prizes from our known 60-card list, and fill the opponent's hidden zones
with legal placeholders. Crude but legal — good enough to roll forward. Upgrade
later by modelling a real opponent belief (track revealed cards from logs).

Availability: needs `cg.api`, which exists in Docker (PYTHONPATH includes
data/engine) and in mcts submissions (cg/ bundled). Elsewhere `available()` is
False and callers fall back to the baseline.
"""
from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

# Legal placeholder ids for the opponent's hidden cards (from the sample).
POKEMON_PLACEHOLDER = 1072   # Snorlax (a Basic Pokémon)
ENERGY_PLACEHOLDER = 1       # Basic {G} Energy


def _search_api():
    try:
        from cg.api import (  # type: ignore
            search_begin,
            search_end,
            search_step,
            to_observation_class,
        )

        return search_begin, search_step, search_end, to_observation_class
    except Exception:
        return None


class EngineForwardModel:
    """A determinized, rollable clone of the live battle. ``state`` = SearchState."""

    def __init__(self, deck, opponent_deck=None) -> None:
        self.deck = list(deck)
        # Known opponent decklist (self-play) -> a better determinization belief.
        # None at ladder time -> legal placeholders. See docs/STRATEGY_NOTES.md.
        self.opponent_deck = list(opponent_deck) if opponent_deck else None
        api = _search_api()
        self._ok = api is not None
        if self._ok:
            self._begin, self._step, self._end, self._to_obs = api

    def available(self) -> bool:
        return self._ok

    def root(self, obs_dict: dict) -> Any:
        """Build one determinization of the current position via search_begin."""
        obs = self._to_obs(obs_dict)
        st = obs.current
        yi = st.yourIndex
        me, opp = st.players[yi], st.players[1 - yi]

        def sample(pool, n):
            pool = list(pool)
            return random.sample(pool, n) if 0 <= n <= len(pool) else (pool + [pool[0]] * n)[:n]

        your_deck = sample(self.deck, me.deckCount)
        your_prize = sample(self.deck, len(me.prize))

        if self.opponent_deck:  # self-play: sample the opponent's deck from their list
            opponent_deck = sample(self.opponent_deck, opp.deckCount)
        else:                    # ladder: legal placeholder belief (TODO: infer from logs)
            opponent_deck = [POKEMON_PLACEHOLDER] * opp.deckCount
        opponent_prize = [ENERGY_PLACEHOLDER] * len(opp.prize)
        opponent_hand = [ENERGY_PLACEHOLDER] * opp.handCount
        opp_active_facedown = len(opp.active) > 0 and opp.active[0] is None
        opponent_active = [POKEMON_PLACEHOLDER] if opp_active_facedown else []

        return self._begin(
            obs, your_deck, your_prize, opponent_deck,
            opponent_prize, opponent_hand, opponent_active,
        )

    def end(self) -> None:
        """Free all search states from the last search (call after each move)."""
        if self._ok:
            try:
                self._end()
            except Exception:
                pass

    def obs_dict(self, state: Any) -> dict:
        return asdict(state.observation)

    def step(self, state: Any, select: list[int]) -> Any:
        return self._step(state.searchId, select)

    def is_terminal(self, state: Any) -> bool:
        cur = state.observation.current
        return cur is not None and cur.result >= 0

    def reward(self, state: Any, player: int) -> float:
        cur = state.observation.current
        if cur is None or cur.result < 0:
            return 0.0
        if cur.result == player:
            return 1.0
        if cur.result in (0, 1):
            return -1.0
        return 0.0  # draw

    def to_play(self, state: Any) -> int:
        return state.observation.current.yourIndex


def get_forward_model(deck) -> EngineForwardModel | None:
    fm = EngineForwardModel(deck)
    return fm if fm.available() else None
