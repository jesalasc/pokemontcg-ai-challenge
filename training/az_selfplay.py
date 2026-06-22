"""Population self-play game generation for AlphaZero training.

Pairs two decks from the roster and plays a game where both seats search with the
(shared) evaluator, each conditioned on its own deck. Records (state, deck, π,
player) per move and labels every sample with the terminal result (win/loss/draw)
— terminal reward only, no shaping.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ptcg import features  # noqa: E402
from ptcg.deckcode import deck_vector  # noqa: E402
from ptcg.forward_model import EngineForwardModel  # noqa: E402
from training.az_mcts import run_search  # noqa: E402


def _topk_from_pi(pi: np.ndarray, k: int, n: int) -> list[int]:
    order = list(np.argsort(pi)[::-1])
    return order[:k] if k > 1 else [int(order[0])]


def _make_agent(evaluator, my_deck, opp_deck, sink, sims, temperature):
    deck_vec = deck_vector(my_deck)
    fm = EngineForwardModel(my_deck, opponent_deck=opp_deck)

    def agent(obs: dict):
        if obs.get("select") is None:
            return list(my_deck)
        sel = obs["select"]
        n = len(sel.get("option", []))
        k = min(int(sel.get("maxCount", 1)), n)
        if not fm.available() or obs.get("search_begin_input") is None or k <= 0:
            return list(range(k))
        try:
            pi, _ = run_search(fm, obs, deck_vec, evaluator, sims=sims, temperature=temperature)
            if pi.size == 0:
                return list(range(k))
            sink.append({
                "state": features.encode_state(obs),
                "deck": deck_vec,
                "options": features.encode_options(obs),
                "pi": pi.astype(np.float32),
                "player": int(obs["current"]["yourIndex"]),
            })
            if k == 1:
                return [int(np.random.choice(len(pi), p=pi))]
            return _topk_from_pi(pi, k, n)
        except Exception:
            return list(range(k))

    return agent


def play_game(evaluator, deck_a, deck_b, sims: int = 32, temperature: float = 1.0):
    """Play one self-play game. Returns (samples, z0) where z0 is seat-0's result."""
    from kaggle_environments import make

    sink: list[dict] = []
    a = _make_agent(evaluator, deck_a, deck_b, sink, sims, temperature)
    b = _make_agent(evaluator, deck_b, deck_a, sink, sims, temperature)
    env = make("cabt")
    env.run([a, b])
    z0 = env.state[0]["reward"]
    z0 = z0 if isinstance(z0, (int, float)) else 0.0
    for s in sink:
        s["z"] = float(z0 if s["player"] == 0 else -z0)
    return sink, z0
