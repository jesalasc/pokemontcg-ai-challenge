"""AlphaZero play agent: deck-conditioned, net-guided search.

Loads a trained AZNet and runs net-guided PUCT over the engine forward model,
conditioned on our deck. If the net or engine isn't available, it falls back to a
legal move (no hand-coded strategy). Needs cg.api + torch (bundle for submission).
"""
from __future__ import annotations

import os
from functools import lru_cache

from ptcg import protocol as P


@lru_cache(maxsize=1)
def _evaluator():
    path = os.environ.get("PTCG_AZ_CHECKPOINT", os.path.join("checkpoints", "az.pt"))
    if not os.path.isfile(path):
        return None
    try:
        import torch

        from training.az_mcts import NetEvaluator
        from training.networks import AZNet

        ckpt = torch.load(path, map_location="cpu")
        net = AZNet(*ckpt["dims"])
        net.load_state_dict(ckpt["model"])
        net.eval()
        return NetEvaluator(net, "cpu")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _forward_model():
    from ptcg.deck import load_deck
    from ptcg.forward_model import EngineForwardModel

    fm = EngineForwardModel(load_deck())
    return fm if fm.available() else None


@lru_cache(maxsize=1)
def _deck_vec():
    from ptcg.deck import load_deck
    from ptcg.deckcode import deck_vector

    return deck_vector(load_deck())


def play(obs: dict) -> list[int]:
    sel = obs.get("select") or {}
    opts = sel.get("option", [])
    k = min(int(sel.get("maxCount", 1)), len(opts))
    if k <= 0:
        return []

    ev, fm = _evaluator(), _forward_model()
    if ev is None or fm is None or obs.get("search_begin_input") is None:
        return P.legal_fallback(obs)

    try:
        import numpy as np

        from training.az_mcts import run_search

        sims = int(os.environ.get("PTCG_AZ_SIMS", "64"))
        pi, _ = run_search(fm, obs, _deck_vec(), ev, sims=sims, temperature=0.0)
        if pi.size == 0:
            return P.legal_fallback(obs)
        order = [int(i) for i in np.argsort(pi)[::-1]]
        return order[:k]
    except Exception:
        return P.legal_fallback(obs)
