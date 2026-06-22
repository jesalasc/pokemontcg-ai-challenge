"""Net-guided PUCT search over the engine forward model (AlphaZero-style).

No rollouts, no heuristics: a leaf's value comes from the evaluator (the net).
Returns the visit-count policy π used both to pick the move and to train the net.

Evaluator contract:  evaluator(obs_dict, deck_vec) -> (priors[N], value)
  priors: probabilities over the obs's N legal options
  value:  scalar in [-1, 1] from the to-move player's perspective

All internal value bookkeeping is in player-0 perspective (zero-sum: p1 = -p0).
"""
from __future__ import annotations

import math

import numpy as np

C_PUCT = 1.5


class _Node:
    __slots__ = ("state", "obs", "to_play", "is_terminal", "n_opts",
                 "expanded", "P", "N", "W", "value_p0", "children")

    def __init__(self, state, fm):
        self.state = state
        self.is_terminal = fm.is_terminal(state)
        self.obs = None
        if self.is_terminal:
            self.to_play, self.n_opts = 0, 0
        else:
            self.obs = fm.obs_dict(state)
            self.to_play = self.obs["current"]["yourIndex"]
            sel = self.obs.get("select") or {}
            self.n_opts = len(sel.get("option", []))
        self.expanded = False
        self.children: dict[int, _Node] = {}
        self.P = self.N = self.W = None
        self.value_p0 = 0.0


def _expand(node: _Node, evaluator, deck_vec) -> None:
    priors, value = evaluator(node.obs, deck_vec)
    n = node.n_opts
    p = np.asarray(priors, dtype=np.float64)
    if p.shape[0] != n or not np.isfinite(p).all() or p.sum() <= 0:
        p = np.ones(n) / max(1, n)
    node.P = p
    node.N = np.zeros(n)
    node.W = np.zeros(n)
    node.value_p0 = value if node.to_play == 0 else -value
    node.expanded = True


def _select(node: _Node) -> int:
    sign = 1.0 if node.to_play == 0 else -1.0
    sqrt_sum = math.sqrt(node.N.sum() + 1e-8)
    q = np.where(node.N > 0, sign * node.W / np.maximum(node.N, 1.0), 0.0)
    u = C_PUCT * node.P * sqrt_sum / (1.0 + node.N)
    return int(np.argmax(q + u))


def run_search(fm, root_obs: dict, deck_vec, evaluator,
               sims: int = 64, temperature: float = 1.0):
    """Return (pi over the root's legal options, root value in p0 perspective)."""
    root = _Node(fm.root(root_obs), fm)
    if root.n_opts == 0:
        fm.end()
        return np.array([]), 0.0
    _expand(root, evaluator, deck_vec)

    for _ in range(sims):
        node = root
        path: list[tuple[_Node, int]] = []
        while True:
            if node.is_terminal:
                v0 = fm.reward(node.state, 0)
                break
            if not node.expanded:
                _expand(node, evaluator, deck_vec)
                v0 = node.value_p0
                break
            a = _select(node)
            child = node.children.get(a)
            if child is None:
                try:
                    child = _Node(fm.step(node.state, [a]), fm)
                    node.children[a] = child
                except Exception:
                    # multi-select / illegal single step here — drop this action
                    # and reselect, instead of aborting the whole search.
                    node.P[a] = 0.0
                    if not node.P.any():
                        v0 = node.value_p0
                        break
                    continue
            path.append((node, a))
            node = child
        for nd, a in path:
            nd.N[a] += 1.0
            nd.W[a] += v0

    visits = root.N
    if temperature <= 1e-6 or visits.sum() == 0:
        pi = np.zeros_like(visits)
        pi[int(np.argmax(visits))] = 1.0
    else:
        v = visits ** (1.0 / temperature)
        pi = v / v.sum()
    fm.end()
    return pi, root.value_p0


# --- evaluators -----------------------------------------------------------
def stub_evaluator(obs: dict, deck_vec) -> tuple[np.ndarray, float]:
    """Uniform priors, zero value — for testing the search plumbing without torch."""
    n = len((obs.get("select") or {}).get("option", []))
    return (np.ones(n) / n if n else np.array([])), 0.0


class NetEvaluator:
    """Wraps an AZNet for use as a search evaluator."""

    def __init__(self, net, device: str = "cpu"):
        self.net = net
        self.device = device

    def __call__(self, obs: dict, deck_vec) -> tuple[np.ndarray, float]:
        import torch

        from ptcg import features

        opts = features.encode_options(obs)
        if len(opts) == 0:
            return np.array([]), 0.0
        s = torch.from_numpy(features.encode_state(obs)).float().unsqueeze(0).to(self.device)
        o = torch.from_numpy(opts).float().unsqueeze(0).to(self.device)
        d = torch.from_numpy(np.asarray(deck_vec, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(s, d, o)
            p = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return p, float(value.item())
