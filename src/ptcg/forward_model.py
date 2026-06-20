"""Forward model abstraction for search (Layer 2).

The cabt engine hands the agent a ``search_begin_input`` blob every turn. With
the *official* engine download (see tools/fetch_engine.sh) the ``cg.sim`` module
exposes ``search_begin / search_step / search_end``, which let you clone the
current battle, have the engine SAMPLE the hidden information (opponent hand &
deck order) for you, and roll the clone forward without touching the real game.
That is exactly the determinization MCTS needs.

The pip-bundled engine (kaggle_environments==1.30.1) does NOT ship those search
functions. So this module:
  * exposes a clean ForwardModel interface MCTS codes against,
  * uses the engine's search API when it's available,
  * reports ``is_available() == False`` otherwise, so MCTS degrades to a
    1-ply heuristic instead of crashing.

When you pull the official engine + the sample MCTS kernel, wire the real calls
in ``EngineForwardModel`` (the integration points are marked TODO).
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class ForwardModel(Protocol):
    """What MCTS needs from a simulator. ``state`` is an opaque engine handle."""

    def root(self, obs: dict) -> Any: ...
    def sample_determinization(self, state: Any) -> Any: ...
    def observation(self, state: Any) -> dict: ...   # obs dict for the rollout policy
    def legal_actions(self, state: Any) -> list[Any]: ...
    def step(self, state: Any, action: list[int]) -> Any: ...
    def is_terminal(self, state: Any) -> bool: ...
    def reward(self, state: Any, player: int) -> float: ...
    def to_play(self, state: Any) -> int: ...


def _engine_search_available() -> bool:
    try:
        from kaggle_environments.envs.cabt.cg import sim  # type: ignore

        return all(hasattr(sim, fn) for fn in ("search_begin", "search_step", "search_end"))
    except Exception:
        return False


class EngineForwardModel:
    """Engine-backed forward model (determinized rollouts).

    Filled in once the official engine's search API is present. Until then,
    ``is_available()`` is False and callers must fall back.
    """

    def __init__(self) -> None:
        self._ok = _engine_search_available()

    def is_available(self) -> bool:
        return self._ok

    # --- ForwardModel interface -------------------------------------------
    # TODO(layer2): implement against cg.sim.search_* using obs["search_begin_input"].
    # Reference: kaggle kernel `kiyotah/reinforcement-learning-and-mcts-sample-code`.
    def root(self, obs: dict) -> Any:  # pragma: no cover - until engine wired
        raise NotImplementedError("Wire EngineForwardModel to cg.sim.search_* (see fetch_engine.sh).")

    def sample_determinization(self, state: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def legal_actions(self, state: Any) -> list[Any]:  # pragma: no cover
        raise NotImplementedError

    def step(self, state: Any, action: list[int]) -> Any:  # pragma: no cover
        raise NotImplementedError

    def is_terminal(self, state: Any) -> bool:  # pragma: no cover
        raise NotImplementedError

    def reward(self, state: Any, player: int) -> float:  # pragma: no cover
        raise NotImplementedError

    def to_play(self, state: Any) -> int:  # pragma: no cover
        raise NotImplementedError


def get_forward_model() -> Optional[EngineForwardModel]:
    """Return an available forward model, or None if search isn't supported."""
    fm = EngineForwardModel()
    return fm if fm.is_available() else None
