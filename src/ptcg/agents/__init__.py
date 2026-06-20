"""Agent registry.

Agents are looked up by name and imported lazily so that the baseline never
pays for heavy deps (e.g. torch is only imported when you ask for "rl").

    from ptcg.agents import get_agent
    play = get_agent("rule_based")   # callable (obs) -> list[int], play phase only

Wrap with ptcg.agents.base.make_safe before handing to the engine.
"""
from __future__ import annotations

from typing import Callable

PlayAgent = Callable[[dict], list]

# name -> "module:function" (lazy import target)
_REGISTRY: dict[str, str] = {
    "random": "ptcg.agents.random_agent:play",
    "rule_based": "ptcg.agents.rule_based:play",
    "mcts": "ptcg.agents.mcts:play",
    "rl": "ptcg.agents.rl_agent:play",
}


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_agent(name: str) -> PlayAgent:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown agent {name!r}. Available: {available()}")
    module_path, func_name = _REGISTRY[name].split(":")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, func_name)
