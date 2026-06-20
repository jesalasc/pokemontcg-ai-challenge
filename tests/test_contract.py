"""Protocol/agent contract: agents always return legal index lists."""
import random

import pytest

from ptcg import protocol as P
from ptcg.agents import available, get_agent
from ptcg.agents.base import make_safe
from ptcg.deck import load_deck

DECK = load_deck()


def random_play_obs(rng):
    n = rng.randint(1, 8)
    max_count = rng.randint(1, min(3, n))
    ctx = rng.choice([0, 1, 30, 38, 41])
    opts = []
    for i in range(n):
        o = {"type": rng.randint(0, 14)}
        if ctx == 0 and rng.random() < 0.3:
            o["attackId"] = rng.randint(1, 9)
        if rng.random() < 0.5:
            o.update(area=rng.randint(1, 7), index=i, playerIndex=rng.randint(0, 1))
        opts.append(o)
    return {
        "select": {"context": ctx, "option": opts, "maxCount": max_count,
                   "minCount": 1, "remainDamageCounter": 0, "remainEnergyCost": 0},
        "current": {"yourIndex": 0, "result": -1, "turn": rng.randint(1, 20),
                    "players": [_player(), _player()]},
        "logs": [],
    }


def _player():
    return {"active": [{"id": 1, "hp": 60, "maxHp": 90, "energies": []}],
            "bench": [], "hand": [], "discard": [], "prize": [1, 2, 3, 4, 5, 6],
            "handCount": 5, "deckCount": 40, "benchMax": 5,
            "poisoned": False, "burned": False, "asleep": False,
            "paralyzed": False, "confused": False}


@pytest.mark.parametrize("name", ["random", "rule_based", "mcts", "rl"])
def test_agent_returns_legal_selection(name):
    agent = make_safe(get_agent(name), DECK)
    rng = random.Random(0)
    for _ in range(300):
        obs = random_play_obs(rng)
        out = agent(obs)
        n = len(P.options(obs))
        k = min(P.max_count(obs), n)
        assert isinstance(out, list)
        assert len(out) == k
        assert len(set(out)) == len(out)
        assert all(isinstance(i, int) and 0 <= i < n for i in out)


def test_registry_lists_all_agents():
    assert set(available()) >= {"random", "rule_based", "mcts", "rl"}


def test_rule_based_develops_before_attacking():
    """In MAIN, a develop action must outrank an attack (which ends the turn)."""
    from ptcg.agents import rule_based

    obs = {
        "select": {"context": 0, "maxCount": 1, "option": [
            {"type": 13, "attackId": 5},   # attack
            {"type": 8},                    # develop
            {"type": 14},                   # end turn
        ]},
        "current": {"yourIndex": 0, "result": -1, "players": [_player(), _player()]},
        "logs": [],
    }
    assert rule_based.play(obs) == [1]  # the develop action, not attack/end
