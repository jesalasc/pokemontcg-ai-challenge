"""Crash safety is the #1 ladder rule: a crash / illegal move = instant loss.

These verify make_safe turns any misbehaving brain into a legal selection.
"""
from ptcg.agents.base import make_safe, sanitize
from ptcg.deck import load_deck

DECK = load_deck()


def _play_obs(n_options=3, max_count=1):
    return {
        "select": {"option": [{"type": i} for i in range(n_options)], "maxCount": max_count},
        "current": {"yourIndex": 0, "result": -1, "players": []},
        "logs": [],
    }


def test_deck_selection_returns_60():
    agent = make_safe(lambda o: [], DECK)
    assert agent({"select": None}) == list(DECK)
    assert len(agent({"select": None})) == 60


def test_raising_agent_falls_back_to_legal():
    def boom(obs):
        raise RuntimeError("kaboom")

    agent = make_safe(boom, DECK)
    out = agent(_play_obs(3, 1))
    assert out == [0]  # first legal option


def test_out_of_range_indices_sanitized():
    agent = make_safe(lambda o: [99], DECK)
    out = agent(_play_obs(3, 1))
    assert out and all(0 <= i < 3 for i in out)


def test_wrong_length_sanitized_to_maxcount():
    agent = make_safe(lambda o: [0, 1, 2], DECK)  # too many for maxCount=1
    assert len(agent(_play_obs(3, 1))) == 1


def test_multiselect_padded_to_maxcount():
    agent = make_safe(lambda o: [0], DECK)  # too few for maxCount=2
    out = agent(_play_obs(4, 2))
    assert len(out) == 2 and len(set(out)) == 2


def test_non_list_return_falls_back():
    agent = make_safe(lambda o: "not a list", DECK)
    out = agent(_play_obs(3, 1))
    assert out == [0]


def test_duplicate_indices_deduped():
    out = sanitize([1, 1, 1], _play_obs(4, 2))
    assert out is not None and len(set(out)) == len(out) == 2
