"""End-to-end engine tests. Skipped automatically when the engine can't load
(i.e. anywhere that isn't Linux x86-64 — run these inside Docker)."""
from conftest import requires_engine


@requires_engine
def test_random_vs_random_completes_zero_sum():
    from harness.run_match import run_match

    res = run_match("random", "random")
    assert res["steps"] > 0
    assert not res["error"]
    # cabt is zero-sum: rewards are (+1,-1), (-1,+1) or (0,0).
    assert sum(r for r in res["rewards"] if isinstance(r, (int, float))) == 0


@requires_engine
def test_rule_based_beats_random_on_a_deck_it_can_pilot(monkeypatch):
    """Validate AGENT LOGIC on a simple deck (the generic baseline pilots well).

    On a complex Stage-2 deck like Dragapult the generic heuristic underperforms by
    design — that's a deck-specific-agent problem, not an agent-logic bug.
    """
    from pathlib import Path

    import ptcg.deck as deck
    from harness.evaluate import evaluate

    simple = Path(__file__).resolve().parents[1] / "data/decks/engine-default/deck.csv"
    if not simple.is_file():
        pytest.skip("engine-default deck not present")
    monkeypatch.setenv("PTCG_DECK", str(simple))
    deck.load_deck.cache_clear()
    try:
        rep = evaluate("rule_based", "random", n=12, verbose=False)
        assert rep["score"] > 0.6, rep
    finally:
        deck.load_deck.cache_clear()
