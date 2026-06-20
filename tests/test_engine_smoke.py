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
def test_rule_based_beats_random_over_a_few_games():
    """Sanity gate: the baseline should clearly beat random (it scored ~0.85)."""
    from harness.evaluate import evaluate

    rep = evaluate("rule_based", "random", n=12, verbose=False)
    assert rep["score"] > 0.6, rep
