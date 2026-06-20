"""Simulator-vs-official-rules regression tests.

The brief flags this as a top risk: the cabt simulator differs from paper PTCG
rules in documented ways, and matches are won/lost on those details. Every time
the domain expert finds a divergence (read the competition's "differences" doc
and watch replays), encode it here as a test so we never regress on it.

Read first:  the "simulator vs official rules" doc on the competition page, and
the sample kernels (tools/pull_samples.sh).
"""
import pytest

# Each entry becomes a real test once verified against the engine. Examples of
# the KINDS of things to pin (replace with confirmed behaviors):
KNOWN_DIFFERENCES = [
    "coin-flip outcome distribution / seeding behavior",
    "mulligan handling and opening-hand redraws",
    "exact prize count on multi-prize (ex/V) knockouts",
    "status-condition timing (poison/burn between turns, sleep/paralysis checks)",
    "deck-out / turn-limit resolution and who wins ties",
]


@pytest.mark.parametrize("difference", KNOWN_DIFFERENCES)
@pytest.mark.xfail(reason="TODO: confirm against engine + sample kernels, then assert", strict=False)
def test_documented_difference(difference):
    raise NotImplementedError(f"Encode and assert engine behavior for: {difference}")
