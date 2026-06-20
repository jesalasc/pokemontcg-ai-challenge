from ptcg.deck import DECK_SIZE, load_deck


def test_deck_is_60_ints():
    deck = load_deck()
    assert len(deck) == DECK_SIZE == 60
    assert all(isinstance(c, int) for c in deck)


def test_deck_is_cached_tuple():
    assert isinstance(load_deck(), tuple)
    assert load_deck() is load_deck()  # lru_cache
