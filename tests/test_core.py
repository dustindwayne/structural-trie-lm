from structural_trie import FingerprintEncoder, ContextTrie


def test_vowels_share_structure():
    e = FingerprintEncoder()
    assert e.encode("cat").shape == e.encode("cot").shape


def test_common_function_words_marked_cheap():
    e = FingerprintEncoder()
    assert e.encode("the").cheap
    assert not e.encode("river").cheap


def test_trie_learns_context():
    t = ContextTrie(order=2)
    t.fit(["A", "B", "C", "A", "B", "C", "A", "B", "D"])
    pred = t.predict(["A", "B"], top_k=2)
    assert pred[0][0] == "C"
