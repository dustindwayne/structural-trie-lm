from __future__ import annotations

from dataclasses import dataclass
from .fingerprint import FingerprintEncoder
from .trie import ContextTrie


@dataclass
class EvalResult:
    samples: int
    top1: float
    top5: float
    baseline_top1: float
    unique_words: int
    unique_fingerprints: int


def evaluate_text(text: str, order: int = 3, split: float = 0.8) -> EvalResult:
    enc = FingerprintEncoder()
    words = enc.tokenize(text)
    if len(words) < 10:
        raise ValueError("corpus is too small")

    cut = max(1, min(len(words) - 1, int(len(words) * split)))
    train_words, test_words = words[:cut], words[cut:]
    train_keys = [enc.encode(w).key for w in train_words]
    test_keys = [enc.encode(w).key for w in test_words]

    trie = ContextTrie(order=order)
    trie.fit(train_keys)

    baseline = trie.global_counts.most_common(1)[0][0]
    top1_hits = top5_hits = baseline_hits = 0
    context = train_keys[-order:]

    for target in test_keys:
        preds = [k for k, _ in trie.predict(context, top_k=5)]
        if preds and preds[0] == target:
            top1_hits += 1
        if target in preds:
            top5_hits += 1
        if target == baseline:
            baseline_hits += 1
        context = (context + [target])[-order:]

    n = len(test_keys)
    return EvalResult(
        samples=n,
        top1=top1_hits / n,
        top5=top5_hits / n,
        baseline_top1=baseline_hits / n,
        unique_words=len(set(words)),
        unique_fingerprints=len({enc.encode(w).key for w in words}),
    )
