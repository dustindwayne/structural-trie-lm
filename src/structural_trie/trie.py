from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class _Node:
    next_counts: Counter[str] = field(default_factory=Counter)
    children: dict[str, "_Node"] = field(default_factory=dict)


class ContextTrie:
    """N-gram-like context trie over structural fingerprint keys."""

    def __init__(self, order: int = 3):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.root = _Node()
        self.global_counts: Counter[str] = Counter()

    def fit(self, sequence: list[str]) -> None:
        self.global_counts.update(sequence)
        for i, target in enumerate(sequence):
            start = max(0, i - self.order)
            context = sequence[start:i]
            for suffix_start in range(len(context) + 1):
                node = self.root
                for key in context[suffix_start:]:
                    node = node.children.setdefault(key, _Node())
                node.next_counts[target] += 1

    def predict(self, context: list[str], top_k: int = 5) -> list[tuple[str, int]]:
        context = context[-self.order:]
        for start in range(0, len(context) + 1):
            node = self.root
            ok = True
            for key in context[start:]:
                node = node.children.get(key)
                if node is None:
                    ok = False
                    break
            if ok and node.next_counts:
                return node.next_counts.most_common(top_k)
        return self.global_counts.most_common(top_k)
