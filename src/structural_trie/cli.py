from __future__ import annotations

import argparse
from pathlib import Path
from .eval import evaluate_text
from .fingerprint import FingerprintEncoder
from .trie import ContextTrie


def cmd_fingerprint(words: list[str]) -> None:
    enc = FingerprintEncoder()
    for word in words:
        fp = enc.encode(word)
        print(f"{word:18} {fp.key}")


def cmd_demo() -> None:
    text = """
    the cat sat on the mat and the dog sat on the rug
    the cat ran to the door and the dog ran to the gate
    the runner was running and the walker was walking
    """
    enc = FingerprintEncoder()
    words = enc.tokenize(text)
    keys = [enc.encode(w).key for w in words]
    trie = ContextTrie(order=2)
    trie.fit(keys)
    ctx = keys[-2:]
    print("context:", ctx)
    print("next structural predictions:")
    for key, count in trie.predict(ctx):
        print(f"  {count:4}  {key}")


def cmd_evaluate(path: str, order: int, split: float) -> None:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    r = evaluate_text(text, order=order, split=split)
    print(f"samples              {r.samples}")
    print(f"top-1 structural     {r.top1:.4f}")
    print(f"top-5 structural     {r.top5:.4f}")
    print(f"frequency baseline   {r.baseline_top1:.4f}")
    print(f"unique words         {r.unique_words}")
    print(f"unique fingerprints  {r.unique_fingerprints}")


def main() -> None:
    p = argparse.ArgumentParser(description="Structural fingerprint + context-trie experiments")
    sub = p.add_subparsers(dest="cmd", required=True)

    fp = sub.add_parser("fingerprint")
    fp.add_argument("words", nargs="+")

    sub.add_parser("demo")

    ev = sub.add_parser("evaluate")
    ev.add_argument("path")
    ev.add_argument("--order", type=int, default=3)
    ev.add_argument("--split", type=float, default=0.8)

    a = p.parse_args()
    if a.cmd == "fingerprint":
        cmd_fingerprint(a.words)
    elif a.cmd == "demo":
        cmd_demo()
    else:
        cmd_evaluate(a.path, a.order, a.split)


if __name__ == "__main__":
    main()
