from __future__ import annotations

from dataclasses import dataclass
import re

VOWELS = set("aeiou")
GROUPS = {
    **{c: "G0" for c in "bcd"},
    **{c: "G1" for c in "fgh"},
    **{c: "G2" for c in "jklmn"},
    **{c: "G3" for c in "pqrst"},
    **{c: "G4" for c in "vwxyz"},
}

PREFIXES = ("re", "un", "pre", "pro", "con", "dis")
SUFFIXES = ("tion", "ness", "ment", "less", "able", "ible", "ing", "ed", "er", "ly", "s", "es")
CLUSTERS = ("ing", "st", "ck")
CHEAP_WORDS = {"and", "or", "the", "a", "an", "of", "to", "in", "is", "it", "as", "by", "for"}

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.I)


@dataclass(frozen=True)
class Fingerprint:
    word: str
    shape: tuple[str, ...]
    prefix: str | None
    suffix: str | None
    cheap: bool

    @property
    def key(self) -> str:
        if self.cheap:
            return f"CHEAP:{self.word}"
        p = self.prefix or "-"
        s = self.suffix or "-"
        return f"P={p}|S={s}|{'/'.join(self.shape)}"


class FingerprintEncoder:
    def tokenize(self, text: str) -> list[str]:
        return [m.group(0).lower() for m in _WORD_RE.finditer(text)]

    def encode(self, word: str) -> Fingerprint:
        w = word.lower()
        prefix = next((p for p in PREFIXES if w.startswith(p) and len(w) > len(p)), None)
        suffix = next((s for s in SUFFIXES if w.endswith(s) and len(w) > len(s)), None)

        shape: list[str] = []
        i = 0
        while i < len(w):
            cluster = next((c for c in CLUSTERS if w.startswith(c, i)), None)
            if cluster:
                shape.append(f"C:{cluster}")
                i += len(cluster)
                continue

            ch = w[i]
            if ch in VOWELS:
                shape.append("V")
            elif ch in GROUPS:
                shape.append(GROUPS[ch])
            else:
                shape.append("?")
            i += 1

        return Fingerprint(w, tuple(shape), prefix, suffix, w in CHEAP_WORDS)
