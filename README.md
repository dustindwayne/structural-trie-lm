# Structural Trie LM Experiments

Experimental code for testing one question:

> Can cheap word-structure signals constrain next-word prediction enough to be useful before expensive neural computation?

This repository implements the earliest testable pieces of that idea without pretending the answer is already yes.

## What is here

- A deterministic word fingerprint encoder.
- Coarse vowel/consonant grouping.
- Markers for common beginnings, endings, clusters, and cheap function words.
- A context trie that learns which structural fingerprints tend to follow prior fingerprints.
- Evaluation utilities for measuring structural next-step prediction.

It is **not** a replacement tokenizer and it is **not** a production language model.

## Fingerprint rules

The default encoder follows the original experiment notes:

- vowels `a e i o u` share a wildcard class
- consonants are grouped as `bcd`, `fgh`, `jklmn`, `pqrst`, `vwxyz`
- common chunks such as `st`, `ck`, `ing` can collapse into structural markers
- common beginnings/endings are marked separately
- cheap function words such as `and`, `or`, `the`, `a`, `of`, `to`, `in`, `is`, `it`, `as`, `by`, `for` get a compact marker

The point is to preserve **shape** while deliberately discarding some spelling identity.

## Quick start

```bash
python -m structural_trie.cli fingerprint running rerun strangely
python -m structural_trie.cli demo
```

Train/evaluate on a text file:

```bash
python -m structural_trie.cli evaluate path/to/corpus.txt --order 3 --split 0.8
```

## What would count as evidence

Useful results would be things like:

- structural prediction accuracy above a frequency-only baseline
- smaller state than a raw-word trie at comparable predictive value
- repeatable improvement on held-out text

If it does not beat those baselines, the idea is not useful in this form. That is the experiment.

## AI assistance

AI was used for implementation, cleanup, and test scaffolding. The structural grouping concept and the broader constraint-first direction came from Dustin Wann's earlier experiments. This repository intentionally avoids claiming later AI-generated elaborations as original results.
