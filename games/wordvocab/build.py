"""Build the shared multi-length vocabulary asset — ``vocab.txt``.

Unions the **full Wordle vocabulary** (``train_words.txt`` + ``val_words.txt``, all 12,972
five-letter words) with **WordNet lemmas** filtered to lowercase-alpha single tokens of
length 3-20, giving the length variety the auxiliary games (char counts, validity, rhyme,
crossword) need. The result is committed as ``vocab.txt`` so downstream packages read it
directly and never need ``nltk`` or a corpus download at runtime — exactly how Wordle commits
its split.

    # one-time corpus download (then offline forever):
    uv run --package game-wordvocab python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
    # (re)generate the committed asset:
    uv run --package game-wordvocab python -m games.wordvocab.build

Regenerating is a deliberate, run-on-purpose step (it changes which words exist for every
auxiliary game). WordNet is the only source that both varies length *and* guarantees a
definition for every word (needed later by validity/crossword).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

_DIR = Path(__file__).parent
_VOCAB_FILE = _DIR / "vocab.txt"

# Wordle's committed split files are the source of its full vocabulary (their union).
_WORDLE_DIR = _DIR.parent / "wordle"
_WORDLE_FILES = (_WORDLE_DIR / "train_words.txt", _WORDLE_DIR / "val_words.txt")

MIN_LEN = 3
MAX_LEN = 20


def load_vocab(path: Path = _VOCAB_FILE) -> list[str]:
    """Read the committed multi-length vocabulary (sorted, one lowercase word per line)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Vocabulary asset missing ({path}). Generate it with: "
            "python -m games.wordvocab.build"
        )
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if word and not word.startswith("#"):
            words.append(word)
    return words


def _load_wordle_words() -> set[str]:
    """The full Wordle vocabulary = union of its committed train/val artifacts."""
    words: set[str] = set()
    for path in _WORDLE_FILES:
        if not path.exists():
            raise FileNotFoundError(
                f"Wordle split file missing ({path}). Generate it with: "
                "python -m games.wordle.split"
            )
        for line in path.read_text(encoding="utf-8").splitlines():
            w = line.strip().lower()
            if w and not w.startswith("#"):
                words.add(w)
    return words


def _wordnet_words(min_len: int = MIN_LEN, max_len: int = MAX_LEN) -> set[str]:
    """All WordNet lemmas that are single lowercase-alpha tokens in the length range.

    Multi-word lemmas (underscore-joined), hyphenated forms, and anything with digits are
    dropped; lemmas are lowercased so proper-noun casing doesn't fragment the set.
    """
    from nltk.corpus import wordnet as wn  # lazy: only the build needs nltk

    out: set[str] = set()
    for lemma in wn.all_lemma_names():
        w = lemma.lower()
        if w.isalpha() and min_len <= len(w) <= max_len:
            out.add(w)
    return out


def build(out_path: Path = _VOCAB_FILE) -> list[str]:
    """Union Wordle vocab + WordNet lemmas, write the sorted committed asset, return it."""
    wordle = _load_wordle_words()
    wordnet = _wordnet_words()
    vocab = sorted((wordle | wordnet))
    out_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    hist = Counter(len(w) for w in vocab)
    print(f"wrote {len(vocab)} words -> {out_path}")
    print(f"  wordle-origin: {len(wordle)}   wordnet-added: {len(wordnet - wordle)}")
    print("  length histogram:")
    for n in range(MIN_LEN, MAX_LEN + 1):
        if hist.get(n):
            bar = "#" * min(60, hist[n] * 60 // max(hist.values()))
            print(f"    {n:>2}: {hist[n]:>6}  {bar}")
    return vocab


if __name__ == "__main__":
    build()
