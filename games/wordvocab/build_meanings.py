"""Build the shared word → definition asset — ``meanings.jsonl``.

For every word in the committed ``vocab.txt`` that WordNet knows, store its first-sense gloss
(``wn.synsets(w)[0].definition()``). The result is committed alongside ``vocab.txt`` so the
validity game (and, later, crossword) read meanings with **no ``nltk`` or corpus download at
runtime** — exactly how ``vocab.txt`` itself is committed. ``nltk`` is only the wordvocab
``[build]`` extra, needed by this script.

    # one-time corpus download (then offline forever):
    uv run --package game-wordvocab python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
    # (re)generate the committed asset:
    uv run --package game-wordvocab python -m games.wordvocab.build_meanings

Regenerating is a deliberate, run-on-purpose step (it changes which words carry a definition).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from games.wordvocab.build import load_vocab

_DIR = Path(__file__).parent
_MEANINGS_FILE = _DIR / "meanings.jsonl"


def load_meanings(path: Path = _MEANINGS_FILE) -> dict[str, str]:
    """Read the committed word → definition map (one ``{"word","definition"}`` JSON per line)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Meanings asset missing ({path}). Generate it with: "
            "python -m games.wordvocab.build_meanings"
        )
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rec = json.loads(line)
        out[rec["word"]] = rec["definition"]
    return out


def build(out_path: Path = _MEANINGS_FILE, vocab_path: Optional[Path] = None) -> dict[str, str]:
    """For each vocab word with a WordNet synset, write its first-sense definition; return the map."""
    from nltk.corpus import wordnet as wn  # lazy: only the build needs nltk

    words = load_vocab(vocab_path) if vocab_path is not None else load_vocab()
    meanings: dict[str, str] = {}
    for w in words:
        synsets = wn.synsets(w)
        if synsets:
            definition = synsets[0].definition().strip()
            if definition:
                meanings[w] = definition

    with out_path.open("w", encoding="utf-8") as f:
        for w in sorted(meanings):
            f.write(json.dumps({"word": w, "definition": meanings[w]}, ensure_ascii=False) + "\n")

    hist = Counter(len(w) for w in meanings)
    print(f"wrote {len(meanings)} definitions -> {out_path}")
    print(f"  vocab words: {len(words)}   with a definition: {len(meanings)} "
          f"({len(meanings) * 100 // max(1, len(words))}%)")
    print("  length spread:", {n: hist[n] for n in sorted(hist)})
    return meanings


if __name__ == "__main__":
    build()
