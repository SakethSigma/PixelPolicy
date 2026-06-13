# wordvocab — shared multi-length vocabulary

A tiny support package: the **one vocabulary**, the **word → meaning asset**, and the
**game-salted split** that the new single-turn word-skill games (char counts, validity, anagram,
rhyme, crossword, …) draw from. It is not a game — it has no server. It exists so every auxiliary game samples words from the same pool
and so a word held out for one game is trained in another (the cross-game design in
[`games/DATA_SOURCING.md`](../DATA_SOURCING.md)).

## What it produces

- **`vocab.txt`** — a committed, sorted word list = the full Wordle vocabulary
  (`games/wordle/{train,val}_words.txt`, 12,972 words) **unioned** with WordNet lemmas filtered
  to lowercase-alpha single tokens of length **3–20**. Committed like Wordle's word lists, so
  downstream packages read it directly and need **no `nltk`** at runtime.
- **`meanings.jsonl`** — a committed word → definition map (one `{"word","definition"}` JSON per
  line). For every word in `vocab.txt` that WordNet knows, it stores the first-sense gloss
  (`wn.synsets(w)[0].definition()`). Built once by `build_meanings.py`; read at runtime with
  `load_meanings()` and **no `nltk`**, exactly like `vocab.txt`. This is the validity oracle (a
  word is *valid* iff it appears here) and the source of the gold `<meaning>`. Used by the
  [validity](../validity/README.md) game (and, later, crossword).
- **`split.py::assign_pool(game, word)`** — the salted train/val rule
  `sha256(f"{game}:{word}") % 1000 < 200 → val else train`. A salted variant of
  `games/wordle/game.py::assign_pool`; deterministic and order-independent, so a bank derives
  its split at load time with no per-game artifact to commit.

## Regenerating the committed assets (deliberate, run-on-purpose)

```bash
# one-time corpus download (then fully offline):
uv run --with nltk --package game-wordvocab python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# (re)build vocab.txt — prints a length histogram:
uv run --with nltk --package game-wordvocab python -m games.wordvocab.build

# (re)build meanings.jsonl — prints how many vocab words carry a definition:
uv run --with nltk --package game-wordvocab python -m games.wordvocab.build_meanings
```

`nltk` is only needed for these build steps (it's the package's optional `[build]` extra);
nothing else in the repo imports it. Regenerating changes which words exist (or carry a meaning)
for the auxiliary games, so do it on purpose and commit the result.

## Files

| File | Role |
|------|------|
| [`build.py`](build.py) | union Wordle vocab + WordNet lemmas (len 3–20) → committed `vocab.txt`; `load_vocab()` reader |
| [`build_meanings.py`](build_meanings.py) | WordNet first-sense gloss per vocab word → committed `meanings.jsonl`; `load_meanings()` reader (no nltk at runtime) |
| [`split.py`](split.py) | `assign_pool(game, word)` — the shared, game-salted train/val rule |
| `vocab.txt` | the committed multi-length word list (regenerate with `build.py`) |
| `meanings.jsonl` | the committed word → WordNet definition map (regenerate with `build_meanings.py`) |
