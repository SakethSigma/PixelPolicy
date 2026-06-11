# wordvocab — shared multi-length vocabulary

A tiny support package: the **one vocabulary** and the **game-salted split** that the new
single-turn word-skill games (char counts, validity, rhyme, crossword, …) draw from. It is not
a game — it has no server. It exists so every auxiliary game samples words from the same pool
and so a word held out for one game is trained in another (the cross-game design in
[`games/DATA_SOURCING.md`](../DATA_SOURCING.md)).

## What it produces

- **`vocab.txt`** — a committed, sorted word list = the full Wordle vocabulary
  (`games/wordle/{train,val}_words.txt`, 12,972 words) **unioned** with WordNet lemmas filtered
  to lowercase-alpha single tokens of length **3–20**. Committed like Wordle's word lists, so
  downstream packages read it directly and need **no `nltk`** at runtime.
- **`split.py::assign_pool(game, word)`** — the salted train/val rule
  `sha256(f"{game}:{word}") % 1000 < 200 → val else train`. A salted variant of
  `games/wordle/game.py::assign_pool`; deterministic and order-independent, so a bank derives
  its split at load time with no per-game artifact to commit.

## Regenerating `vocab.txt` (deliberate, run-on-purpose)

```bash
# one-time corpus download (then fully offline):
uv run --with nltk --package game-wordvocab python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# (re)build the committed asset — prints a length histogram:
uv run --with nltk --package game-wordvocab python -m games.wordvocab.build
```

`nltk` is only needed for this build step (it's the package's optional `[build]` extra);
nothing else in the repo imports it. Regenerating changes which words exist for every auxiliary
game, so do it on purpose and commit the result.

## Files

| File | Role |
|------|------|
| [`build.py`](build.py) | union Wordle vocab + WordNet lemmas (len 3–20) → committed `vocab.txt`; `load_vocab()` reader |
| [`split.py`](split.py) | `assign_pool(game, word)` — the shared, game-salted train/val rule |
| `vocab.txt` | the committed multi-length word list (regenerate with `build.py`) |
