# Character counts

Word-skill game **#1** (see [`games/DATA_SOURCING.md`](../DATA_SOURCING.md)). A **single-turn**
environment: `reset()` poses one word, `step(answer)` scores the submitted character analysis
(length + vowel/consonant split) and ends the episode. It teaches the model to map a word to
its characters — the word↔token boundary and vowel/consonant awareness.

Same architecture as [Wordle](../wordle/README.md): a **pure core** that a human can play and an
RL trainer can step in-process by the thousands, through the same code path, so feedback and
rules never drift. The env knows nothing about models, tokens, or rewards.

## Single-turn status convention

Mirrors Wordle's `"won"` so the distillation rejection filter stays game-agnostic:

```
"in_progress"  → challenge posed, no answer yet
"correct"      → step()'s answer matched the computed analysis   (the solved/"good" status)
"incorrect"    → step()'s answer did not match (or was unparseable)
```

Ground truth is pure Python (no corpora): classify each letter against `aeiou`, keeping repeats,
so `length == #vowels + #consonants` always holds. `y` is a consonant.

## Quick start

```bash
# play in the terminal (in-process, no server)
uv run --package game-charcount python -m games.charcount.play
uv run --package game-charcount python -m games.charcount.play --word planet

# run the HTTP server, then play the SAME game over HTTP
uv run --package game-charcount uvicorn games.charcount.server:app
uv run --package game-charcount python -m games.charcount.play --http http://127.0.0.1:8000

# tests (core scoring, render round-trip, Local/HTTP parity)
uv run --package game-charcount pytest games/charcount/tests/ -q
```

> Requires `games/wordvocab/vocab.txt` (the shared word pool). Build it once with
> `python -m games.wordvocab.build` — see [wordvocab](../wordvocab/README.md).

## Data model

`GameState` is what every endpoint and client method returns:

```jsonc
{
  "game_id": "uuid",
  "word": "planet",                 // the challenge the model analyzes
  "status": "correct",              // "in_progress" | "correct" | "incorrect"
  "submitted": "length: 6\n...",    // the raw answer step() scored (null before step)
  "solution": {                     // revealed only once status != in_progress
    "length": 6, "vowels": ["a","e"], "consonants": ["p","l","n","t"]
  }
}
```

The canonical `<answer>` block (what the synthetic teacher writes and a human types) — letters
are space-separated and UPPERCASE:

```
length: 6
vowels (2): A E
consonants (4): P L N T
```

`game.py::parse_answer` reads this format back tolerantly (any order, optional `(count)`
parentheticals, commas/spaces, any case); a malformed answer scores `incorrect` — the same
"malformed costs you the round" contract as Wordle.

## HTTP API

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `POST` | `/reset` | `{"mode":"train"\|"val","word":"planet"?}` | New episode; `word` pins the challenge. |
| `POST` | `/step` | `{"game_id":"...","answer":"length: 6\n..."}` | Returns the scored terminal `GameState`. |
| `GET`  | `/state/{game_id}` | — | Observe without acting. |

`404` unknown `game_id`; `400` stepping an already-finished episode.

## Word pool & split

`CharCountBank` loads the shared [`games/wordvocab/vocab.txt`](../wordvocab/README.md) and derives
its own train/val split via the salted rule `assign_pool("charcount", word)` — deterministic, so
no per-game split file is committed. Wordle's own train *and* val words mostly land in
charcount's **train** pool (the salt decouples the splits): the model learns to analyze the very
words it plays Wordle on, which is the point, and does not leak Wordle eval (a different skill).

## Files

```
games/charcount/
├── game.py     # pure core: CharCountGame, CharCountBank, analyze/parse/score, models
├── render.py   # render_observation (the prompt) + render_answer (canonical block)
├── client.py   # CharCountClient Protocol + Local/HTTP transports (verb: step)
├── server.py   # async FastAPI: /reset /step /state
├── play.py     # terminal play (in-process or --http)
└── tests/      # test_game / test_render / test_client
```

## Training data

Char counts is **programmatic** (the label is cheap and exact, no reasoning wanted), so its SFT
data is produced by [`distillation/programmatic.py`](../../distillation/README.md) — not Claude.
See that module for the generator and the unified dataset schema.
