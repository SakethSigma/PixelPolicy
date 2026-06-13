# Character set

Word-skill game **#7** (see [`games/DATA_SOURCING.md`](../DATA_SOURCING.md)). A **single-turn**
environment: `reset()` poses a small list of words, `step(answer)` scores the submitted **used**
and **unused** letters of the alphabet and ends the episode. It teaches the model to aggregate
letter coverage across several words — tracking which letters of a–z are in play (directly useful
for Wordle, where you reason about which letters remain).

Same architecture as [Wordle](../wordle/README.md) and [charcount](../charcount/README.md): a
**pure core** that a human can play and an RL trainer can step in-process by the thousands,
through the same code path, so feedback and rules never drift. The env knows nothing about
models, tokens, or rewards.

## Single-turn status convention

Mirrors Wordle's `"won"` so the distillation rejection filter stays game-agnostic:

```
"in_progress"  → words posed, no answer yet
"correct"      → submitted used AND unused sets match the truth   (the solved/"good" status)
"incorrect"    → either set wrong, or the answer was unparseable
```

Ground truth is pure Python (no corpora): `used` = the union of letters across all the words;
`unused` = the 26-letter alphabet minus `used`. **Both** sets must match to score `correct`.

## Quick start

```bash
# play in the terminal (in-process, no server)
uv run --package game-charset python -m games.charset.play
uv run --package game-charset python -m games.charset.play --words cat,planet

# run the HTTP server, then play the SAME game over HTTP
uv run --package game-charset uvicorn games.charset.server:app
uv run --package game-charset python -m games.charset.play --http http://127.0.0.1:8000

# tests (core scoring, render round-trip, Local/HTTP parity)
uv run --package game-charset pytest games/charset/tests/ -q
```

> Requires `games/wordvocab/vocab.txt` (the shared word pool). Build it once with
> `python -m games.wordvocab.build` — see [wordvocab](../wordvocab/README.md).

## Data model

`GameState` is what every endpoint and client method returns:

```jsonc
{
  "game_id": "uuid",
  "words": ["cat", "dog"],          // the challenge words the model analyzes
  "status": "correct",              // "in_progress" | "correct" | "incorrect"
  "submitted": "used (5): A C D...", // the raw answer step() scored (null before step)
  "solution": {                     // revealed only once status != in_progress
    "used":   ["a","c","d","g","o","t"],
    "unused": ["b","e","f","h","i","j","k","l","m","n","p","q","r","s","u","v","w","x","y","z"]
  }
}
```

The canonical `<answer>` block (what the synthetic teacher writes and a human types) — letters
are space-separated and UPPERCASE. For `{cat, dog}`:

```
<answer>
used (5): A C D G O T
unused (21): B E F H I J K L M N P Q R S U V W X Y Z
</answer>
```

`game.py::parse_answer` reads this format back tolerantly (any order, optional `(count)`
parentheticals, commas/spaces, any case); **both** the `used` and `unused` lines must be present,
and a malformed answer scores `incorrect` — the same "malformed costs you the round" contract as
Wordle.

## HTTP API

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `POST` | `/reset` | `{"mode":"train"\|"val","words":["cat","dog"]?}` | New episode; `words` pins the challenge, else one is sampled. |
| `POST` | `/step` | `{"game_id":"...","answer":"used (5): ..."}` | Returns the scored terminal `GameState`. |
| `GET`  | `/state/{game_id}` | — | Observe without acting. |

`404` unknown `game_id`; `400` stepping an already-finished episode.

## Word pool & split

`CharsetBank` loads the shared [`games/wordvocab/vocab.txt`](../wordvocab/README.md) and derives
its own train/val split via the salted rule `assign_pool("charset", word)` — deterministic, so no
per-game split file is committed. Each challenge **mixes lengths**: **one five-letter Wordle word
plus 1–3 non-five-letter words** (2–4 words total), so a challenge always combines familiar
Wordle vocabulary with varied-length general words. The salt decouples charset's split from
Wordle's own split.

## Files

```
games/charset/
├── game.py     # pure core: CharsetGame, CharsetBank, analyze/encode_words/parse_answer, models
├── render.py   # render_observation (the prompt) + render_answer (canonical used/unused block)
├── client.py   # CharsetClient Protocol + Local/HTTP transports (verb: step)
├── server.py   # async FastAPI: /reset /step /state
├── play.py     # terminal play (in-process or --http; --words)
└── tests/      # test_game / test_render / test_client
```

## Training data

Character set is **programmatic** (the label is cheap and exact, no reasoning wanted), so its SFT
data is produced by [`distillation/programmatic.py`](../../distillation/README.md) — not Claude:

```bash
uv run --package distillation python -m distillation.programmatic --game charset
```

The built default emits **12,000 rows**, all valid by construction, with an even spread of 2-, 3-,
and 4-word challenges. See that module for the generator and the unified dataset schema.
