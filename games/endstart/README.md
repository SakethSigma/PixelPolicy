# Ends-with → starts-with

Word-skill game **#4** (see [`games/DATA_SOURCING.md`](../DATA_SOURCING.md)). A **single-turn**
environment: `reset()` poses one word + 5 candidates, `step(answer)` scores the chosen candidate
and ends the episode. It teaches the model **first/last-character attention** — map a word to its
last letter, then find the candidate that starts with it.

This is the **last** of the original six word-skill games to be built (it was the lone spec left
in [DATA_SOURCING.md](../DATA_SOURCING.md)); none of the original family now remains unbuilt.

Same architecture as [Wordle](../wordle/README.md): a **pure core** that a human can play and an
RL trainer can step in-process by the thousands, through the same code path, so feedback and
rules never drift. The env knows nothing about models, tokens, or rewards.

## The challenge

Given a `word1` and **five** candidate words, choose the single candidate whose **first** letter
equals `word1`'s **last** letter. Exactly one candidate matches; the other four are distractors
that start with *different* letters, so the answer is unique. The candidate order is **shuffled**,
so the correct position is unbiased.

## Single-turn status convention

Mirrors Wordle's `"won"` so the distillation rejection filter stays game-agnostic:

```
"in_progress"  → challenge posed, no answer yet
"correct"      → step()'s answer is the candidate starting with word1's last letter   (the solved/"good" status)
"incorrect"    → step()'s answer did not match (or was unparseable)
```

Ground truth is pure Python (no corpora, no `<think>` requested): `word1[-1] == candidate[0]`.

## Quick start

```bash
# play in the terminal (in-process, no server)
uv run --package game-endstart python -m games.endstart.play

# run the HTTP server, then play the SAME game over HTTP
uv run --package game-endstart uvicorn games.endstart.server:app
uv run --package game-endstart python -m games.endstart.play --http http://127.0.0.1:8000

# tests (core scoring, render round-trip, Local/HTTP parity) — 15 passing
uv run --package game-endstart pytest games/endstart/tests/ -q
```

> Requires `games/wordvocab/vocab.txt` (the shared word pool). Build it once with
> `python -m games.wordvocab.build` — see [wordvocab](../wordvocab/README.md). Seeds are drawn
> from the shared vocab via the salted `endstart` split.

## Data model

`GameState` is what every endpoint and client method returns:

```jsonc
{
  "game_id": "uuid",
  "word": "mango",                  // the word whose last letter we match
  "options": ["river", "oasis", "tundra", "cliff", "marsh"],  // shuffled candidates
  "status": "correct",              // "in_progress" | "correct" | "incorrect"
  "submitted": "oasis",             // the raw answer step() scored (null before step)
  "solution": "oasis"               // the correct candidate; revealed only once status != in_progress
}
```

The canonical `<answer>` block (what the synthetic teacher writes and a human types):

```
<answer>oasis</answer>
```

`parse_action` reads the word inside the last `<answer>` tag (any case); a malformed or absent
answer scores `incorrect` — the same "malformed costs you the round" contract as Wordle.

## HTTP API

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `POST` | `/reset` | `{"mode":"train"\|"val","word":"mango|river,oasis,..."?}` | New episode; `word` pins an encoded `word1|options` target. |
| `POST` | `/step` | `{"game_id":"...","answer":"oasis"}` | Returns the scored terminal `GameState`. |
| `GET`  | `/state/{game_id}` | — | Observe without acting. |

`404` unknown `game_id`; `400` stepping an already-finished episode.

## Files

```
games/endstart/
├── game.py     # pure core: EndstartGame, EndstartBank, make_challenge/encode/score, models
├── render.py   # render_observation (the prompt) + render_answer (canonical block)
├── client.py   # EndstartClient Protocol + Local/HTTP transports (verb: step)
├── server.py   # async FastAPI: /reset /step /state
├── play.py     # terminal play (in-process or --http)
└── tests/      # test_game / test_render / test_client
```

## Training data

Endstart is **programmatic** (the label is cheap and exact, no reasoning wanted), so its SFT data
is produced by [`distillation/programmatic.py --game endstart`](../../distillation/README.md) —
not Claude. The built default emits **6,000 rows**, all valid by construction. See that module
for the generator and the unified dataset schema.
