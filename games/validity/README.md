# Validity + meaning

Word-skill game **#2** (see [`games/DATA_SOURCING.md`](../DATA_SOURCING.md)). A **single-turn**
environment: `reset()` poses one word — a real word or a generated pseudo-word — and
`step(answer)` scores the verdict (`valid`/`invalid`, plus a meaning when valid) and ends the
episode. It teaches the model **vocabulary membership** and **meaning recall**.

Same architecture as [Wordle](../wordle/README.md) and [charcount](../charcount/README.md): a
**pure core** that a human can play and an RL trainer can step in-process by the thousands,
through the same code path, so feedback and rules never drift. The env knows nothing about
models, tokens, or rewards.

## Single-turn status convention

Mirrors Wordle's `"won"` so the distillation rejection filter stays game-agnostic:

```
"in_progress"  → challenge posed, no answer yet
"correct"      → verdict matched membership (and, when valid, a non-empty meaning was given)
"incorrect"    → verdict wrong, missing meaning for a valid word, or unparseable
```

Ground truth comes from the committed **meanings asset**
([`games/wordvocab/meanings.jsonl`](../wordvocab/README.md), built once from WordNet), so there
is **no `nltk` at runtime**: a word is *valid* iff it carries a WordNet definition, and that
definition is the gold `<meaning>`. The meaning is checked **loosely** (non-empty) — the model
must recall *a* gloss, not reproduce WordNet's exact wording.

## Quick start

```bash
# play in the terminal (in-process, no server)
uv run --package game-validity python -m games.validity.play               # a real word
uv run --package game-validity python -m games.validity.play --kind invalid # a pseudo-word
uv run --package game-validity python -m games.validity.play --word kindle

# run the HTTP server, then play the SAME game over HTTP
uv run --package game-validity uvicorn games.validity.server:app
uv run --package game-validity python -m games.validity.play --http http://127.0.0.1:8000

# tests (core scoring, render round-trip, Local/HTTP parity)
uv run --package game-validity pytest games/validity/tests/ -q
```

> Requires [`games/wordvocab/meanings.jsonl`](../wordvocab/README.md) (the validity oracle) and,
> via the Wordle vocabulary, `games/wordle/{train,val}_words.txt`. Build `meanings.jsonl` once
> with `python -m games.wordvocab.build_meanings` — see [wordvocab](../wordvocab/README.md).

## Data model

`GameState` is what every endpoint and client method returns:

```jsonc
{
  "game_id": "uuid",
  "word": "kindle",                 // the challenge the model judges
  "status": "correct",              // "in_progress" | "correct" | "incorrect"
  "submitted": "<answer>valid...",  // the raw answer step() scored (null before step)
  "solution": {                     // revealed only once status != in_progress
    "valid": true,
    "meaning": "a fire that has been kindled or is burning"  // null for invalid words
  }
}
```

The canonical answer (what the synthetic teacher writes and a human types back tolerantly):

```
<answer>valid</answer>
<meaning>a fire that has been kindled or is burning</meaning>
```

```
<answer>invalid</answer>
```

`game.py::parse_answer` reads this back tolerantly: it takes the verdict from the `<answer>` tag
(never from the meaning, whose gloss may itself contain the word "invalid", e.g. *annul* →
"declare invalid"), checks `invalid` before `valid` (since one contains the other), and accepts
a `<meaning>` block when present. A malformed answer scores `incorrect` — the same
"malformed costs you the round" contract as Wordle.

## HTTP API

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `POST` | `/reset` | `{"mode":"train"\|"val","word":"kindle"?,"kind":"valid"\|"invalid"?}` | New episode; `word` pins the challenge, else `kind` picks a real word or a synthesized pseudo-word. |
| `POST` | `/step` | `{"game_id":"...","answer":"<answer>valid</answer>..."}` | Returns the scored terminal `GameState`. |
| `GET`  | `/state/{game_id}` | — | Observe without acting. |

`404` unknown `game_id`; `400` stepping an already-finished episode.

## Word pool & split

Unlike the other word-skill games, validity draws its word universe from the **Wordle
vocabulary** (`train_words.txt` + `val_words.txt` union, 12,972 words) — *not* the full
multi-length vocab — so the model learns the spelling and meaning of every Wordle word. Only the
**6,627** Wordle words that carry a WordNet definition can be *valid* challenges; `ValidityBank`
exposes those as `valid_words` and splits them with the salted rule `assign_pool("validity",
word)`. The salt decouples this split from Wordle's own, so **Wordle's held-out val words
deliberately enter validity training** — meaning recall is a different skill, not Wordle-eval
leakage. Pseudo-words (the *invalid* challenges) are built by perturbing a real Wordle word
(one swap/insert/delete) and are kept only once confirmed absent from both WordNet *and* the
Wordle vocab, so the "invalid" label is trustworthy.

## Files

```
games/validity/
├── game.py     # pure core: ValidityGame, ValidityBank, parse_answer/perturb, the validity oracle
├── render.py   # render_observation (the prompt) + render_answer (canonical block)
├── client.py   # ValidityClient Protocol + Local/HTTP transports (verb: step)
├── server.py   # async FastAPI: /reset /step /state
├── play.py     # terminal play (in-process or --http; --kind valid|invalid, --word)
└── tests/      # test_game / test_render / test_client
```

## Training data

Validity is **programmatic** (the label is cheap and exact, no reasoning wanted), so its SFT
data is produced by [`distillation/programmatic.py --game validity`](../../distillation/README.md)
— not Claude. The default run emits **13,254 rows** = 6,627 valid + 6,627 invalid (balanced
50/50), each self-checked through the env. See that module for the generator and the unified
dataset schema.
