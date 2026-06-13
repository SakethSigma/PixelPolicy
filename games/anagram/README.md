# Anagrams

Word-skill game **#3** (see [`games/DATA_SOURCING.md`](../DATA_SOURCING.md)). A **single-turn**
environment: `reset()` poses two words and `step(answer)` scores a yes/no verdict and ends the
episode. It teaches **letter-multiset reasoning**. This is a *reasoning* game distilled from
Claude with **rejection sampling** — the env's ground truth is the gate.

Same architecture as [Wordle](../wordle/README.md) and [charcount](../charcount/README.md): a
**pure core** that a human can play and an RL trainer can step in-process by the thousands,
through the same code path, so feedback and rules never drift. The env knows nothing about
models, tokens, or rewards.

## Single-turn status convention

Mirrors Wordle's `"won"` so the distillation rejection filter stays game-agnostic:

```
"in_progress"  → challenge posed, no answer yet
"correct"      → verdict matched the multiset check   (the solved/"good" status)
"incorrect"    → verdict wrong or unparseable
```

Ground truth is pure Python: two words are anagrams iff their sorted letter multisets are equal
(`sorted(w1) == sorted(w2)`).

## Reasoning, `require_think`

This is a *reasoning* game distilled at **high adaptive-thinking effort**. The agent's system
prompt asks the model to **"think it through"** and deliberately does **not** tell it *how* to
decide (e.g. by sorting letters) — it must work out the letter-multiset comparison itself, which
is the skill we distil. The model opens `<think>` via its chat template and reasons before the
final `<answer>` — `<think>…</think><answer>yes|no</answer>`. The rejection gate keeps a distilled
trace only when it is **correct AND contains a `<think>` block**. This is the
`GameSpec.require_think=True` flag, enforced in
[`distillation/batch_play.py`](../../distillation/README.md) (the same flag the crossword game
uses).

## Quick start

```bash
# play in the terminal (in-process, no server)
uv run --package game-anagram python -m games.anagram.play                 # a sampled pair
uv run --package game-anagram python -m games.anagram.play --pair listen,silent

# run the HTTP server, then play the SAME game over HTTP
uv run --package game-anagram uvicorn games.anagram.server:app
uv run --package game-anagram python -m games.anagram.play --http http://127.0.0.1:8000

# tests (core scoring, render round-trip, Local/HTTP parity)
uv run --package game-anagram pytest games/anagram/tests/ -q
```

> Requires [`games/wordvocab/vocab.txt`](../wordvocab/README.md) (the shared word pool). Build it
> once with `python -m games.wordvocab.build` — see [wordvocab](../wordvocab/README.md).

## Data model

`GameState` is what every endpoint and client method returns:

```jsonc
{
  "game_id": "uuid",
  "word1": "listen",
  "word2": "silent",
  "status": "correct",              // "in_progress" | "correct" | "incorrect"
  "submitted": "<answer>yes...",    // the raw answer step() scored (null before step)
  "solution": {                     // revealed only once status != in_progress
    "are_anagrams": true
  }
}
```

The canonical answer (what the teacher writes and a human types) is `yes` or `no`, inside a
single `<answer>` tag:

```
<answer>yes</answer>
```

`game.py::parse_answer` reads the **last** yes/no token out of the reply; an answer with no
yes/no scores `incorrect` — the same "malformed costs you the round" contract as Wordle.

## HTTP API

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `POST` | `/reset` | `{"mode":"train"\|"val","word":"listen,silent"?}` | New episode; `word` is a `"w1,w2"` pair that pins the challenge, else a pair is sampled. |
| `POST` | `/step` | `{"game_id":"...","answer":"<answer>yes</answer>"}` | Returns the scored terminal `GameState`. |
| `GET`  | `/state/{game_id}` | — | Observe without acting. |

`404` unknown `game_id`; `400` stepping an already-finished episode.

## Word pool & split

`AnagramBank` loads the **full multi-length vocabulary**
([`games/wordvocab/vocab.txt`](../wordvocab/README.md), 85,909 words — general words, not
Wordle-only) and derives its own train/val split via the salted rule `assign_pool("anagram",
word)`. Within each pool it indexes anagram groups (sorted-signature buckets of size ≥ 2) for
positives and by-length lists for negatives. Generated pairs are a **40/60 positive/negative**
mix: positives come from the signature groups; of the negatives, most are **hard** same-length
near-misses (highest letter overlap that is not an anagram) with some easy different-length pairs
mixed in, so the model can't shortcut on length or letter set alone.

## Files

```
games/anagram/
├── game.py     # pure core: AnagramGame, AnagramBank, signature/are_anagrams, pair construction
├── render.py   # render_observation (the prompt) + render_answer (yes/no)
├── client.py   # AnagramClient Protocol + Local/HTTP transports (verb: step)
├── server.py   # async FastAPI: /reset /step /state
├── play.py     # terminal play (in-process or --http; --pair w1,w2)
└── tests/      # test_game / test_render / test_client
```

## Training data

Anagrams is **Claude-distilled** (reasoning is wanted), produced by
[`distillation/batch_play.py`](../../distillation/README.md) via the Anthropic Batch API with
rejection sampling:

```bash
uv run --package distillation python -m distillation.batch_play \
  --game anagram --episodes 1000 --model claude-sonnet-4-6 --effort high
```

The 1,000-episode run (seed 0, high effort) scored all 1,000 correct; after the `require_think`
gate (keep only correct traces that carry a `<think>` block — 68 had none and were dropped)
**932 valid rows** remain. Cost ≈ $1.78 (Batch API, 50% off). See that module for the rejection
filter and the unified dataset schema.
