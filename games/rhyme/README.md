# Rhymes

Word-skill game **#5** (see [`games/DATA_SOURCING.md`](../DATA_SOURCING.md)). A **single-turn**
environment with two variants: `reset()` poses a word (and, for MCQ, five options) and
`step(answer)` scores the reply and ends the episode. It teaches the model the **sound /
phonetic mapping** of words.

Same architecture as [Wordle](../wordle/README.md) and [charcount](../charcount/README.md): a
**pure core** that a human can play and an RL trainer can step in-process by the thousands,
through the same code path, so feedback and rules never drift. The env knows nothing about
models, tokens, or rewards.

## Variants

- **`free`** (default) — "name a word that rhymes with X". Correct = any member of the rhyme set.
- **`mcq`** — a word + 5 options, exactly one of which rhymes. Correct = pick that option.

## Single-turn status convention

Mirrors Wordle's `"won"` so the distillation rejection filter stays game-agnostic:

```
"in_progress"  → challenge posed, no answer yet
"correct"      → the answer rhymes (and, for MCQ, is one of the options)
"incorrect"    → the answer did not rhyme (or was unparseable / not an option)
```

Ground truth is the **CMU Pronouncing Dictionary** via the `pronouncing` library (bundled,
offline — no download): a word rhymes with `word` iff it is in `pronouncing.rhymes(word)`.
Options and gold answers are restricted to plain alphabetic words so generated challenges parse
cleanly.

## Quick start

```bash
# play in the terminal (in-process, no server)
uv run --package game-rhyme python -m games.rhyme.play                 # free variant
uv run --package game-rhyme python -m games.rhyme.play --variant mcq   # multiple-choice
uv run --package game-rhyme python -m games.rhyme.play --word bright

# run the HTTP server, then play the SAME game over HTTP
uv run --package game-rhyme uvicorn games.rhyme.server:app
uv run --package game-rhyme python -m games.rhyme.play --http http://127.0.0.1:8000

# tests (core scoring, render round-trip, Local/HTTP parity)
uv run --package game-rhyme pytest games/rhyme/tests/ -q
```

> Requires [`games/wordvocab/vocab.txt`](../wordvocab/README.md) (the shared word pool) and the
> bundled CMU dict shipped with the `pronouncing` dependency (offline, no download). Build the
> vocab once with `python -m games.wordvocab.build` — see [wordvocab](../wordvocab/README.md).

## Data model

`GameState` is what every endpoint and client method returns:

```jsonc
{
  "game_id": "uuid",
  "word": "bright",                 // the word to rhyme with
  "variant": "mcq",                 // "free" | "mcq"
  "options": ["table","flight",...],// MCQ only: the five shuffled choices (null for free)
  "status": "correct",              // "in_progress" | "correct" | "incorrect"
  "submitted": "<answer>flight...", // the raw answer step() scored (null before step)
  "solution": {                     // revealed only once status != in_progress
    "variant": "mcq",
    "correct_option": "flight",     // MCQ: the one rhyming option
    "examples": []                  // free: a few accepted rhymes
  }
}
```

The canonical `<answer>` block is just the chosen/answer word, lowercase:

```
<answer>flight</answer>
```

`game.py::parse_answer` reads the body of the last `<answer>…</answer>` tag (or the last word of
the reply if no tag is present); an answer with no parseable word scores `incorrect` — the same
"malformed costs you the round" contract as Wordle.

## HTTP API

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `POST` | `/reset` | `{"mode":"train"\|"val","word":"bright"?,"variant":"free"\|"mcq"?}` | New episode; `word` pins the challenge, `variant` selects free vs MCQ. |
| `POST` | `/step` | `{"game_id":"...","answer":"<answer>flight</answer>"}` | Returns the scored terminal `GameState`. |
| `GET`  | `/state/{game_id}` | — | Observe without acting. |

`404` unknown `game_id`; `400` stepping an already-finished episode.

## Word pool & split

`RhymeBank` loads the **full multi-length vocabulary**
([`games/wordvocab/vocab.txt`](../wordvocab/README.md), 85,909 words — general words, not
Wordle-only) and derives its own train/val split via the salted rule `assign_pool("rhyme",
word)`. Only words with at least one clean (alphabetic) rhyme are usable as challenge seeds. The
salt decouples this split from Wordle's, so a word held out here is trained in another game (the
cross-game design in [`games/DATA_SOURCING.md`](../DATA_SOURCING.md)).

## Files

```
games/rhyme/
├── game.py     # pure core: RhymeGame, RhymeBank, rhymes/alpha_rhymes, MCQ option building
├── render.py   # render_observation (free or MCQ prompt) + render_answer (the answer word)
├── client.py   # RhymeClient Protocol + Local/HTTP transports (verb: step)
├── server.py   # async FastAPI: /reset /step /state
├── play.py     # terminal play (in-process or --http; --variant free|mcq, --word)
└── tests/      # test_game / test_render / test_client
```

## Training data

Rhymes is **programmatic** (the label is cheap and exact, no reasoning wanted), so its SFT data
is produced by [`distillation/programmatic.py --game rhyme`](../../distillation/README.md) — not
Claude. The default run emits **10,000 rows** = 5,000 MCQ + 5,000 free, each self-checked through
the env. See that module for the generator and the unified dataset schema.
