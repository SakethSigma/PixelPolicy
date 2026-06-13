# Crossword fill

Word-skill game **#6** (see [`games/DATA_SOURCING.md`](../DATA_SOURCING.md)). A **single-turn**
environment: `reset()` poses a crossword clue and `step(answer)` scores the solved word and ends
the episode. It teaches **meaning + partial-pattern → word retrieval** (the core crossword
skill). This is a *reasoning* game distilled from Claude with **rejection sampling** — the env's
ground truth is the gate.

Same architecture as [Wordle](../wordle/README.md) and [charcount](../charcount/README.md): a
**pure core** that a human can play and an RL trainer can step in-process by the thousands,
through the same code path, so feedback and rules never drift. The env knows nothing about
models, tokens, or rewards.

## The clue

`reset()` poses three things, all derived deterministically from a single seed word:

- a **definition** — a WordNet gloss read from the committed meanings asset
  ([`games/wordvocab/meanings.jsonl`](../wordvocab/README.md));
- the **word length**;
- a **masked pattern** — about half the letters revealed in place, the rest hidden as `_`
  (e.g. `p _ r _ h`). The revealed positions are chosen by a **word-seeded RNG**, so the mask is
  fixed per word: pinning a seed word reconstructs the whole clue.

The seed word is **never** exposed in the in-progress `GameState` — only the clue is — so the
observation a model reads cannot leak the answer; the word is revealed only in the terminal
`solution`.

## Single-turn status convention

Mirrors Wordle's `"won"` so the distillation rejection filter stays game-agnostic:

```
"in_progress"  → clue posed, no answer yet
"correct"      → answer exactly equals the seed word (and matches the revealed letters)
"incorrect"    → answer wrong or unparseable
```

Ground truth is the **seed word**: `step(answer)` scores `correct` iff the parsed answer equals
the seed word and is consistent with the revealed letters of the pattern.

## Reasoning, `require_think`

This is a *reasoning* game distilled at **high adaptive-thinking effort**. The model opens
`<think>` via its chat template and reasons from the definition + revealed letters, then writes a
single `<answer>` — `<think>…</think><answer>word</answer>`. The rejection gate keeps a distilled
trace only when it is **correct AND contains a `<think>` block**. This is the
`GameSpec.require_think=True` flag, enforced in
[`distillation/batch_play.py`](../../distillation/README.md) (the same flag the anagram game
uses).

## Quick start

```bash
# play in the terminal (in-process, no server)
uv run --package game-crossword python -m games.crossword.play
uv run --package game-crossword python -m games.crossword.play --word crane

# run the HTTP server, then play the SAME game over HTTP
uv run --package game-crossword uvicorn games.crossword.server:app
uv run --package game-crossword python -m games.crossword.play --http http://127.0.0.1:8000

# tests (core scoring, render round-trip, Local/HTTP parity)
uv run --package game-crossword pytest games/crossword/tests/ -q
```

> Requires [`games/wordvocab/vocab.txt`](../wordvocab/README.md) (the shared word pool) and
> `games/wordvocab/meanings.jsonl` (the committed word→definition asset). Build them once with
> `python -m games.wordvocab.build` and `python -m games.wordvocab.build_meanings` — see
> [wordvocab](../wordvocab/README.md).

## Data model

`GameState` is what every endpoint and client method returns:

```jsonc
{
  "game_id": "uuid",
  "definition": "a hot drink made by infusing dried leaves in boiling water",
  "length": 5,                      // the seed word's length
  "pattern": "c_a_e",               // revealed letters + "_" for hidden positions
  "status": "correct",              // "in_progress" | "correct" | "incorrect"
  "submitted": "<think>...<answer>crane...", // the raw answer step() scored (null before step)
  "solution": {                     // revealed only once status != in_progress
    "word": "crane",
    "definition": "a hot drink made by infusing dried leaves in boiling water"
  }
}
```

The seed word lives **only** in the terminal `solution` — never in the in-progress state.

The canonical answer (what the teacher writes and a human types) is the solved word, lowercase,
inside a single `<answer>` tag (preceded by the model's `<think>` reasoning):

```
<answer>crane</answer>
```

`game.py::parse_answer` reads the body of the last `<answer>…</answer>` tag (or the last
alphabetic word of the reply if no tag is present); an answer with no parseable word scores
`incorrect` — the same "malformed costs you the round" contract as Wordle.

## HTTP API

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `POST` | `/reset` | `{"mode":"train"\|"val","word":"crane"?}` | New episode; `word` pins the seed word (the clue is derived from it), else one is sampled. |
| `POST` | `/step` | `{"game_id":"...","answer":"<answer>crane</answer>"}` | Returns the scored terminal `GameState`. |
| `GET`  | `/state/{game_id}` | — | Observe without acting. |

`404` unknown `game_id`; `400` stepping an already-finished episode.

## Word pool & split

`CrosswordBank` loads the committed meanings asset and builds clues from **two** word sources,
since every crossword seed needs a definition (only words carrying a WordNet gloss in
`meanings.jsonl` are eligible):

- the **Wordle vocabulary** (train + val union), and
- **general** multi-length words — the rest of the shared
  [`games/wordvocab/vocab.txt`](../wordvocab/README.md) (lengths 3–20), excluding the Wordle set.

`sample_targets(n)` draws **half from each pool**, so a run mixes familiar five-letter Wordle
words with varied-length general vocabulary (it is **not** Wordle-only). This is the cross-game
design in [`games/DATA_SOURCING.md`](../DATA_SOURCING.md): a word held out elsewhere is trained
here.

## Files

```
games/crossword/
├── game.py     # pure core: CrosswordGame, CrosswordBank, make_pattern/parse_answer, models
├── render.py   # render_observation (definition + length + spaced pattern) + render_answer
├── client.py   # CrosswordClient Protocol + Local/HTTP transports (verb: step)
├── server.py   # async FastAPI: /reset /step /state
├── play.py     # terminal play (in-process or --http; --word)
└── tests/      # test_game / test_render / test_client
```

## Training data

Crossword is **Claude-distilled** (reasoning is wanted), produced by
[`distillation/batch_play.py`](../../distillation/README.md) via the Anthropic Batch API with
rejection sampling:

```bash
uv run --package distillation python -m distillation.batch_play \
  --game crossword --episodes 1500 --model claude-sonnet-4-6 --effort high
```

The 1,500-episode run scored 1,415 correct; after the `require_think` gate (keep only traces that
are correct **and** carry a `<think>` block — 1,498/1,500 did) **1,415 valid rows** remain, split
750 Wordle-seed + 750 general-seed. Cost ≈ $4.96 (Batch API, 50% off). See that module for the
rejection filter and the unified dataset schema.
