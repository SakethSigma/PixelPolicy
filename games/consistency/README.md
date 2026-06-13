# Candidate consistency

Game **#12** (see [`games/DATA_SOURCING.md`](../DATA_SOURCING.md)). A **single-turn** environment:
`reset()` shows a Wordle board (1–3 past guesses + their ✓/-/x feedback) and a **candidate** word,
`step(answer)` scores a yes/no verdict and ends the episode. It teaches the **positive** side of
feedback reasoning — "is this word *still possible* given every clue?" — the filter a Wordle
player uses to narrow the answer set.

It is the complement of [`mistakeid`](../mistakeid/README.md): mistakeid teaches *locating* a
guess's errors, consistency teaches the binary *is-this-still-in-the-running* selection.

Same architecture as [Wordle](../wordle/README.md): a **pure core** that a human can play and an
RL trainer can step in-process by the thousands, through the same code path, so feedback and
rules never drift. The env knows nothing about models, tokens, or rewards.

## The challenge

You are shown a Wordle board — 1–3 past guesses, each scored per letter:

```
✓  the letter is correct and in the right position
-  the letter is in the word but in the wrong position
x  the letter is not in the word
```

…and a **candidate** word. Answer `yes` if the candidate is consistent with **every** clue
(keeps each `✓` in place, contains each `-` letter in a *different* position, and respects the
counts implied by `x`), `no` otherwise. Challenges are **balanced 50/50** yes/no, and kept small —
**under 4k tokens** (max ~250 tokens).

## Single-turn status convention

Mirrors Wordle's `"won"` so the distillation rejection filter stays game-agnostic:

```
"in_progress"  → board + candidate posed, no answer yet
"correct"      → the yes/no verdict matches whether the candidate is consistent   (the solved/"good" status)
"incorrect"    → wrong verdict (or unparseable)
```

Ground truth **reuses Wordle's own scorer** (a runtime dependency on `game-wordle`): a candidate
`c` is consistent with a row `(guess, fb)` iff `compute_feedback(guess, c) == fb` for **every**
row — which captures greens / yellows / greys including the duplicate rules exactly. The teacher
prefixes its verdict with a short, true **worded rationale** (see [Training data](#training-data));
no `<think>` is requested.

## Quick start

```bash
# play in the terminal (in-process, no server)
uv run --package game-consistency python -m games.consistency.play

# run the HTTP server, then play the SAME game over HTTP
uv run --package game-consistency uvicorn games.consistency.server:app
uv run --package game-consistency python -m games.consistency.play --http http://127.0.0.1:8000

# tests (core scoring, render round-trip, Local/HTTP parity) — 17 passing
uv run --package game-consistency pytest games/consistency/tests/ -q
```

> Requires `games/wordvocab/` (boards are built from the Wordle vocabulary, via `game-wordle`'s
> `WordBank`). The candidate and guesses are drawn from that vocab.

## Data model

`GameState` is what every endpoint and client method returns:

```jsonc
{
  "game_id": "uuid",
  "rows": [["CRANE", "xx-xx"], ["SLATE", "x-x-x"]],  // (guess, ✓/-/x feedback) — shown as symbols
  "candidate": "BLAND",
  "status": "correct",              // "in_progress" | "correct" | "incorrect"
  "submitted": "<answer>yes</answer>",  // the raw answer step() scored (null before step)
  "solution": true                 // whether the candidate is consistent; revealed only once status != in_progress
}
```

The canonical completion (what the synthetic teacher writes and a human may type): a short worded
rationale of the per-clue check, then the `<answer>` block. The minimal answer a human can type is
just the tag:

```
If the word were PLANT, guessing CRANE would score x x ✓ ✓ x, which matches the clue. Every clue is satisfied, so PLANT is still possible.
<answer>yes</answer>

<answer>no</answer>     # the bare tag is also accepted
```

`parse_action` reads the yes/no verdict inside the last `<answer>` tag (the leading rationale is
ignored when scoring); a malformed or absent answer scores `incorrect` — the same "malformed costs
you the round" contract as Wordle.

## HTTP API

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `POST` | `/reset` | `{"mode":"train"\|"val","word":"CRANE:xxgxx|...;BLAND"?}` | New episode; `word` pins an encoded `rows;candidate` target. |
| `POST` | `/step` | `{"game_id":"...","answer":"<answer>yes</answer>"}` | Returns the scored terminal `GameState`. |
| `GET`  | `/state/{game_id}` | — | Observe without acting. |

`404` unknown `game_id`; `400` stepping an already-finished episode.

## Challenge generation

`ConsistencyBank` builds each challenge from the Wordle vocabulary: pick a hidden target, score
1–3 random (non-target) guesses against it to form the board, then choose a candidate that is
consistent or inconsistent with that board, **50/50**.

## Files

```
games/consistency/
├── game.py     # pure core: ConsistencyGame, ConsistencyBank, is_consistent, encode/decode, models
├── render.py   # render_observation (the board + candidate prompt)
├── client.py   # ConsistencyClient Protocol + Local/HTTP transports (verb: step)
├── server.py   # async FastAPI: /reset /step /state
├── play.py     # terminal play (in-process or --http)
└── tests/      # test_game / test_render / test_client
```

## Training data

Consistency is **programmatic** (the label reuses Wordle's exact scorer, no Claude), so its SFT
data is produced by
[`distillation/programmatic.py --game consistency`](../../distillation/README.md). The built
default emits **10,000 rows** (5,000 yes + 5,000 no), all valid by construction. See that module
for the generator and the unified dataset schema.

Each completion begins with a short, **programmatically-generated worded rationale** that walks the
per-clue check, then the `<answer>` verdict. The rationale is produced by
`games/consistency/render.py::render_reasoning`, which recomputes the candidate's feedback for each
clue and, on the first failing clue, pinpoints the conflict (a green-slot mismatch, an absent
letter that is present, a yellow letter re-placed in the same slot, or a missing required letter):

```
If the word were PLANT, guessing CRANE would score x x ✓ ✓ x, which matches the clue. Every clue is satisfied, so PLANT is still possible.
<answer>yes</answer>

If the word were DOGGY, guessing CRANE would score x x x x x, but the clue shows x x ✓ ✓ x: position 3 must be A, but DOGGY has G there. So DOGGY is ruled out.
<answer>no</answer>
```

The rationale is **always true** — it is derived from the same `compute_feedback` computation as
the ground-truth label, and every generated row is self-checked (its parsed verdict must score
`correct`), so the dataset never teaches false reasoning. It is **templated/programmatic**, not
chain-of-thought: it is **not** wrapped in `<think>`, so the row's `has_think` stays `False`. The
agent system prompt now asks the model to briefly explain how it checked the clues before the final
`<answer>` tag.
