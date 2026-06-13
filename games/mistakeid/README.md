# Wordle mistake identification

Word-skill game **#8** (see [`games/DATA_SOURCING.md`](../DATA_SOURCING.md)). A **single-turn**
environment: `reset()` poses a Wordle board (past guesses + their per-letter feedback) and a
**proposed next guess**, and `step(answer)` scores whether the player correctly identified the
*repeated mistakes* in that proposed guess, then ends the episode. It teaches the model to read
Wordle feedback and notice when a guess throws away information. This is a *reasoning* game
distilled from Claude with **rejection sampling** — the env's ground truth is the gate.

Same architecture as [Wordle](../wordle/README.md) and [charcount](../charcount/README.md): a
**pure core** that a human can play and an RL trainer can step in-process by the thousands,
through the same code path, so feedback and rules never drift. The env knows nothing about
models, tokens, or rewards.

## What counts as a mistake

Given the board's feedback, exactly two error kinds count as *repeated mistakes* in the proposed
guess:

- a **grey** mistake — the guess reuses a letter already proven absent (a letter that was only
  ever marked grey `x`, never green/yellow → truly not in the word);
- a **yellow** mistake — the guess re-places a letter in a slot already shown yellow (`-`) for it
  (the letter is in the word, but known *not* to go in that position).

Nothing else counts. The board renders feedback as `✓` (green), `-` (yellow), `x` (grey) tiles.

## Single-turn status convention

Mirrors Wordle's `"won"` so the distillation rejection filter stays game-agnostic:

```
"in_progress"  → board + proposed guess posed, no answer yet
"correct"      → reported mistakes (and the yes/no flag) match the truth   (the solved/"good" status)
"incorrect"    → reported set wrong, flag wrong, or answer unparseable
```

Ground truth needs **no target word**: the env computes the true error set from the board feedback
alone. `step(answer)` parses the reported `{(position, letter, grey|yellow)}` set plus a yes/no
flag and scores `correct` iff they exactly match the computed truth.

## Reasoning, `require_think`

This is a *reasoning* game distilled at **`max`** adaptive-thinking effort. The model opens
`<think>` via its chat template, reasons over the feedback, then writes a single `<answer>`. The
rejection gate keeps a distilled trace only when it is **correct AND contains a `<think>` block**.
This is the `GameSpec.require_think=True` flag, enforced in
[`distillation/batch_play.py`](../../distillation/README.md) (the same flag the anagram and
crossword games use).

> `max` is the top adaptive-thinking effort. The valid levels for `claude-sonnet-4-6` are
> `low` / `medium` / `high` / `max` — `xhigh` is **not** a supported level for this model.

## Quick start

```bash
# play in the terminal (in-process, no server)
uv run --package game-mistakeid python -m games.mistakeid.play

# run the HTTP server, then play the SAME game over HTTP
uv run --package game-mistakeid uvicorn games.mistakeid.server:app
uv run --package game-mistakeid python -m games.mistakeid.play --http http://127.0.0.1:8000

# tests (core scoring, render round-trip, Local/HTTP parity)
uv run --package game-mistakeid pytest games/mistakeid/tests/ -q
```

> Self-contained: the game reads its committed `games/mistakeid/challenges.jsonl` asset and has
> **no** `games/wordvocab` or distillation dependency at runtime.

## Data model

`GameState` is what every endpoint and client method returns:

```jsonc
{
  "game_id": "uuid",
  "rounds": [["crane", "xxgxx"], ["..."]], // (guess, feedback) for each past round
  "attempt": "trace",                       // the proposed next guess to review
  "status": "correct",                      // "in_progress" | "correct" | "incorrect"
  "submitted": "<think>...<answer>...",     // the raw answer step() scored (null before step)
  "solution": {                             // revealed only once status != in_progress
    "has_mistakes": true,
    "errors": [{"position": 4, "letter": "R", "kind": "grey"}]
  }
}
```

The canonical answer (what the teacher writes and a human types) is a `mistakes:` flag, then one
line per error, inside a single `<answer>` tag (preceded by the model's `<think>` reasoning) —
positions are 1-based and letters UPPERCASE:

```
<answer>
mistakes: yes
position 4, letter R, grey
position 1, letter A, yellow
</answer>
```

When the proposed guess repeats no mistake, the answer is just `mistakes: no`. `game.py::parse_report`
reads this format back tolerantly (spacing/case, `gray` spelling); an answer with no `mistakes:`
flag scores `incorrect` — the same "malformed costs you the round" contract as Wordle.

## HTTP API

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `POST` | `/reset` | `{"mode":"train"\|"val"}` | New episode; samples a board + proposed guess. |
| `POST` | `/step` | `{"game_id":"...","answer":"<answer>mistakes: yes\n..."}` | Returns the scored terminal `GameState`. |
| `GET`  | `/state/{game_id}` | — | Observe without acting. |

`404` unknown `game_id`; `400` stepping an already-finished episode.

## Challenge source

Challenges are not generated from vocabulary; they are extracted from the **original Wordle teacher
trajectories** into the committed `games/mistakeid/challenges.jsonl` asset, built once by
[`build_challenges.py`](build_challenges.py). The bank holds **165 mistake boards** (a proposed
guess that repeats a grey/yellow mistake) and **1,498 clean boards**. `MistakeBank.sample_targets`
returns a **50/50 mistake/clean mix** (capped by the available mistake supply), so a balanced run
maxes out at 165 + 165 = **330** episodes.

## Files

```
games/mistakeid/
├── game.py            # pure core: MistakeGame, MistakeBank, score_feedback/true_errors/parse_report, models
├── render.py          # render_observation (board with ✓/-/x tiles + proposed guess) + render_answer
├── client.py          # MistakeClient Protocol + Local/HTTP transports (verb: step)
├── server.py          # async FastAPI: /reset /step /state
├── play.py            # terminal play (in-process or --http)
├── build_challenges.py # one-time extractor: Wordle teacher trajectories → challenges.jsonl
├── challenges.jsonl   # committed challenge asset (165 mistake + 1,498 clean boards)
└── tests/             # test_game / test_render / test_client
```

## Training data

Mistake identification is **Claude-distilled** (reasoning is wanted), produced by
[`distillation/batch_play.py`](../../distillation/README.md) via the Anthropic Batch API with
rejection sampling:

```bash
uv run --package distillation python -m distillation.batch_play \
  --game mistakeid --episodes 330 --model claude-sonnet-4-6 --effort max
```

The 330-episode run (165 mistake + 165 clean) yielded **317 valid rows** (157 mistake + 160 clean)
after the `require_think` gate dropped the 13 wrong traces. Cost ≈ $2.78 (Batch API, 50% off). See
that module for the rejection filter and the unified dataset schema.
